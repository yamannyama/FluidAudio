import CoreML
import Foundation

/// Audio that is still being recorded: mono samples at the diarizer's sample rate, appended as they
/// arrive and kept in a raw file so a long recording does not sit in memory.
@available(macOS 14.0, iOS 17.0, *)
public final class GrowingAudioSampleSource: AudioSampleSource, @unchecked Sendable {
    public let fileURL: URL
    private let descriptor: Int32
    private let lock = NSLock()
    private var count = 0
    private var closed = false

    public init(fileURL: URL) throws {
        self.fileURL = fileURL
        try? FileManager.default.removeItem(at: fileURL)
        descriptor = open(fileURL.path, O_RDWR | O_CREAT | O_TRUNC, 0o600)
        guard descriptor >= 0 else {
            throw OfflineDiarizationError.processingFailed("Could not create \(fileURL.lastPathComponent) (errno \(errno))")
        }
    }

    deinit { if !closed { close(descriptor) } }

    public var sampleCount: Int { lock.withLock { count } }

    /// Appends samples. Safe to call from the capture thread while another thread reads.
    public func append(_ samples: UnsafeBufferPointer<Float>) {
        guard let base = samples.baseAddress, samples.count > 0 else { return }
        lock.lock()
        defer { lock.unlock() }
        guard !closed else { return }
        let bytes = samples.count * MemoryLayout<Float>.stride
        let written = pwrite(descriptor, base, bytes, off_t(count * MemoryLayout<Float>.stride))
        if written == bytes { count += samples.count }
    }

    public func copySamples(into destination: UnsafeMutablePointer<Float>, offset: Int, count: Int) throws {
        // Like ArrayAudioSampleSource: copy what exists and leave the rest of the destination as it is.
        let total = sampleCount
        let start = max(0, offset)
        let count = min(count, total - start)
        guard count > 0 else { return }
        let offset = start
        let bytes = count * MemoryLayout<Float>.stride
        var done = 0
        while done < bytes {
            let n = pread(descriptor, UnsafeMutableRawPointer(destination).advanced(by: done), bytes - done,
                          off_t(offset * MemoryLayout<Float>.stride + done))
            guard n > 0 else { throw OfflineDiarizationError.processingFailed("Growing source read failed (errno \(errno))") }
            done += n
        }
    }

    /// Closes and deletes the backing file.
    public func cleanup() {
        lock.lock()
        if !closed { closed = true; close(descriptor) }
        lock.unlock()
        try? FileManager.default.removeItem(at: fileURL)
    }
}

@available(macOS 14.0, iOS 17.0, *)
private struct SlicedAudioSampleSource: AudioSampleSource {
    let base: AudioSampleSource
    let start: Int
    let sampleCount: Int
    func copySamples(into destination: UnsafeMutablePointer<Float>, offset: Int, count: Int) throws {
        try base.copySamples(into: destination, offset: start + offset, count: count)
    }
}

/// Runs segmentation and embedding extraction while the audio is still growing, so that only clustering is
/// left when the recording ends. Each segmentation window and its embeddings depend on that window alone,
/// so the result equals `OfflineDiarizerManager.prepare` over the finished audio.
///
/// Call `advance()` from time to time while recording, then `finish()` once, then `manager.cluster(_:)`.
/// Not safe for concurrent calls: use from one task.
@available(macOS 14.0, iOS 17.0, *)
public final class LiveDiarizationPreparer: @unchecked Sendable {
    private let source: GrowingAudioSampleSource
    private let config: OfflineDiarizerConfig
    private let models: OfflineDiarizerModels
    private let extractor: OfflineEmbeddingExtractor
    private var logProbs: [[[Float]]] = []
    private var speakerWeights: [[[Float]]] = []
    private var chunkOffsets: [Double] = []
    private var numFrames = 0
    private var numSpeakers = 0
    private var frameDuration: Double = 0
    private var embeddings: [TimedEmbedding] = []
    private var segmentationSeconds: TimeInterval = 0
    private var embeddingSeconds: TimeInterval = 0
    private var finished = false

    /// Chunks (segmentation windows) prepared so far.
    public var preparedChunkCount: Int { chunkOffsets.count }
    /// Seconds of audio covered by the prepared chunks.
    public var preparedSeconds: Double { Double(chunkOffsets.count * config.samplesPerStep) / Double(config.segmentation.sampleRate) }
    /// Model time spent so far.
    public var workSeconds: TimeInterval { segmentationSeconds + embeddingSeconds }

    public init(manager: OfflineDiarizerManager, source: GrowingAudioSampleSource) throws {
        guard let models = manager.livePreparationModels else { throw OfflineDiarizationError.modelNotLoaded("offline-diarizer") }
        let config = manager.livePreparationConfig
        try config.validate()
        self.source = source
        self.config = config
        self.models = models
        extractor = OfflineEmbeddingExtractor(
            fbankModel: models.fbankModel,
            embeddingModel: models.embeddingModel,
            pldaTransform: PLDATransform(pldaRhoModel: models.pldaRhoModel, psi: models.pldaPsi),
            config: config
        )
    }

    /// Prepares the windows that are already complete. Does nothing until at least `minimumNewChunks` are ready.
    /// - Returns: the number of chunks added.
    @discardableResult
    public func advance(minimumNewChunks: Int = 28) async throws -> Int {
        try await run(final: false, minimumNewChunks: minimumNewChunks)
    }

    /// Prepares the rest (including the zero-padded tail windows) and returns the value for `cluster(_:)`.
    public func finish() async throws -> PreparedDiarization {
        let began = Date()
        try await run(final: true, minimumNewChunks: 1)
        finished = true
        let segmentation = SegmentationOutput(
            logProbs: logProbs, speakerWeights: speakerWeights, numChunks: chunkOffsets.count, numFrames: numFrames,
            numSpeakers: numSpeakers, chunkOffsets: chunkOffsets, frameDuration: frameDuration)
        return PreparedDiarization(
            audioSource: source, segmentation: segmentation, timedEmbeddings: embeddings, audioLoadingSeconds: 0,
            segmentationSeconds: segmentationSeconds, embeddingExtractionSeconds: embeddingSeconds,
            prepareWallSeconds: Date().timeIntervalSince(began))
    }

    @discardableResult
    private func run(final: Bool, minimumNewChunks: Int) async throws -> Int {
        guard !finished else { return 0 }
        let total = source.sampleCount
        let window = config.samplesPerWindow, step = config.samplesPerStep
        let done = chunkOffsets.count
        let blockStart = done * step
        guard total > blockStart else {
            if final, done == 0 { throw OfflineDiarizationError.noSpeechDetected }
            return 0
        }
        // Windows that lie fully inside the audio so far. The tail windows (zero-padded) wait for `finish()`.
        let complete = total >= window ? (total - window) / step + 1 : 0
        let keep: Int?
        if final {
            keep = nil
        } else {
            guard complete - done >= max(1, minimumNewChunks) else { return 0 }
            keep = complete - done
        }
        let sliceCount = final ? total - blockStart : (complete - 1) * step + window - blockStart
        let slice = SlicedAudioSampleSource(base: source, start: blockStart, sampleCount: sliceCount)

        let segmentationStart = Date()
        let output = try await OfflineSegmentationProcessor().process(
            audioSource: slice, segmentationModel: models.segmentationModel, config: config, chunkHandler: nil)
        segmentationSeconds += Date().timeIntervalSince(segmentationStart)

        let kept = min(keep ?? output.numChunks, output.numChunks)
        guard kept > 0 else { return 0 }
        let offsetSeconds = Double(blockStart) / Double(config.segmentation.sampleRate)
        var chunks: [SegmentationChunk] = []
        chunks.reserveCapacity(kept)
        for local in 0..<kept {
            let weights = output.speakerWeights.indices.contains(local) ? output.speakerWeights[local] : []
            let probs = output.logProbs.indices.contains(local) ? output.logProbs[local] : []
            let offset = (output.chunkOffsets.indices.contains(local) ? output.chunkOffsets[local] : Double(local * step) / Double(config.segmentation.sampleRate)) + offsetSeconds
            logProbs.append(probs)
            speakerWeights.append(weights)
            chunkOffsets.append(offset)
            guard !weights.isEmpty else { continue }
            chunks.append(SegmentationChunk(chunkIndex: done + local, chunkOffsetSeconds: offset, frameDuration: output.frameDuration,
                                            logProbs: probs, speakerWeights: weights))
        }
        numFrames = output.numFrames
        numSpeakers = output.numSpeakers
        frameDuration = output.frameDuration

        let embeddingStart = Date()
        let pending = chunks
        let stream = AsyncThrowingStream<SegmentationChunk, Error> { continuation in
            for chunk in pending { continuation.yield(chunk) }
            continuation.finish()
        }
        embeddings += try await extractor.extractEmbeddings(audioSource: source, segmentationStream: stream)
        embeddingSeconds += Date().timeIntervalSince(embeddingStart)
        return kept
    }
}
