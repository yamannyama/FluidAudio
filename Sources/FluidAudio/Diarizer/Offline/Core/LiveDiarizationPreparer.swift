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
    private let source: AudioSampleSource
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
    /// Chunks and embeddings already written by `saveCheckpoint(to:)`.
    private var savedChunks = 0
    private var savedEmbeddings = 0
    private var checkpointChunkBytes = 0
    private var checkpointEmbeddingBytes = 0

    /// Chunks (segmentation windows) prepared so far.
    public var preparedChunkCount: Int { chunkOffsets.count }
    /// Seconds of audio covered by the prepared chunks.
    public var preparedSeconds: Double { Double(chunkOffsets.count * config.samplesPerStep) / Double(config.segmentation.sampleRate) }
    /// Model time spent so far.
    public var workSeconds: TimeInterval { segmentationSeconds + embeddingSeconds }

    /// Seconds of audio in the source (for a finished file, the whole length).
    public var totalChunkEstimate: Int { max(1, (source.sampleCount + config.samplesPerStep - 1) / config.samplesPerStep) }

    /// `source` may be still growing (`GrowingAudioSampleSource`) or a finished file. For a finished file, call
    /// `advance(minimumNewChunks: 1, maximumNewChunks:)` in batches and `saveCheckpoint(to:)` between them, so that an
    /// interrupted run can continue with `init(manager:source:checkpoint:)`.
    public init(manager: OfflineDiarizerManager, source: AudioSampleSource) throws {
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
    public func advance(minimumNewChunks: Int = 28, maximumNewChunks: Int? = nil) async throws -> Int {
        try await run(final: false, minimumNewChunks: minimumNewChunks, maximumNewChunks: maximumNewChunks)
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
    private func run(final: Bool, minimumNewChunks: Int, maximumNewChunks: Int? = nil) async throws -> Int {
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
            keep = min(complete - done, max(1, maximumNewChunks ?? Int.max))
        }
        let sliceCount = final ? total - blockStart : (keep! - 1) * step + window
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

// MARK: - Checkpoint

/// Saves and restores the prepared chunks, so a run that was stopped (app suspended, process killed) continues
/// from the last saved chunk instead of the start. Files are append-only: each save writes only the chunks and
/// embeddings added since the previous save, then replaces a small `meta.json` that says how much is valid.
@available(macOS 14.0, iOS 17.0, *)
extension LiveDiarizationPreparer {
    private struct CheckpointMeta: Codable {
        var version = 1
        var sampleCount: Int
        var samplesPerStep: Int
        var chunks: Int
        var embeddings: Int
        var chunkBytes: Int
        var embeddingBytes: Int
        var numFrames: Int
        var numSpeakers: Int
        var frameDuration: Double
        var segmentationSeconds: Double
        var embeddingSeconds: Double
    }

    /// Number of chunks a checkpoint in `directory` holds for this source, or 0 if there is none that fits.
    public static func checkpointChunkCount(in directory: URL, sampleCount: Int) -> Int {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("meta.json")),
              let meta = try? JSONDecoder().decode(CheckpointMeta.self, from: data), meta.sampleCount == sampleCount else { return 0 }
        return meta.chunks
    }

    /// Restores from `directory` when it holds a checkpoint for the same audio length and step. Otherwise starts empty.
    public convenience init(manager: OfflineDiarizerManager, source: AudioSampleSource, checkpoint directory: URL) throws {
        try self.init(manager: manager, source: source)
        restore(from: directory)
    }

    private func restore(from directory: URL) {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("meta.json")),
              let meta = try? JSONDecoder().decode(CheckpointMeta.self, from: data),
              meta.version == 1, meta.sampleCount == source.sampleCount, meta.samplesPerStep == config.samplesPerStep,
              let chunkData = try? Data(contentsOf: directory.appendingPathComponent("chunks.bin")), chunkData.count >= meta.chunkBytes,
              let embeddingData = try? Data(contentsOf: directory.appendingPathComponent("embeddings.bin")), embeddingData.count >= meta.embeddingBytes
        else { return }
        var reader = CheckpointReader(data: chunkData.prefix(meta.chunkBytes))
        var probsList: [[[Float]]] = [], weightsList: [[[Float]]] = [], offsets: [Double] = []
        for _ in 0..<meta.chunks {
            guard let offset = reader.double(), let probs = reader.matrix(), let weights = reader.matrix() else { return }
            offsets.append(offset); probsList.append(probs); weightsList.append(weights)
        }
        var embeddingReader = CheckpointReader(data: embeddingData.prefix(meta.embeddingBytes))
        var restored: [TimedEmbedding] = []
        for _ in 0..<meta.embeddings {
            guard let chunkIndex = embeddingReader.int(), let speakerIndex = embeddingReader.int(),
                  let startFrame = embeddingReader.int(), let endFrame = embeddingReader.int(),
                  let frameWeights = embeddingReader.floats(), let startTime = embeddingReader.double(), let endTime = embeddingReader.double(),
                  let embedding256 = embeddingReader.floats(), let rho128 = embeddingReader.doubles() else { return }
            restored.append(TimedEmbedding(chunkIndex: chunkIndex, speakerIndex: speakerIndex, startFrame: startFrame, endFrame: endFrame,
                                           frameWeights: frameWeights, startTime: startTime, endTime: endTime,
                                           embedding256: embedding256, rho128: rho128))
        }
        logProbs = probsList; speakerWeights = weightsList; chunkOffsets = offsets; embeddings = restored
        numFrames = meta.numFrames; numSpeakers = meta.numSpeakers; frameDuration = meta.frameDuration
        segmentationSeconds = meta.segmentationSeconds; embeddingSeconds = meta.embeddingSeconds
        savedChunks = meta.chunks; savedEmbeddings = meta.embeddings
        checkpointChunkBytes = meta.chunkBytes; checkpointEmbeddingBytes = meta.embeddingBytes
    }

    /// Writes the chunks prepared since the last save. Cheap to call after every `advance`.
    public func saveCheckpoint(to directory: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let chunksURL = directory.appendingPathComponent("chunks.bin"), embeddingsURL = directory.appendingPathComponent("embeddings.bin")
        if savedChunks == 0 { try? fm.removeItem(at: chunksURL); try? fm.removeItem(at: embeddingsURL); checkpointChunkBytes = 0; checkpointEmbeddingBytes = 0 }
        var chunkBlob = Data()
        for i in savedChunks..<chunkOffsets.count {
            CheckpointWriter.double(chunkOffsets[i], into: &chunkBlob)
            CheckpointWriter.matrix(logProbs[i], into: &chunkBlob)
            CheckpointWriter.matrix(speakerWeights[i], into: &chunkBlob)
        }
        var embeddingBlob = Data()
        for e in embeddings[savedEmbeddings...] {
            for v in [e.chunkIndex, e.speakerIndex, e.startFrame, e.endFrame] { CheckpointWriter.int(v, into: &embeddingBlob) }
            CheckpointWriter.floats(e.frameWeights, into: &embeddingBlob)
            CheckpointWriter.double(e.startTime, into: &embeddingBlob)
            CheckpointWriter.double(e.endTime, into: &embeddingBlob)
            CheckpointWriter.floats(e.embedding256, into: &embeddingBlob)
            CheckpointWriter.doubles(e.rho128, into: &embeddingBlob)
        }
        try CheckpointWriter.write(chunkBlob, to: chunksURL, at: checkpointChunkBytes)
        try CheckpointWriter.write(embeddingBlob, to: embeddingsURL, at: checkpointEmbeddingBytes)
        let meta = CheckpointMeta(sampleCount: source.sampleCount, samplesPerStep: config.samplesPerStep, chunks: chunkOffsets.count,
                                  embeddings: embeddings.count, chunkBytes: checkpointChunkBytes + chunkBlob.count,
                                  embeddingBytes: checkpointEmbeddingBytes + embeddingBlob.count, numFrames: numFrames, numSpeakers: numSpeakers,
                                  frameDuration: frameDuration, segmentationSeconds: segmentationSeconds, embeddingSeconds: embeddingSeconds)
        try JSONEncoder().encode(meta).write(to: directory.appendingPathComponent("meta.json"), options: [.atomic])
        checkpointChunkBytes = meta.chunkBytes; checkpointEmbeddingBytes = meta.embeddingBytes
        savedChunks = meta.chunks; savedEmbeddings = meta.embeddings
    }
}

private enum CheckpointWriter {
    static func int(_ v: Int, into d: inout Data) { var x = Int64(v).littleEndian; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }
    static func double(_ v: Double, into d: inout Data) { var x = v.bitPattern.littleEndian; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }
    static func floats(_ v: [Float], into d: inout Data) {
        int(v.count, into: &d)
        for f in v { var x = f.bitPattern.littleEndian; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }
    }
    static func doubles(_ v: [Double], into d: inout Data) { int(v.count, into: &d); for x in v { double(x, into: &d) } }
    static func matrix(_ m: [[Float]], into d: inout Data) { int(m.count, into: &d); for row in m { floats(row, into: &d) } }
    /// Writes `data` at byte `offset`, dropping anything after it (a save that was cut off half way).
    static func write(_ data: Data, to url: URL, at offset: Int) throws {
        if !FileManager.default.fileExists(atPath: url.path) { FileManager.default.createFile(atPath: url.path, contents: nil) }
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.truncate(atOffset: UInt64(offset))
        try handle.seek(toOffset: UInt64(offset))
        try handle.write(contentsOf: data)
    }
}

private struct CheckpointReader {
    let data: Data
    var position: Int
    init(data: Data) { self.data = Data(data); position = 0 }
    mutating func raw64() -> UInt64? {
        guard position + 8 <= data.count else { return nil }
        var v: UInt64 = 0
        _ = withUnsafeMutableBytes(of: &v) { data.copyBytes(to: $0, from: position..<(position + 8)) }
        position += 8
        return UInt64(littleEndian: v)
    }
    mutating func int() -> Int? { raw64().map { Int(Int64(bitPattern: $0)) } }
    mutating func double() -> Double? { raw64().map { Double(bitPattern: $0) } }
    mutating func floats() -> [Float]? {
        guard let n = int(), n >= 0, position + n * 4 <= data.count else { return nil }
        var out = [Float](repeating: 0, count: n)
        out.withUnsafeMutableBytes { _ = data.copyBytes(to: $0, from: position..<(position + n * 4)) }
        position += n * 4
        return out.map { Float(bitPattern: UInt32(littleEndian: $0.bitPattern)) }
    }
    mutating func doubles() -> [Double]? {
        guard let n = int(), n >= 0 else { return nil }
        var out: [Double] = []; out.reserveCapacity(n)
        for _ in 0..<n { guard let v = double() else { return nil }; out.append(v) }
        return out
    }
    mutating func matrix() -> [[Float]]? {
        guard let n = int(), n >= 0 else { return nil }
        var out: [[Float]] = []; out.reserveCapacity(n)
        for _ in 0..<n { guard let row = floats() else { return nil }; out.append(row) }
        return out
    }
}
