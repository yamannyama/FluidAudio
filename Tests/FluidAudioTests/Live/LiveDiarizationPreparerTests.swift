import XCTest

@testable import FluidAudio

/// Live preparation (fed minute by minute) must give the same result as the one-shot run.
/// Needs real audio and models: set LIVE_PREP_AUDIO to a file path. Skipped otherwise.
@available(macOS 14.0, iOS 17.0, *)
final class LiveDiarizationPreparerTests: XCTestCase {
    func testLiveMatchesOneShot() async throws {
        guard let path = ProcessInfo.processInfo.environment["LIVE_PREP_AUDIO"] else { throw XCTSkip("LIVE_PREP_AUDIO not set") }
        let samples = try AudioConverter().resampleAudioFile(URL(fileURLWithPath: path))
        var config = OfflineDiarizerConfig.default
        config.exposeChunkEmbeddings = true
        let manager = OfflineDiarizerManager(config: config)
        try await manager.prepareModels()

        let t0 = Date()
        let whole = try await manager.process(audio: samples)
        let wholeSeconds = Date().timeIntervalSince(t0)

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("live-prep-\(UUID().uuidString).f32")
        let source = try GrowingAudioSampleSource(fileURL: url)
        defer { source.cleanup() }
        let preparer = try LiveDiarizationPreparer(manager: manager, source: source)
        // 0.1 s pieces like a capture tap; try to advance after every 12 s
        let piece = 1_600
        var fed = 0, sinceAdvance = 0, advances = 0
        while fed < samples.count {
            let n = min(piece, samples.count - fed)
            samples[fed..<(fed + n)].withUnsafeBufferPointer { source.append($0) }
            fed += n; sinceAdvance += n
            if sinceAdvance >= 16_000 * 12 { sinceAdvance = 0; if try await preparer.advance() > 0 { advances += 1 } }
        }
        let coveredBeforeFinish = preparer.preparedSeconds
        let t1 = Date()
        let prepared = try await preparer.finish()
        let finishSeconds = Date().timeIntervalSince(t1)
        let t2 = Date()
        let live = try manager.cluster(prepared)
        let clusterSeconds = Date().timeIntervalSince(t2)

        print("LIVEPREP audio=\(samples.count / 16_000)s whole=\(wholeSeconds)s advances=\(advances) covered=\(coveredBeforeFinish)s work=\(preparer.workSeconds)s finish=\(finishSeconds)s cluster=\(clusterSeconds)s segments whole=\(whole.segments.count) live=\(live.segments.count)")
        XCTAssertEqual(whole.segments.count, live.segments.count)
        var mismatched = 0, maxEmbeddingGap: Float = 0
        for (a, b) in zip(whole.segments, live.segments) {
            if a.speakerId != b.speakerId || abs(a.startTimeSeconds - b.startTimeSeconds) > 0.001 || abs(a.endTimeSeconds - b.endTimeSeconds) > 0.001 { mismatched += 1 }
            for (x, y) in zip(a.embedding, b.embedding) { maxEmbeddingGap = max(maxEmbeddingGap, abs(x - y)) }
        }
        print("LIVEPREP mismatched=\(mismatched) maxEmbeddingGap=\(maxEmbeddingGap) chunkEmbeddings whole=\(whole.chunkEmbeddings?.count ?? -1) live=\(live.chunkEmbeddings?.count ?? -1)")
        XCTAssertEqual(mismatched, 0)
        XCTAssertEqual(whole.chunkEmbeddings?.count, live.chunkEmbeddings?.count)
    }
}
