import Foundation
import JetlinkKit
import JetlinkTestSupport
import Testing

@testable import JetlinkServer

/// A build the disk has no room for is refused before it starts: out of room
/// part way, CoreML fails with "MIL FileWriter" and a nearly full disk stops
/// the Neural Engine for every model.
@Suite("Storage checks", .serialized)
struct StorageCheckTests {
  func host(_ golden: Golden, in tmp: TemporaryDirectory, backend: FlakyBackend) throws -> (EngineHost, ServerCache, Request) {
    let cache = try ServerCache(root: tmp.url, backend: backend)
    let model = try Data(contentsOf: golden.model)
    let request = try Request(sha256: golden.sha256, nbytes: Int64(model.count), frameSkip: 4)
    try model.write(to: cache.modelPath(request))
    return (EngineHost(cache: cache), cache, request)
  }

  @Test("A build that would not fit is refused, and says what it needs")
  func refusesABuild() throws {
    let tmp = try TemporaryDirectory()
    let backend = FlakyBackend()
    backend.need = StorageNeed(model: 1, prepared: Int64.max / 4, buildScratch: 1, neuralEngine: 1, headroom: 1)
    let (host, cache, request) = try host(Golden("tiny_stateful"), in: tmp, backend: backend)
    defer { host.close() }
    _ = host.request(request, session: nil)
    #expect(host.settles())
    let event = host.snapshot()
    #expect(event.state == .failed)
    #expect(event.detail.contains("free to prepare"))
    #expect(backend.counts.builds == 0)
    #expect(!cache.entry(request).exists)
  }

  @Test("With room for its files, even with little left over, or with no estimate, the build goes ahead")
  func buildsWithRoom() throws {
    let tight = StorageNeed(model: 1, prepared: 1, buildScratch: 1, neuralEngine: 1, headroom: Int64.max / 4)
    for need in [StorageNeed(model: 1, prepared: 1, buildScratch: 1, neuralEngine: 1, headroom: 1), tight, nil] {
      let tmp = try TemporaryDirectory()
      let backend = FlakyBackend()
      backend.need = need
      let (host, _, request) = try host(Golden("tiny_stateful"), in: tmp, backend: backend)
      _ = host.request(request, session: nil)
      #expect(host.settles())
      #expect(host.snapshot().state == .ready)
      #expect(backend.counts.builds == 1)
      host.close()
    }
  }
}
