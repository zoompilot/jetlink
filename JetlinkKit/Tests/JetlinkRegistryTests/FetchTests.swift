import Foundation
import JetlinkTestSupport
import Testing

@testable import JetlinkRegistry

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

/// tests/test_registry.py's fetch section, plus what the Swift download adds:
/// progress by whole percent, a real socket, and no .part left on any failure.
struct FetchTests {
  @Test func fallsThroughToTheServerThatHasIt() async throws {
    let tmp = try TemporaryDirectory()
    let net = MockNet(RegistryFixture.smallRoutes())
    let registry = Registry(layout: tmp.layout, session: net.session)
    let seen = ProgressLog()

    let path = try await registry.fetch(RegistryFixture.smallRef, progress: seen.callback)

    #expect(path == (try registry.modelPath(sha256: RegistryFixture.blobSHA)))
    #expect(try Data(contentsOf: path) == RegistryFixture.blob)
    #expect(tmp.names("models") == ["\(RegistryFixture.blobSHA.prefix(16)).onnx"])
    #expect(seen.all.last == 1.0)
    #expect(net.urls.contains(LFS.bucketURL(oid: RegistryFixture.blobSHA)) && net.urls.contains("\(LFS.endpoints[1])/objects/batch"))
    #expect(!net.urls.contains("\(LFS.endpoints[3])/objects/batch"), "the first server with the object serves it")
  }

  @Test func aModelAlreadyOnDiskTouchesNoNetwork() async throws {
    let tmp = try TemporaryDirectory()
    _ = try await Registry(layout: tmp.layout, session: MockNet(RegistryFixture.smallRoutes()).session).fetch(RegistryFixture.smallRef)
    let offline = MockNet()
    let path = try await Registry(layout: tmp.layout, session: offline.session).fetch(RegistryFixture.smallRef)
    #expect(try Data(contentsOf: path) == RegistryFixture.blob)
    // the index was kept, so even that is not asked for again
    #expect(offline.calls.isEmpty)
  }

  @Test func aWrongHashLeavesNothingBehind() async throws {
    let tmp = try TemporaryDirectory()
    let net = MockNet(RegistryFixture.smallRoutes(oid: String(repeating: "b", count: 64)))
    let error = await #expect(throws: RegistryError.self) {
      try await Registry(layout: tmp.layout, session: net.session).fetch(RegistryFixture.smallRef)
    }
    #expect(error?.kind == .verify)
    #expect(error?.message.contains("hash") == true)
    #expect(tmp.names("models").isEmpty)
  }

  @Test func aShortDownloadLeavesNothingBehind() async throws {
    let tmp = try TemporaryDirectory()
    let net = MockNet(RegistryFixture.smallRoutes(size: Int64(RegistryFixture.blob.count) + 99))
    let error = await #expect(throws: RegistryError.self) {
      try await Registry(layout: tmp.layout, session: net.session).fetch(RegistryFixture.smallRef)
    }
    #expect(error?.kind == .verify)
    #expect(error?.message.contains("bytes") == true)
    #expect(tmp.names("models").isEmpty)
  }

  @Test func aCancelledDownloadLeavesNothingBehind() async throws {
    let tmp = try TemporaryDirectory()
    let stops = StopScript([false, true, true])
    let error = await #expect(throws: RegistryError.self) {
      try await Registry(layout: tmp.layout, session: MockNet(RegistryFixture.smallRoutes()).session).fetch(
        RegistryFixture.smallRef, shouldStop: stops.callback)
    }
    #expect(error?.kind == .cancelled)
    #expect(error?.message.contains("cancelled") == true)
    #expect(tmp.names("models").isEmpty)
  }

  // LocalServer speaks Darwin's and Glibc's sockets.
  #if canImport(Darwin) || canImport(Glibc)
    @Test func aCancelledTaskStopsTheDownload() async throws {
      let tmp = try TemporaryDirectory()
      let server = try LocalServer(total: 64 << 20)
      defer { server.stop() }
      let pointer = Pointer(oid: String(repeating: "c", count: 64), size: server.total)
      let dest = tmp.url.appending(path: "models/cccccccccccccccc.onnx")
      let task = Task {
        try await LFS.download(href: server.url, pointer: pointer, dest: dest, session: .shared, progress: { _ in }, shouldStop: { false })
      }
      try await Task.sleep(for: .milliseconds(20))
      task.cancel()
      let result = await task.result
      #expect(throws: RegistryError.self) { try result.get() }
      #expect(tmp.names("models").isEmpty)
    }
  #endif

  @Test func noServerHasIt() async throws {
    let tmp = try TemporaryDirectory()
    var routes = RegistryFixture.smallRoutes()
    routes["\(LFS.endpoints[2])/objects/batch"] = .body(RegistryFixture.batch(oid: RegistryFixture.blobSHA, size: Int64(RegistryFixture.blob.count), href: nil))
    let error = await #expect(throws: RegistryError.self) {
      try await Registry(layout: tmp.layout, session: MockNet(routes).session).fetch(RegistryFixture.smallRef)
    }
    #expect(error?.isNetwork == true)
    #expect(error?.message.contains("no LFS server") == true)
  }

  @Test func anHrefThatFailsIsANetworkErrorAndLeavesNothing() async throws {
    let tmp = try TemporaryDirectory()
    var routes = RegistryFixture.smallRoutes()
    routes["https://blob.example/object"] = .status(403)
    let error = await #expect(throws: RegistryError.self) {
      try await Registry(layout: tmp.layout, session: MockNet(routes).session).fetch(RegistryFixture.smallRef)
    }
    #expect(error?.kind == .network)
    #expect(error?.message.hasPrefix("could not download \(RegistryFixture.blobSHA.prefix(16)): HTTP Error 403") == true)
    #expect(tmp.names("models").isEmpty)

    routes["https://blob.example/object"] = .failure
    let dropped = await #expect(throws: RegistryError.self) {
      try await Registry(layout: tmp.layout, session: MockNet(routes).session).fetch(RegistryFixture.smallRef)
    }
    #expect(dropped?.kind == .network)
    #expect(tmp.names("models").isEmpty)
  }

  @Test func tooLittleDiskIsRefusedBeforeAnyBytes() async throws {
    let tmp = try TemporaryDirectory()
    let huge = Pointer(oid: RegistryFixture.blobSHA, size: 1 << 60)
    let error = await #expect(throws: RegistryError.self) {
      try await LFS.download(
        href: "https://blob.example/object", pointer: huge, dest: tmp.url.appending(path: "models/x.onnx"),
        session: MockNet().session, progress: { _ in }, shouldStop: { false })
    }
    #expect(error?.message.hasPrefix("need \(Int64(1 << 60) >> 20) MB for the model") == true)
    #expect(tmp.names("models").isEmpty)
  }

  @Test func progressIsReportedByWholePercent() async throws {
    let tmp = try TemporaryDirectory()
    let blob = Data((0..<(1 << 20)).map { UInt8(truncatingIfNeeded: $0 &* 31) })
    let pieces = stride(from: 0, to: blob.count, by: 1000).map { blob.subdata(in: $0..<min($0 + 1000, blob.count)) }
    var routes = RegistryFixture.smallRoutes(blob, oid: sha256Hex(blob))
    routes["https://blob.example/object"] = .chunks(pieces)
    let seen = ProgressLog()

    _ = try await Registry(layout: tmp.layout, session: MockNet(routes).session).fetch(RegistryFixture.smallRef, progress: seen.callback)

    let values = seen.all
    #expect(values.last == 1.0)
    #expect(values == values.sorted())
    let percents = values.dropLast().map { Int($0 * 100) }
    #expect(Set(percents).count == percents.count, "one call per whole percent")
    #expect(values.count <= 102)
  }
}

#if canImport(Darwin) || canImport(Glibc)
  /// The download through URLSession's real HTTP stack, from a server on loopback.
  struct LocalDownloadTests {
    static let benchmark = ProcessInfo.processInfo.environment["JETLINK_BENCH"] == "1"

    private func download(megabytes: Int64, session: URLSession) async throws -> (seconds: Double, progress: [Double]) {
      let tmp = try TemporaryDirectory()
      let server = try LocalServer(total: megabytes << 20)
      defer { server.stop() }
      let pointer = Pointer(oid: server.sha256, size: server.total)
      let dest = try tmp.layout.modelPath(sha256: pointer.oid)
      let seen = ProgressLog()

      let clock = ContinuousClock()
      let start = clock.now
      let path = try await LFS.download(href: server.url, pointer: pointer, dest: dest, session: session, progress: seen.callback, shouldStop: { false })
      let elapsed = clock.now - start

      #expect(path == dest)
      #expect(Files.status(dest)?.size == server.total)
      #expect(tmp.names("models") == [dest.lastPathComponent])
      let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
      return (seconds, seen.all)
    }

    /// comma's bucket answers an object's address with a redirect to where
    /// the bytes are, and the href it gives is that address.
    @Test func followsARedirectAsTheBucketAnswers() async throws {
      let tmp = try TemporaryDirectory()
      let server = try LocalServer(total: 1 << 20)
      defer { server.stop() }
      let pointer = Pointer(oid: server.sha256, size: server.total)
      let dest = try tmp.layout.modelPath(sha256: pointer.oid)
      let path = try await LFS.download(
        href: server.redirectURL, pointer: pointer, dest: dest, session: .shared, progress: { _ in }, shouldStop: { false })
      #expect(Files.status(path)?.size == server.total)
    }

    @Test func streamsHashesAndVerifiesOverARealSocket() async throws {
      let (seconds, progress) = try await download(megabytes: 16, session: .shared)
      #expect(progress.last == 1.0)
      #expect(progress.count >= 2 && progress.count <= 102)
      print("local download: 16 MB in \(String(format: "%.3f", seconds)) s, \(String(format: "%.0f", 16 / seconds)) MB/s")
    }

    /// JETLINK_BENCH=1 swift test --filter throughput. JETLINK_BENCH_MB sets the
    /// size (256 by default); it is written to the temp dir and deleted.
    @Test(.enabled(if: benchmark)) func throughput() async throws {
      let megabytes = Int64(ProcessInfo.processInfo.environment["JETLINK_BENCH_MB"] ?? "") ?? 256
      let configuration = URLSessionConfiguration.ephemeral
      for (label, session) in [("shared", URLSession.shared), ("ephemeral", URLSession(configuration: configuration))] {
        let (seconds, _) = try await download(megabytes: megabytes, session: session)
        let rate = Double(megabytes) / seconds
        print("local download (\(label) session): \(megabytes) MB in \(String(format: "%.3f", seconds)) s = \(String(format: "%.0f", rate)) MB/s")
        #expect(rate >= 100, "the download must not be per-byte slow")
      }
    }
  }
#endif
