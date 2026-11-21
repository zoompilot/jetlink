import Foundation
import JetlinkKit
import JetlinkTestSupport
import Testing

@testable import JetlinkServer

#if canImport(CryptoKit)
  import CryptoKit
#else
  import Crypto
#endif

/// The comma's end, as jetlink/addons/wire.py packs it.
private func hex(_ s: String) -> [UInt8] {
  stride(from: 0, to: s.count, by: 2).map { UInt8(s.dropFirst($0).prefix(2), radix: 16)! }
}

/// A byte stream each way: what the comma wrote, then a closed link.
final class FakeAddonPipes: BulkPipes, @unchecked Sendable {
  private let lock = NSLock()
  private var inbound: [UInt8]
  private(set) var written: [UInt8] = []

  init(_ inbound: [UInt8]) {
    self.inbound = inbound
  }

  func read(into buffer: UnsafeMutableRawPointer, count: Int, timeout: TimeInterval) throws -> Int {
    try lock.withLock {
      guard !inbound.isEmpty else { throw LinkError.closed("unplugged") }
      let n = min(count, inbound.count)
      inbound.withUnsafeBytes { buffer.copyMemory(from: $0.baseAddress!, byteCount: n) }
      inbound.removeFirst(n)
      return n
    }
  }

  func write(from buffer: UnsafeRawPointer, count: Int, timeout: TimeInterval) throws -> Int {
    lock.withLock { written += UnsafeRawBufferPointer(start: buffer, count: count) }
    return count
  }

  func abort() {}
  func close() {}
}

/// An interface never on the bus: the provisioning tests serve nothing over it.
struct AbsentAddon: AddonSource {
  func present() -> Bool { false }
  func open() throws -> AddonPipes { throw LinkError.closed("absent") }
}

final class Ticks: @unchecked Sendable {
  private let lock = NSLock()
  private var value: UInt64 = 10_000_000_000
  var now: UInt64 { lock.withLock { value } }
  func advance(ms: UInt64) { lock.withLock { value += ms * 1_000_000 } }
}

@Suite("Add-ons", .serialized)
struct AddonTests {
  static let addon = Fixture.url("tiny_addon.onnx")

  @Test("The wire is byte for byte the Python side's")
  func wire() throws {
    #expect(AddonWire.hello(tags: [7, 3], yielding: false) == hex("4a415231ffffffff010000006400000000000000020000000300000007000000"))
    #expect(
      AddonWire.answer(seq: 5, tag: 9, status: .ok, serverUs: 12, floats: [1.5, -2])
        == hex("4a4152310500000009000000000000000c000000020000000000c03f000000c0"))
    #expect(AddonWire.answer(seq: 1, tag: 1, status: .ok, floats: [Float](repeating: 0, count: 122)).count == 513)
    let request = try hex("4a4151310400000001efcdab010002000300000106000000000102030405").withUnsafeBytes { try AddonWire.request(header: $0) }
    #expect(request == AddonWire.Request(seq: 4, tag: 0xabcd_ef01, shape: [1, 2, 3], encoding: .u8, nbytes: 6))
    #expect(AddonWire.tag(sha256: "abcdef0123") == 0xabcd_ef01)
  }

  @Test("A malformed header is refused")
  func malformed() {
    let good = hex("4a4151310400000001efcdab010002000300000106000000")
    for (offset, byte) in [(0, UInt8(0x58)), (18, 9), (19, 2), (20, 7)] {
      var bad = good
      bad[offset] = byte
      #expect(throws: AddonWire.WireError.self) { try bad.withUnsafeBytes { try AddonWire.request(header: $0) } }
    }
    #expect(throws: AddonWire.WireError.self) { try Array(good.prefix(10)).withUnsafeBytes { try AddonWire.request(header: $0) } }
  }

  @Test("An interface class is three hex bytes, never Jetlink's own")
  func interfaceClass() {
    #expect(AddonWire.interfaceClass("ff/56/41") == [0xFF, 0x56, 0x41])
    #expect(AddonWire.triple([0xFF, 0x56, 0x41]) == "ff/56/41")
    for bad in ["ff/ff/ff", "ff/56", "ff/56/41/00", "fg/56/41", "f/56/41"] { #expect(AddonWire.interfaceClass(bad) == nil) }
  }

  @Test("A pass fits only in the gap before the next frame, or with nothing driving")
  func gate() {
    let ticks = Ticks()
    let gate = AddonGate(clock: { ticks.now })
    #expect(gate.gap(cost: 5_000_000, wait: 0))
    for _ in 0..<10 {
      gate.frameStarted()
      ticks.advance(ms: 50)
    }
    gate.frameStarted()
    #expect(!gate.gap(cost: 5_000_000, wait: 0), "in a frame")
    #expect(!gate.yielding, "one miss is not yielding")
    #expect(!gate.gap(cost: 5_000_000, wait: 0) && !gate.gap(cost: 5_000_000, wait: 0))
    #expect(gate.yielding)
    ticks.advance(ms: 10)
    gate.frameEnded()
    #expect(gate.gap(cost: 5_000_000, wait: 0), "the gap after it")
    #expect(!gate.gap(cost: 40_000_000, wait: 0), "too long for the gap")
    ticks.advance(ms: 40)
    #expect(!gate.gap(cost: 5_000_000, wait: 0), "the next frame is due")
    ticks.advance(ms: 2_000)
    #expect(gate.gap(cost: 5_000_000, wait: 0), "nothing driving")
    #expect(!gate.yielding)
  }

  @Test("At 20 Hz with 15 ms frames, a request at any point in the period gets a gap within the comma's budget")
  func gateSweep() {
    let cost: UInt64 = 8_000_000
    let wait = max(0, AddonService.answerBudget - Double(cost) / 1e9)
    var served = 0
    for offset in stride(from: 0, to: 50, by: 2) {
      let gate = AddonGate()
      let frames = DispatchQueue(label: "frames")
      let done = DispatchSemaphore(value: 0)
      frames.async {
        for _ in 0..<4 {
          gate.frameStarted()
          Thread.sleep(forTimeInterval: 0.015)
          gate.frameEnded()
          Thread.sleep(forTimeInterval: 0.035)
        }
        done.signal()
      }
      Thread.sleep(forTimeInterval: 0.100 + Double(offset) / 1000)
      let started = Date()
      if gate.gap(cost: cost, wait: wait), Date().timeIntervalSince(started) + Double(cost) / 1e9 < 0.050 { served += 1 }
      done.wait()
    }
    // real sleeps: a loaded runner misses a few; a gate that never waits serves about half
    #expect(served >= 20, "\(served) of 25 offsets served")
  }

  @Test("A model builds on the backend and answers every output, in name order")
  func model() throws {
    let cache = try TemporaryDirectory()
    let model = try AddonModel.prepare(Self.addon, backend: cpuBackend(), root: cache.url, log: ServerLog(category: "test"))
    defer { model.engine.close() }
    #expect(model.shape == [3, 4, 4])
    #expect(model.outputs.map(\.name) == ["a_scaled", "b_mean"])
    let digest = SHA256.hash(data: try Data(contentsOf: Self.addon)).map { String(format: "%02x", $0) }.joined()
    #expect(model.sha256 == digest && model.tag == AddonWire.tag(sha256: digest))
    let image = (0..<48).map { UInt8($0 * 5) }
    let request = AddonWire.Request(seq: 1, tag: model.tag, shape: [3, 4, 4], encoding: .u8, nbytes: 48)
    let floats = try image.withUnsafeBytes { try model.run(request, image: $0) }
    #expect(floats.count == 51)
    for i in 0..<48 { #expect(abs(floats[i] - Float(image[i]) / 255 * 2) < 1e-5) }
    for c in 0..<3 {
      let mean = image[(c * 16)..<(c * 16 + 16)].reduce(Float(0)) { $0 + Float($1) / 255 } / 16
      #expect(abs(floats[48 + c] - mean) < 1e-5)
    }
    let f32 = AddonWire.Request(seq: 2, tag: model.tag, shape: [3, 4, 4], encoding: .f32, nbytes: 192)
    let again = try image.map { Float($0) / 255 }.withUnsafeBytes { try model.run(f32, image: $0) }
    #expect(zip(again, floats).allSatisfy { abs($0 - $1) < 1e-6 })
    let artifacts = try FileManager.default.contentsOfDirectory(atPath: cache.url.appending(path: "addons/engines").path)
    #expect(artifacts.contains { $0.hasPrefix(model.sha256.prefix(16)) })
  }

  @Test("The service answers a request over the link, refuses an unknown model, and hellos when idle")
  func service() throws {
    let cache = try TemporaryDirectory()
    let backend = cpuBackend()
    let host = EngineHost(cache: try ServerCache(root: cache.url, backend: backend), hooks: ServerHooks())
    defer { host.close() }
    let model = try AddonModel.prepare(Self.addon, backend: backend, root: cache.url, log: ServerLog(category: "test"), host: host)
    defer { model.engine.close() }
    let service = AddonService(host: host, cacheRoot: cache.url) { _ in fatalError("unused") }
    let served = AddonService.Served(kind: [0xFF, 0x54, 0x59], name: "tiny", models: [model.tag: model])

    func request(_ seq: UInt32, _ tag: UInt32) -> [UInt8] {
      var out = Array("JAQ1".utf8)
      for v: UInt32 in [seq, tag] { withUnsafeBytes(of: v.littleEndian) { out += $0 } }
      for v: UInt16 in [3, 4, 4] { withUnsafeBytes(of: v.littleEndian) { out += $0 } }
      out += [0, 1]
      withUnsafeBytes(of: UInt32(48).littleEndian) { out += $0 }
      return out + [UInt8](repeating: 255, count: 48)
    }
    let pipes = FakeAddonPipes(request(1, model.tag) + request(2, 0xDEAD))
    #expect(throws: LinkError.self) { try service.serve(served, pipes: AddonPipes(pipes)) }
    let ok = Array(pipes.written.prefix(24 + 51 * 4))
    #expect(ok.prefix(4) == ArraySlice("JAR1".utf8))
    #expect(ok[12..<16] == [0, 0, 0, 0] && ok[20] == 51)
    let unknown = Array(pipes.written.dropFirst(24 + 51 * 4))
    #expect(unknown.count == 24 && unknown[4] == 2 && unknown[12] == UInt8(AddonWire.Status.unknownModel.rawValue))
    #expect(served.passes == 1)
  }

  @Test("Over a phone's cable the server dials the comma at the add-on's port and answers there")
  func tcp() throws {
    let cache = try TemporaryDirectory()
    let backend = cpuBackend()
    let host = EngineHost(cache: try ServerCache(root: cache.url, backend: backend), hooks: ServerHooks())
    defer { host.close() }
    let model = try AddonModel.prepare(Self.addon, backend: backend, root: cache.url, log: ServerLog(category: "test"), host: host)
    defer { model.engine.close() }
    let service = AddonService(host: host, cacheRoot: cache.url) { _ in fatalError("unused") }
    let served = AddonService.Served(kind: [0xFF, 0x54, 0x59], name: "tiny", port: 1, models: [model.tag: model])
    let later = AddonInterface(kind: [0xFF, 0x54, 0x59])
    #expect(!TCPAddonSource(host: "127.0.0.1", interface: later).present(), "no port, never dialed")
    later.learn(port: 1)
    #expect(TCPAddonSource(host: "127.0.0.1", interface: later).present(), "a port learned later reaches the same interface")
    #expect(!TCPAddonSource(host: "127.0.0.1", interface: later, reachable: { false }).present(), "not while the comma is not dialed")

    let comma = try TCPListener(host: "127.0.0.1", port: 0)
    defer { comma.close() }
    let source = TCPAddonSource(host: "127.0.0.1", interface: AddonInterface(kind: [0xFF, 0x54, 0x59], port: comma.port))
    #expect(source.present())
    let pipes = try source.open()
    let fd = try #require(comma.acceptConnection()?.fd)
    defer { Sys.close(fd) }
    let done = DispatchSemaphore(value: 0)
    DispatchQueue.global().async {
      _ = try? service.serve(served, pipes: pipes)
      done.signal()
    }

    var request = Array("JAQ1".utf8)
    for v: UInt32 in [7, model.tag] { withUnsafeBytes(of: v.littleEndian) { request += $0 } }
    for v: UInt16 in [3, 4, 4] { withUnsafeBytes(of: v.littleEndian) { request += $0 } }
    request += [0, 1]
    withUnsafeBytes(of: UInt32(48).littleEndian) { request += $0 }
    request += [UInt8](repeating: 255, count: 48)
    // in two pieces: a stream need not keep a message whole
    for piece in [request.prefix(10), request.dropFirst(10)] {
      var bytes = Array(piece)
      var vector = bytes.withUnsafeMutableBytes { iovec(iov_base: $0.baseAddress, iov_len: $0.count) }
      #expect(Sys.writev(fd, &vector, 1) == bytes.count)
    }
    func read(_ count: Int) -> [UInt8] {
      var out = [UInt8](repeating: 0, count: count)
      var have = 0
      while have < count {
        let got = out.withUnsafeMutableBytes { Sys.read(fd, $0.baseAddress! + have, count - have) }
        if got <= 0 { break }
        have += got
      }
      return Array(out.prefix(have))
    }
    var answer: [UInt8] = []
    while answer.isEmpty {
      let header = read(24)
      guard header.count == 24 else { break }
      let count = Int(header[20]) | Int(header[21]) << 8
      let body = read(count * 4)
      if header[4..<8] != [0xFF, 0xFF, 0xFF, 0xFF] { answer = header + body }  // hellos skipped
    }
    #expect(answer.prefix(4) == ArraySlice("JAR1".utf8) && answer[4] == 7 && answer[20] == 51)
    pipes.pipes.abort()
    #expect(done.wait(timeout: .now() + 5) == .success)
    #expect(served.passes == 1)
  }

  static let addonSHA = SHA256.hash(data: try! Data(contentsOf: addon)).map { String(format: "%02x", $0) }.joined()

  /// ENGINE_REQ for an add-on, with the upload when asked, then the pushed states until one is final.
  func provision(_ client: TestClient, model: URL, sha256: String, interface: String = "ff/54/59") throws -> [String: Any] {
    let bytes = try Data(contentsOf: model)
    try client.sendJSON(.engineReq, ["sha256": sha256, "nbytes": bytes.count, "addon": ["interface": interface, "name": "tiny"]])
    var state = try client.recv(.engineResp).json
    if state["state"] as? String == "need_upload" {
      var payload = withUnsafeBytes(of: UInt64(0).littleEndian) { Data($0) }
      payload.append(bytes)
      try client.send(.uploadChunk, payload)
      try client.sendJSON(.uploadDone, ["sha256": sha256])
      state = try client.recv(.engineResp).json
    }
    while state["state"] as? String == "building" { state = try client.recv(.engineResp).json }
    return state
  }

  @Test("An add-on ENGINE_REQ is uploaded, built and served, and the driving engine stays loaded")
  func provisioned() throws {
    let golden = try Golden("tiny_queued")
    try serve { server, client in
      let root = server.configuration.cacheRoot
      let service = AddonService(host: server.host, cacheRoot: root) { _ in AbsentAddon() }
      service.start()
      let hello = try client.hello()
      let addons = try #require(hello["addons"] as? [String: Any])
      #expect(addons["ready"] as? [String] == [] && addons["failed"] as? [String] == [])
      _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)

      let state = try provision(client, model: Self.addon, sha256: Self.addonSHA)
      #expect(state["state"] as? String == "ready" && state["sha256"] as? String == Self.addonSHA && state["spec"] == nil)
      #expect(server.host.loadedSHA() == golden.sha256)
      let again = try client.hello()
      #expect((again["addons"] as? [String: Any])?["ready"] as? [String] == [Self.addonSHA])
      #expect(try provision(client, model: Self.addon, sha256: Self.addonSHA)["state"] as? String == "ready")
      // the driving model's frames still find their engine: the add-on never became the session's request
      _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
      service.stop()

      let restarted = AddonService(host: server.host, cacheRoot: root) { _ in AbsentAddon() }
      restarted.start()
      defer { restarted.stop() }
      #expect(eventually { restarted.helloState()["ready"] as? [String] == [Self.addonSHA] })
    }
  }

  @Test("A model the backend cannot serve as an add-on answers failed, and says so after a restart")
  func failedBuild() throws {
    let golden = try Golden("tiny_queued")
    try serve { server, client in
      let root = server.configuration.cacheRoot
      let service = AddonService(host: server.host, cacheRoot: root) { _ in AbsentAddon() }
      service.start()
      _ = try client.hello()
      let state = try provision(client, model: golden.model, sha256: golden.sha256)
      #expect(state["state"] as? String == "failed")
      #expect(server.host.loadedSHA() == nil)
      service.stop()
      let restarted = AddonService(host: server.host, cacheRoot: root) { _ in AbsentAddon() }
      restarted.start()
      defer { restarted.stop() }
      #expect(restarted.helloState()["failed"] as? [String] == [golden.sha256])
      #expect(try provision(client, model: golden.model, sha256: golden.sha256)["state"] as? String == "failed")
    }
  }

  @Test("A server without add-ons says nothing of them, and refuses an add-on request")
  func noService() throws {
    try serve { _, client in
      #expect(try client.hello()["addons"] == nil)
      try client.sendJSON(.engineReq, ["sha256": Self.addonSHA, "nbytes": 10, "addon": ["interface": "ff/54/59"]])
      let reply = try client.recv()
      #expect(reply.type == Wire.Msg.error.rawValue)
      #expect(try reply.json["error"] as? String == "no_addons")
    }
  }
}
