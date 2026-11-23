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
/// `idleFirst`: the first read finds nothing, as a second with no request does.
/// `chunk`: at most that many bytes a write, as a socket may take.
final class FakeAddonPipes: BulkPipes, @unchecked Sendable {
  private let lock = NSLock()
  private var inbound: [UInt8]
  private var idle: Bool
  private let chunk: Int
  private(set) var written: [UInt8] = []

  init(_ inbound: [UInt8], idleFirst: Bool = false, chunk: Int = .max) {
    self.inbound = inbound
    idle = idleFirst
    self.chunk = chunk
  }

  func read(into buffer: UnsafeMutableRawPointer, count: Int, timeout: TimeInterval) throws -> Int {
    try lock.withLock {
      if idle {
        idle = false
        return 0
      }
      guard !inbound.isEmpty else { throw LinkError.closed("unplugged") }
      let n = min(count, inbound.count)
      inbound.withUnsafeBytes { buffer.copyMemory(from: $0.baseAddress!, byteCount: n) }
      inbound.removeFirst(n)
      return n
    }
  }

  func write(from buffer: UnsafeRawPointer, count: Int, timeout: TimeInterval) throws -> Int {
    let n = min(count, chunk)
    lock.withLock { written += UnsafeRawBufferPointer(start: buffer, count: n) }
    return n
  }

  func abort() {}
  func close() {}
}

/// An interface never on the bus: the provisioning tests serve nothing over it.
struct AbsentAddon: AddonSource {
  func present() -> Bool { false }
  func open() throws -> AddonPipes { throw LinkError.closed("absent") }
}

final class Flag: @unchecked Sendable {
  private let lock = NSLock()
  private var value = false
  func set(_ new: Bool) { lock.withLock { value = new } }
  func get() -> Bool { lock.withLock { value } }
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
    #expect(gate.admit(cost: 5_000_000, wait: 0))
    gate.passEnded()
    for _ in 0..<10 {
      gate.frameStarted()
      ticks.advance(ms: 50)
    }
    gate.frameStarted()
    #expect(!gate.admit(cost: 5_000_000, wait: 0), "in a frame")
    #expect(!gate.yielding, "one miss is not yielding")
    #expect(!gate.admit(cost: 5_000_000, wait: 0) && !gate.admit(cost: 5_000_000, wait: 0))
    #expect(gate.yielding)
    ticks.advance(ms: 10)
    gate.frameEnded()
    #expect(gate.admit(cost: 5_000_000, wait: 0), "the gap after it")
    gate.passEnded()
    #expect(!gate.admit(cost: 40_000_000, wait: 0), "too long for the gap")
    ticks.advance(ms: 40)
    #expect(!gate.admit(cost: 5_000_000, wait: 0), "the next frame is due")
    ticks.advance(ms: 2_000)
    #expect(gate.admit(cost: 5_000_000, wait: 0), "nothing driving")
    #expect(!gate.yielding)
  }

  @Test("Two add-ons never share a gap: the second decides again once the first is done")
  func gateTwoAddons() {
    let ticks = Ticks()
    let gate = AddonGate(clock: { ticks.now })
    for _ in 0..<10 {
      gate.frameStarted()
      ticks.advance(ms: 50)
    }
    gate.frameStarted()
    ticks.advance(ms: 10)
    gate.frameEnded()
    // 40 ms to the next frame, room for either 20 ms pass but not both
    #expect(gate.admit(cost: 20_000_000, wait: 0), "the first add-on")
    #expect(!gate.admit(cost: 20_000_000, wait: 0), "the second, while the first holds the gap")
    let second = DispatchSemaphore(value: 0)
    let admitted = Flag()
    DispatchQueue.global().async {
      admitted.set(gate.admit(cost: 20_000_000, wait: 1))
      second.signal()
    }
    Thread.sleep(forTimeInterval: 0.05)
    // the first pass took 25 ms: the next frame is now 15 ms away
    ticks.advance(ms: 25)
    gate.passEnded()
    #expect(second.wait(timeout: .now() + 5) == .success)
    #expect(!admitted.get(), "a frame is due: the second yields rather than run on the spent gap")
    ticks.advance(ms: 2_000)
    #expect(gate.admit(cost: 20_000_000, wait: 0), "nothing driving")
    gate.passEnded()
  }

  @Test("A model whose request or answer cannot cross the wire is refused when built, saying why")
  func wireLimits() throws {
    try AddonModel.fitsWire("ok", input: 3 * 640 * 640, outputs: 1000)
    #expect(AddonWire.maxAnswer == 4 << 20 && AddonWire.maxRequest == 4 << 20, "jetlink/addons/wire.py's MAX_ANSWER and MAX_REQUEST")
    try AddonModel.fitsWire("edge", input: 1, outputs: (AddonWire.maxAnswer - AddonWire.headerBytes) / 4)
    #expect {
      try AddonModel.fitsWire("dense", input: 1, outputs: (AddonWire.maxAnswer - AddonWire.headerBytes) / 4 + 1)
    } throws: { "\($0)".contains("answer") }
    #expect {
      try AddonModel.fitsWire("huge", input: AddonWire.maxRequest, outputs: 1)
    } throws: { "\($0)".contains("request") }
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
      if gate.admit(cost: cost, wait: wait) {
        gate.passEnded()
        if Date().timeIntervalSince(started) + Double(cost) / 1e9 < 0.050 { served += 1 }
      }
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

  @Test("A hello the link takes only part of is finished before the next answer, so the stream stays framed")
  func shortHello() throws {
    let cache = try TemporaryDirectory()
    let backend = cpuBackend()
    let host = EngineHost(cache: try ServerCache(root: cache.url, backend: backend), hooks: ServerHooks())
    defer { host.close() }
    let model = try AddonModel.prepare(Self.addon, backend: backend, root: cache.url, log: ServerLog(category: "test"), host: host)
    defer { model.engine.close() }
    let service = AddonService(host: host, cacheRoot: cache.url) { _ in fatalError("unused") }
    let served = AddonService.Served(kind: [0xFF, 0x54, 0x59], name: "tiny", models: [model.tag: model])
    var request = Array("JAQ1".utf8)
    for v: UInt32 in [1, model.tag] { withUnsafeBytes(of: v.littleEndian) { request += $0 } }
    for v: UInt16 in [3, 4, 4] { withUnsafeBytes(of: v.littleEndian) { request += $0 } }
    request += [0, 1]
    withUnsafeBytes(of: UInt32(48).littleEndian) { request += $0 }
    request += [UInt8](repeating: 255, count: 48)
    let pipes = FakeAddonPipes(request, idleFirst: true, chunk: 5)
    #expect(throws: LinkError.self) { try service.serve(served, pipes: AddonPipes(pipes)) }
    // a whole hello, then a whole answer, read as the comma reads a stream: by each header's count
    var stream = pipes.written[...]
    var messages: [(seq: UInt32, count: Int)] = []
    while stream.count >= 24 {
      let header = Array(stream.prefix(24))
      #expect(header.prefix(4) == ArraySlice("JAR1".utf8))
      let seq = header[4..<8].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
      let count = Int(header[20..<24].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) })
      messages.append((seq, count))
      stream = stream.dropFirst(AddonWire.padded(24 + count * 4))
    }
    #expect(stream.isEmpty)
    #expect(messages.map(\.seq) == [AddonWire.helloSeq, 1] && messages.map(\.count) == [1, 51])
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
      #expect((again["addons"] as? [String: Any])?["serving"] as? [String: [String]] == ["ff/54/59": [Self.addonSHA]])
      #expect(try provision(client, model: Self.addon, sha256: Self.addonSHA)["state"] as? String == "ready")
      // the same model for another add-on: ready there only once that interface serves it, from the same engine
      #expect(try provision(client, model: Self.addon, sha256: Self.addonSHA, interface: "ff/56/41")["state"] as? String == "ready")
      let both = ["ff/54/59": [Self.addonSHA], "ff/56/41": [Self.addonSHA]]
      #expect((try client.hello()["addons"] as? [String: Any])?["serving"] as? [String: [String]] == both)
      // the web page's Models tab: each interface, its model by name and a pass's time, nothing building or failed
      let page = service.pageState()
      let rows = try #require(page["addons"] as? [[String: Any]])
      #expect(rows.map { $0["interface"] as? String } == ["ff/54/59", "ff/56/41"])
      let model = try #require((rows.first?["models"] as? [[String: Any]])?.first)
      #expect(model["name"] as? String == "tiny" && model["sha256"] as? String == Self.addonSHA && (model["pass_ms"] as? Double).map { $0 >= 0 } == true)
      #expect(rows.first?["passes"] as? Int == 0 && rows.first?["last_answer_s"] == nil)
      #expect((page["building"] as? [Any])?.isEmpty == true && (page["failed"] as? [Any])?.isEmpty == true)
      // the driving model's frames still find their engine: the add-on never became the session's request
      _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
      service.stop()

      let restarted = AddonService(host: server.host, cacheRoot: root) { _ in AbsentAddon() }
      restarted.start()
      defer { restarted.stop() }
      #expect(eventually { restarted.helloState()["serving"] as? [String: [String]] == both })
      #expect(restarted.helloState()["ready"] as? [String] == [Self.addonSHA])
    }
  }

  @Test("A hello's addons_keep unloads what it leaves out on the interfaces it names, deletes it once nothing serves it, and a later request builds it again")
  func turnedOff() throws {
    let golden = try Golden("tiny_queued")
    try serve { server, client in
      let root = server.configuration.cacheRoot
      let service = AddonService(host: server.host, cacheRoot: root) { _ in AbsentAddon() }
      service.start()
      defer { service.stop() }
      let hello = try client.hello()
      #expect((hello["addons"] as? [String: Any])?["forgets"] as? Bool == true)
      _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
      #expect(try provision(client, model: Self.addon, sha256: Self.addonSHA)["state"] as? String == "ready")
      #expect(try provision(client, model: Self.addon, sha256: Self.addonSHA, interface: "ff/56/41")["state"] as? String == "ready")
      let store = AddonModel.store(root: root, backend: server.host.backend)
      let artifact = try store.entry(Self.addonSHA)
      let bytes = try server.host.cache.modelPath(Self.addonSHA)
      #expect(artifact.exists && FileManager.default.fileExists(atPath: bytes.path))

      // no word: nothing changes; an interface not named is left alone
      let both = ["ff/54/59": [Self.addonSHA], "ff/56/41": [Self.addonSHA]]
      #expect((try client.hello()["addons"] as? [String: Any])?["serving"] as? [String: [String]] == both)
      let off = try client.hello(addonsKeep: ["ff/54/59": []])
      #expect((off["addons"] as? [String: Any])?["serving"] as? [String: [String]] == ["ff/56/41": [Self.addonSHA]])
      // still served on the other interface: kept on disk
      #expect(artifact.exists && FileManager.default.fileExists(atPath: bytes.path))

      let none = try client.hello(addonsKeep: ["ff/54/59": [], "ff/56/41": []])
      #expect((none["addons"] as? [String: Any])?["serving"] as? [String: [String]] == [:])
      #expect(!artifact.exists && !FileManager.default.fileExists(atPath: bytes.path))
      #expect(server.host.loadedSHA() == golden.sha256)
      let serving = (try? JSONSerialization.jsonObject(with: Data(contentsOf: store.layout.root.appending(path: "serving.json")))) as? [String: Any]
      #expect((serving?[server.host.backend.tag()] as? [String: Any])?.isEmpty == true)

      // turned on again: asked for, uploaded and built again
      #expect(try provision(client, model: Self.addon, sha256: Self.addonSHA)["state"] as? String == "ready")
      #expect((try client.hello()["addons"] as? [String: Any])?["serving"] as? [String: [String]] == ["ff/54/59": [Self.addonSHA]])
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

  @Test("One slow pass does not keep an add-on off the server: each refused pass brings its cost back toward the measured one")
  func costRecovers() throws {
    let cache = try TemporaryDirectory()
    let model = try AddonModel.prepare(Self.addon, backend: cpuBackend(), root: cache.url, log: ServerLog(category: "test"))
    defer { model.engine.close() }
    let baseline = model.baselineNs
    model.measured(48_000_000)
    #expect(model.costNs < 48_000_000 && model.costNs > baseline, "half way up, not all the way")
    model.measured(48_000_000)
    let ticks = Ticks()
    let gate = AddonGate(clock: { ticks.now })
    for _ in 0..<10 {
      gate.frameStarted()
      ticks.advance(ms: 50)
    }
    gate.frameStarted()
    ticks.advance(ms: 15)
    gate.frameEnded()
    #expect(!gate.admit(cost: model.costNs, wait: 0), "too dear for the gap after a 15 ms frame")
    var refused = 1
    model.relax()
    while !gate.admit(cost: model.costNs, wait: 0) {
      model.relax()
      refused += 1
      try #require(refused < 10, "the estimate never came back down")
    }
    gate.passEnded()
    #expect(refused <= 3)
    for _ in 0..<100 { model.relax() }
    #expect(model.costNs == baseline, "never below what it measured")
  }

  @Test("A failure that may not last is tried again before it is kept as failed")
  func transientFailure() throws {
    try serve { server, client in
      let root = server.configuration.cacheRoot
      let flaky = FlakyBackend()
      flaky.failLoads(with: HostError.failed("out of memory"), times: 1)
      let service = AddonService(host: server.host, backend: flaky, cacheRoot: root) { _ in AbsentAddon() }
      service.start()
      defer { service.stop() }
      _ = try client.hello()
      let first = try provision(client, model: Self.addon, sha256: Self.addonSHA)
      #expect(first["state"] as? String == "failed" && first["retry"] as? Bool == true)
      #expect(service.helloState()["failed"] as? [String] == [], "not kept: the comma sends it again")
      #expect(try provision(client, model: Self.addon, sha256: Self.addonSHA)["state"] as? String == "ready")

      flaky.failLoads(with: HostError.failed("out of memory"), times: 2)
      service.keep(["ff/54/59": []] as [String: Any])
      for retry in [true, false] {
        let state = try provision(client, model: Self.addon, sha256: Self.addonSHA)
        #expect(state["state"] as? String == "failed" && (state["retry"] as? Bool ?? false) == retry)
      }
      #expect(service.helloState()["failed"] as? [String] == [Self.addonSHA], "a second failure is kept")
    }
  }

  @Test("A driving model asked for while an add-on builds stops that build, which is left for the next park, not failed")
  func drivingFirst() throws {
    let golden = try Golden("tiny_queued")
    try serve { server, client in
      let root = server.configuration.cacheRoot
      let slow = StoppableBackend()
      let outcome = Outcome()
      let thread = Thread {
        do {
          let model = try AddonModel.prepare(Self.addon, backend: slow, root: root, log: ServerLog(category: "test"), host: server.host)
          model.engine.close()
          outcome.set("built")
        } catch let deferred as AddonDeferred {
          outcome.set("deferred: \(deferred.description)")
        } catch {
          outcome.set("failed: \(error)")
        }
      }
      thread.start()
      try #require(slow.started.wait(timeout: .now() + 5) == .success, "the add-on build never started")
      _ = try client.hello()
      _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
      #expect(server.host.loadedSHA() == golden.sha256, "the driving model is ready")
      let deadline = Date().addingTimeInterval(5)
      while outcome.get() == nil && Date() < deadline { Thread.sleep(forTimeInterval: 0.01) }
      #expect(outcome.get()?.hasPrefix("deferred") == true, "\(outcome.get() ?? "still building")")
      #expect(slow.stopped.get(), "the build was told to stop")
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

/// The CPU backend, whose build runs until it is told to stop (as a TensorRT
/// build is, through its progress monitor), or for 5 s at most.
final class StoppableBackend: EngineBackend, @unchecked Sendable {
  private let inner = cpuBackend()
  let started = DispatchSemaphore(value: 0)
  let stopped = Flag()

  var name: String { inner.name }
  var suffix: String { inner.suffix }
  var runtimeVersion: String { inner.runtimeVersion }
  func deviceTag() -> String { inner.deviceTag() }

  func deriveSpec(model: URL, sha256: String, nbytes: Int64, frameSkip: Int) throws -> ModelSpec {
    try inner.deriveSpec(model: model, sha256: sha256, nbytes: nbytes, frameSkip: frameSkip)
  }

  func build(model: URL, artifact: URL, report: @escaping ProgressFn, metaExtra: [String: Any]) throws {
    try inner.build(model: model, artifact: artifact, report: report, metaExtra: metaExtra)
  }

  func build(
    model: URL, artifact: URL, report: @escaping ProgressFn, metaExtra: [String: Any], shouldStop: @escaping @Sendable () -> Bool
  ) throws {
    started.signal()
    let deadline = Date().addingTimeInterval(5)
    while Date() < deadline {
      if shouldStop() {
        stopped.set(true)
        throw HostError.failed("the build was stopped")
      }
      Thread.sleep(forTimeInterval: 0.01)
    }
    throw HostError.failed("never told to stop")
  }

  func load(artifact: URL, report: @escaping ProgressFn) throws -> any Engine {
    try inner.load(artifact: artifact, report: report)
  }
}

final class Outcome: @unchecked Sendable {
  private let lock = NSLock()
  private var value: String?
  func set(_ new: String) { lock.withLock { value = new } }
  func get() -> String? { lock.withLock { value } }
}
