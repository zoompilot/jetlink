import Foundation
import JetlinkORT
import JetlinkTestSupport
import Testing

@testable import JetlinkServer

/// The golden files make_server_fixtures.py writes, read in place.
enum Fixture {
  static let directory = TinyModel.fixtures

  static func url(_ name: String) -> URL {
    directory.appending(path: name)
  }

  static func data(_ name: String) throws -> Data {
    try Data(contentsOf: url(name))
  }

  static func json(_ name: String) throws -> [String: Any] {
    try JSONSerialization.jsonObject(with: data(name)) as! [String: Any]
  }
}

/// onnxruntime's CPU provider, which every platform's tests run the model on.
func cpuBackend() -> any EngineBackend {
  OrtBackend(profile: .cpu, preparer: ONNXPreparer(), keepAlive: false)
}

/// A server on the CPU backend, or `backend`, over TCP on loopback, and a
/// client connected to it. `gadget` serves USB beside.
func serve(
  hooks: ServerHooks = ServerHooks(), backend: any EngineBackend = cpuBackend(), gadget: (any GadgetSource)? = nil,
  _ body: (Server, TestClient) throws -> Void
) throws {
  let cache = try TemporaryDirectory()
  let server = try Server(
    configuration: Server.Configuration(host: "127.0.0.1", port: 0, cacheRoot: cache.url, preload: false, usb: gadget != nil),
    backend: backend, gadget: gadget, hooks: hooks)
  try server.start()
  defer { server.shutdown() }
  let client = try TestClient(port: server.port!)
  defer { client.close() }
  try body(server, client)
}

/// The CPU backend with what a TensorRT one adds: fields it describes itself
/// with in the hello, and loads or runs that fail with an error of the
/// test's choosing.
final class FlakyBackend: EngineBackend, @unchecked Sendable {
  private let inner = cpuBackend()
  private let lock = NSLock()
  private var runFailure: (any Error)?
  private var loadFailure: (any Error)?
  private var loadFailures = 0
  private var built = 0
  private var loaded = 0
  private let fields: [String: String]
  private let coolsWhenIdle: Bool

  /// `coolsWhenIdle`: its engines say they do, as a Neural Engine one does.
  init(describing fields: [String: String] = [:], coolsWhenIdle: Bool = false) {
    self.fields = fields
    self.coolsWhenIdle = coolsWhenIdle
  }

  func describe() -> [String: String] { inner.describe().merging(fields) { $1 } }

  var name: String { inner.name }
  var suffix: String { inner.suffix }
  var runtimeVersion: String { inner.runtimeVersion }
  func deviceTag() -> String { inner.deviceTag() }

  /// Every run from now on throws `error`; nil runs the model again.
  func failRuns(with error: (any Error)?) {
    lock.withLock { runFailure = error }
  }

  /// The next `times` loads throw `error`, and the counts start again.
  func failLoads(with error: (any Error)?, times: Int = .max) {
    lock.withLock {
      loadFailure = error
      loadFailures = times
      built = 0
      loaded = 0
    }
  }

  var counts: (builds: Int, loads: Int) { lock.withLock { (built, loaded) } }

  func deriveSpec(model: URL, sha256: String, nbytes: Int64, frameSkip: Int) throws -> ModelSpec {
    try inner.deriveSpec(model: model, sha256: sha256, nbytes: nbytes, frameSkip: frameSkip)
  }

  func build(model: URL, artifact: URL, report: @escaping ProgressFn, metaExtra: [String: Any]) throws {
    lock.withLock { built += 1 }
    try inner.build(model: model, artifact: artifact, report: report, metaExtra: metaExtra)
  }

  func load(artifact: URL, report: @escaping ProgressFn) throws -> any Engine {
    let failure = lock.withLock { () -> (any Error)? in
      loaded += 1
      guard let loadFailure, loadFailures > 0 else { return nil }
      loadFailures -= 1
      return loadFailure
    }
    if let failure { throw failure }
    return FailingEngine(try inner.load(artifact: artifact, report: report), coolsWhenIdle: coolsWhenIdle) { [weak self] in
      self?.lock.withLock { self?.runFailure }
    }
  }
}

/// The engine it loads: the real one, whose run throws when told to.
final class FailingEngine: Engine {
  let inner: any Engine
  let coolsWhenIdle: Bool
  let failure: () -> (any Error)?
  /// Warm runs between sessions so far, counted under the host's lock.
  private(set) var idleRuns = 0

  init(_ inner: any Engine, coolsWhenIdle: Bool = false, failure: @escaping () -> (any Error)?) {
    self.inner = inner
    self.coolsWhenIdle = coolsWhenIdle
    self.failure = failure
  }

  var inputs: [String: TensorSpec] { inner.inputs }
  var outputs: [String: TensorSpec] { inner.outputs }
  var lastGpuUs: UInt32 { inner.lastGpuUs }
  func hostInput(_ name: String) -> UnsafeMutableRawPointer? { inner.hostInput(name) }
  func output(_ name: String) -> UnsafeRawPointer? { inner.output(name) }
  func loopState(_ pairs: [(input: String, output: String)]) throws { try inner.loopState(pairs) }
  func resetState() { inner.resetState() }
  func warm() throws -> String { try inner.warm() }
  func close() { inner.close() }

  func run() throws {
    if let failure = failure() { throw failure }
    try inner.run()
  }

  func runIdle() throws {
    idleRuns += 1
    if let failure = failure() { throw failure }
    try inner.runIdle()
  }
}

/// A message the server sent.
struct Reply {
  let type: UInt16
  let seq: UInt32
  let payload: Data

  var json: [String: Any] {
    (try? JSONSerialization.jsonObject(with: payload)) as? [String: Any] ?? [:]
  }

  var status: UInt32 {
    payload.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: UInt32.self) }
  }
}

/// The comma's side of the wire, as much of it as the tests need: jetlink's
/// client.py, message by message, over TCP (`TestClient`) or the fake USB
/// pipes (`GadgetClient`).
protocol CommaClient: AnyObject {
  /// Sends one message with the next seq, or `explicit`; returns the seq.
  func sendMessage(_ type: Wire.Msg, _ payload: Data, flags: Wire.Flag, seq explicit: UInt32?) throws -> UInt32
  /// The next message the server sent, whatever it is.
  func recv() throws -> Reply
}

extension CommaClient {
  @discardableResult
  func send(_ type: Wire.Msg, _ payload: Data = Data(), flags: Wire.Flag = [], seq explicit: UInt32? = nil) throws -> UInt32 {
    try sendMessage(type, payload, flags: flags, seq: explicit)
  }

  @discardableResult
  func sendJSON(_ type: Wire.Msg, _ object: [String: Any]) throws -> UInt32 {
    try send(type, try JSONSerialization.data(withJSONObject: object))
  }

  /// The next reply of `type`, skipping progress on the way.
  func recv(_ type: Wire.Msg) throws -> Reply {
    while true {
      let reply = try recv()
      if reply.type == type.rawValue { return reply }
      if reply.type == Wire.Msg.progress.rawValue { continue }
      throw TestError("expected \(type), got message type \(reply.type): \(String(decoding: reply.payload, as: UTF8.self))")
    }
  }

  /// ENGINE_REQ, uploading the model when asked, until the engine is ready.
  func ensureEngine(model: URL, sha256: String) throws -> [String: Any] {
    let bytes = try Data(contentsOf: model)
    try sendJSON(.engineReq, ["sha256": sha256, "nbytes": bytes.count, "frame_skip": 4])
    var state = try recv(.engineResp).json
    if state["state"] as? String == "need_upload" {
      // Two chunks, so the offsets are exercised.
      let half = bytes.count / 2
      for (offset, chunk) in [(0, bytes[..<half]), (half, bytes[half...])] {
        var payload = withUnsafeBytes(of: UInt64(offset).littleEndian) { Data($0) }
        payload.append(contentsOf: chunk)
        try send(.uploadChunk, payload)
      }
      try sendJSON(.uploadDone, ["sha256": sha256])
      state = try recv(.engineResp).json
    }
    while state["state"] as? String == "building" {
      state = try recv(.engineResp).json
    }
    guard state["state"] as? String == "ready" else {
      throw TestError("engine not ready: \(state)")
    }
    return state
  }

  /// The hello a comma on this protocol sends.
  func hello(name: String = "test", link: [String: Any]? = nil, addonsKeep: [String: [String]]? = nil) throws -> [String: Any] {
    var client: [String: Any] = ["name": name, "nonce": 1]
    if let link { client["link"] = link }
    if let addonsKeep { client["addons_keep"] = addonsKeep }
    try send(.helloReq, JSONSerialization.data(withJSONObject: ["client": client]))
    return try recv(.helloResp).json
  }

  /// Hello, the model, then Python's golden frames, each reply checked against
  /// Python's output: bit for bit on Apple, by correlation elsewhere and
  /// wherever `exact` is false. Every other frame asks for the whole vector
  /// (WANT_HIDDEN); the rest get it less hidden_state. Returns the hello and
  /// the frames sent.
  @discardableResult
  func replay(_ golden: Golden, exact: Bool = Golden.exact) throws -> (hello: [String: Any], frames: Int) {
    let hello = try hello()
    let ready = try ensureEngine(model: golden.model, sha256: golden.sha256)
    let spec = try ModelSpec.from(ready["spec"] as! [String: Any])
    let frameBytes = golden.frameBytes(spec)
    let count = golden.frames.count / frameBytes
    for i in 0..<count {
      let whole = i % 2 == 1
      var request = withUnsafeBytes(of: UInt32(i).littleEndian) { Data($0) }
      request.append(contentsOf: withUnsafeBytes(of: (whole ? Wire.Flag.wantHidden.rawValue : 0).littleEndian) { Data($0) })
      // the image and the scalars: a queued graph's recorded prev_feat is
      // not sent, the server feeds back its own
      let start = i * frameBytes
      request.append(golden.frames[start..<(start + spec.warpedBytes + spec.packedBytes)])
      try send(.inferReq, request)
      let reply = try recv(.inferResp)
      #expect(reply.status == Wire.Status.ok.rawValue)
      var expected = Data(golden.served[(i * spec.outputBytes)..<((i + 1) * spec.outputBytes)])
      if !whole, let hidden = spec.hiddenRange {
        expected.removeSubrange((hidden.lowerBound * 4)..<(hidden.upperBound * 4))
      }
      let got = Data(reply.payload[Wire.inferRespSize...])
      if exact {
        #expect(got == expected, "frame \(i) differs from Python's by up to \(Golden.worstDifference(got, expected))")
      } else {
        let correlation = Golden.correlation(got, expected)
        #expect(correlation >= 0.999, "frame \(i) correlates \(correlation) with Python's, differing by up to \(Golden.worstDifference(got, expected))")
      }
    }
    return (hello, count)
  }
}

/// The comma over TCP.
final class TestClient: CommaClient {
  let transport: TCPTransport
  private var seq: UInt32 = 0

  init(port: UInt16) throws {
    transport = try TCPTransport.connect(host: "127.0.0.1", port: port)
    transport.setReceiveTimeout(60)
  }

  func sendMessage(_ type: Wire.Msg, _ payload: Data, flags: Wire.Flag, seq explicit: UInt32?) throws -> UInt32 {
    let seq =
      explicit
      ?? {
        self.seq += 1
        return self.seq
      }()
    try transport.send(type, seq: seq, data: [payload], flags: flags)
    return seq
  }

  func recv() throws -> Reply {
    let message = try transport.recv()
    return Reply(type: message.msgType, seq: message.seq, payload: Data(message.payload))
  }

  func close() {
    transport.close()
  }
}

/// A tiny model, its identity, and the frames and outputs Python recorded.
struct Golden {
  /// Whether onnxruntime's CPU provider here computes Python's outputs bit
  /// for bit. Its Android and Linux builds run an fp16 graph's MatMul and
  /// ReduceMean in fp16 where the Apple build does not, so tiny_queued (all
  /// fp16) lands within about 3% of Python's and tiny_stateful (fp32) bit for
  /// bit; there replies are held to what verify_parity asks of a phone.
  static var exact: Bool {
    #if os(Android) || os(Linux)
      false
    #else
      true
    #endif
  }

  /// Pearson correlation of two runs of float32 outputs, as verify_parity
  /// computes it.
  static func correlation(_ a: Data, _ b: Data) -> Double {
    guard a.count == b.count, !a.isEmpty else { return 0 }
    let x = a.withUnsafeBytes { $0.bindMemory(to: Float.self).map(Double.init) }
    let y = b.withUnsafeBytes { $0.bindMemory(to: Float.self).map(Double.init) }
    let mx = x.reduce(0, +) / Double(x.count)
    let my = y.reduce(0, +) / Double(y.count)
    var sxy = 0.0, sxx = 0.0, syy = 0.0
    for (p, q) in zip(x, y) {
      sxy += (p - mx) * (q - my)
      sxx += (p - mx) * (p - mx)
      syy += (q - my) * (q - my)
    }
    return sxx == 0 || syy == 0 ? (x == y ? 1 : 0) : sxy / (sxx * syy).squareRoot()
  }

  /// The largest difference between two runs of float32 outputs, relative to
  /// the larger magnitude where that is over 1.
  static func worstDifference(_ a: Data, _ b: Data) -> Float {
    guard a.count == b.count else { return .infinity }
    let x = a.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    let y = b.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    var worst: Float = 0
    for (p, q) in zip(x, y) {
      if p.isNaN != q.isNaN { return .infinity }
      if p.isNaN { continue }
      worst = max(worst, abs(p - q) / max(1, max(abs(p), abs(q))))
    }
    return worst
  }

  let name: String
  let model: URL
  let sha256: String
  let spec: [String: Any]
  /// Per frame: warped, then the floats protocol 2 sent; for a queued graph
  /// the scalars and then a prev_feat.
  let frames: Data
  /// The outputs Python computed for those frames, prev_feat included.
  let expected: Data
  /// What a comma is served for the same images and scalars, with the
  /// server feeding back its own hidden state: `expected` for a stateful
  /// graph, whose hidden state never left it.
  let served: Data

  init(_ name: String) throws {
    self.name = name
    model = Fixture.url("\(name).onnx")
    spec = try Fixture.json("\(name).spec.json")
    sha256 = spec["sha256"] as! String
    frames = try Fixture.data("\(name).frames.bin")
    expected = try Fixture.data("\(name).expected.bin")
    let fed = Fixture.url("\(name).fed.expected.bin")
    served = FileManager.default.fileExists(atPath: fed.path) ? try Data(contentsOf: fed) : expected
  }

  /// One recorded frame's bytes.
  func frameBytes(_ spec: ModelSpec) -> Int {
    spec.warpedBytes + (spec.packedCount + spec.prevFeatCount) * 4
  }
}
