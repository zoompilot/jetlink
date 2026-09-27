// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.

import XCTest

@testable import JetlinkKit

/// The whole server over TCP, as the comma sees it, with tests/tiny_model.py's
/// model on onnxruntime's CPU provider so it runs anywhere. The outputs are
/// held to what jetlink.queues and onnxruntime make of the same frames on the
/// unmodified model (ios/scripts/server_fixtures.py).
final class ServerTests: XCTestCase {
  var root: URL!

  override func setUpWithError() throws {
    root = FileManager.default.temporaryDirectory.appendingPathComponent("jetlink-tests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: root)
  }

  func fixture(_ name: String) throws -> Data {
    let url = try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: nil, subdirectory: "Fixtures"))
    return try Data(contentsOf: url)
  }

  struct Frames {
    let spec: ModelSpec
    let count: Int
    let input: Data
    let expected: Data

    func request(_ i: Int) -> (reset: Bool, warped: Data, packed: Data) {
      let size = 1 + spec.warpedNbytes + spec.packedNbytes
      let base = input.startIndex + i * size
      return (
        input[base] == 1, input[(base + 1)..<(base + 1 + spec.warpedNbytes)],
        input[(base + 1 + spec.warpedNbytes)..<(base + size)]
      )
    }

    func output(_ i: Int) -> [Float] {
      let n = spec.outputNelem
      return expected[(expected.startIndex + i * n * 4)..<(expected.startIndex + (i + 1) * n * 4)].withUnsafeBytes {
        Array($0.bindMemory(to: Float.self))
      }
    }
  }

  func frames() throws -> Frames {
    let json = try JSON.parse(fixture("tiny_spec.json"))
    return Frames(
      spec: try ModelSpec(json: json), count: try XCTUnwrap(json["frames"]?.int), input: try fixture("tiny_frames.in.bin"),
      expected: try fixture("tiny_frames.out.bin"))
  }

  func startServer(_ preference: ComputePreference = .cpu) throws -> (JetlinkServer, EngineHost) {
    let host = EngineHost(cache: try EngineCache(root: root), preference: preference)
    let server = JetlinkServer(host: host, bindAddress: "127.0.0.1", port: 0)
    try server.start()
    return (server, host)
  }

  // MARK: the tests

  func testAFullSessionMatchesTheReference() throws {
    let f = try frames()
    let model = try fixture("tiny_model.onnx.bin")
    let (server, host) = try startServer()
    defer {
      server.stop()
      host.close()
    }
    let c = try Client(port: server.port)

    let hello = try c.hello()
    XCTAssertEqual(hello["protocol"]?.int, 2)
    XCTAssertEqual(hello["backend"]?.string, "ort")
    XCTAssertEqual(hello["engine_state"]?.string, "none")
    XCTAssertEqual(hello["sleep_after"]?.double, 0)

    let first = try c.engineReq(f.spec.sha256, model.count)
    XCTAssertEqual(first["state"]?.string, "need_upload")
    XCTAssertEqual(first["chunk"]?.int, uploadChunk)

    // Two chunks, the second at an offset, as a small chunk size would send them.
    let half = model.count / 2
    try c.upload(offset: 0, model[model.startIndex..<(model.startIndex + half)])
    try c.upload(offset: half, model[(model.startIndex + half)...])
    let done = try c.uploadDone()
    XCTAssertTrue(["building", "ready"].contains(done["state"]?.string ?? ""), "\(done)")
    let ready = try c.awaitReady()
    XCTAssertEqual(try ModelSpec(json: XCTUnwrap(ready["spec"])), f.spec)
    XCTAssertTrue(c.progressStages.contains("upload"))
    XCTAssertTrue(c.progressStages.contains("load"))

    for i in 0..<f.count {
      let (reset, warped, packed) = f.request(i)
      let (status, frameID, out) = try c.infer(frameID: UInt32(100 + i), reset: reset, warped: warped, packed: packed)
      XCTAssertEqual(status, InferStatus.ok.rawValue)
      XCTAssertEqual(frameID, UInt32(100 + i))
      assertClose(out, f.output(i), "frame \(i)")
    }

    // A replayed seq is dropped: the next message back answers the ping.
    try c.sendRaw(.inferReq, seq: 1, Data(count: 4))
    let pong = try c.ping()
    XCTAssertEqual(pong, Msg.pong.rawValue)

    // A frame of the wrong size is refused rather than misread.
    let (bad, _, _) = try c.infer(frameID: 7, reset: false, warped: Data(count: 10), packed: Data())
    XCTAssertEqual(bad, InferStatus.badShape.rawValue)

    let state = try c.state()
    XCTAssertEqual(state["engine_state"]?.string, "ready")
    XCTAssertEqual(state["loaded"]?.string, f.spec.sha256)

    let shutdown = try c.shutdown()
    XCTAssertEqual(shutdown["ok"]?.bool, false)
    c.close()

    // The engine outlives the connection: a new client finds it ready.
    let c2 = try Client(port: server.port)
    _ = try c2.hello()
    let again = try c2.engineReq(f.spec.sha256, model.count)
    XCTAssertEqual(again["state"]?.string, "ready")
    let (reset, warped, packed) = f.request(0)
    let (status, _, out) = try c2.infer(frameID: 1, reset: reset, warped: warped, packed: packed)
    XCTAssertEqual(status, InferStatus.ok.rawValue)
    assertClose(out, f.output(0), "frame 0 after reconnecting")
    c2.close()
  }

  func testARestartedServerPreloadsWithoutAnUpload() throws {
    let f = try frames()
    let model = try fixture("tiny_model.onnx.bin")
    do {
      let (server, host) = try startServer()
      let c = try Client(port: server.port)
      _ = try c.hello()
      _ = try c.engineReq(f.spec.sha256, model.count)
      try c.upload(offset: 0, model)
      _ = try c.uploadDone()
      _ = try c.awaitReady()
      c.close()
      server.stop()
      host.close()
    }
    // The model file goes too: modeld never carries it, so the engine on
    // disk has to be enough.
    try FileManager.default.removeItem(at: EngineCache(root: root).modelPath(f.spec.sha256))
    let (server, host) = try startServer()
    defer {
      server.stop()
      host.close()
    }
    let c = try Client(port: server.port)
    _ = try c.hello()
    let st = try c.engineReq(f.spec.sha256, model.count)
    XCTAssertNotEqual(st["state"]?.string, "need_upload", "\(st)")
    _ = try c.awaitReady()
    let (reset, warped, packed) = f.request(0)
    let (status, _, out) = try c.infer(frameID: 5, reset: reset, warped: warped, packed: packed)
    XCTAssertEqual(status, InferStatus.ok.rawValue)
    assertClose(out, f.output(0), "frame 0 after a restart")
  }

  /// The tiny model records no intermediate shapes, which the Python fills in
  /// with shape inference and this cannot: CoreML still has to serve it, with
  /// its MatMul left as it is (found running the app on a simulator).
  func testCoreMLServesAModelThatRecordsNoShapes() throws {
    let f = try frames()
    let model = try fixture("tiny_model.onnx.bin")
    let (server, host) = try startServer(.coreml)
    defer {
      server.stop()
      host.close()
    }
    let c = try Client(port: server.port)
    _ = try c.hello()
    _ = try c.engineReq(f.spec.sha256, model.count)
    try c.upload(offset: 0, model)
    _ = try c.uploadDone()
    _ = try c.awaitReady()
    for i in 0..<f.count {
      let (reset, warped, packed) = f.request(i)
      let (status, _, out) = try c.infer(frameID: UInt32(i), reset: reset, warped: warped, packed: packed)
      XCTAssertEqual(status, InferStatus.ok.rawValue)
      // CoreML's fp16 on the GPU accumulates differently from the CPU reference.
      let worst = zip(out, f.output(i)).map { abs($0 - $1) }.max() ?? 0
      let scale = f.output(i).map(abs).max() ?? 1
      XCTAssertLessThanOrEqual(worst, 0.02 * max(scale, 1), "frame \(i): max abs difference \(worst)")
    }
  }

  func testACorruptUploadIsRejected() throws {
    let f = try frames()
    var model = try fixture("tiny_model.onnx.bin")
    model[model.startIndex + 1000] ^= 0xFF
    let (server, host) = try startServer()
    defer {
      server.stop()
      host.close()
    }
    let c = try Client(port: server.port)
    _ = try c.hello()
    _ = try c.engineReq(f.spec.sha256, model.count)
    try c.upload(offset: 0, model)
    let done = try c.uploadDone()
    XCTAssertEqual(done["state"]?.string, "failed")
    XCTAssertEqual(done["detail"]?.string, "sha256 mismatch after upload")
    XCTAssertFalse(FileManager.default.fileExists(atPath: try EngineCache(root: root).modelPath(f.spec.sha256).path))
  }

  func testInferBeforeAnEngineIsNotReady() throws {
    let f = try frames()
    let (server, host) = try startServer()
    defer {
      server.stop()
      host.close()
    }
    let c = try Client(port: server.port)
    _ = try c.hello()
    let (reset, warped, packed) = f.request(0)
    let (status, _, _) = try c.infer(frameID: 3, reset: reset, warped: warped, packed: packed)
    XCTAssertEqual(status, InferStatus.notReady.rawValue)
  }

  func assertClose(_ got: [Float], _ want: [Float], _ what: String, file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(got.count, want.count, what, file: file, line: line)
    var worst: Float = 0
    for (a, b) in zip(got, want) { worst = max(worst, abs(a - b)) }
    // Both sides are onnxruntime 1.29.0 on the CPU over float16; the Swift
    // side's graph has the image casts and the layout hint removed, which
    // changes no value.
    XCTAssertLessThanOrEqual(worst, 1e-3, "\(what): max abs difference \(worst)", file: file, line: line)
  }
}

/// A comma's side of the protocol, enough for the tests.
final class Client {
  let conn: TcpConnection
  var seq: UInt32 = 0
  var progressStages: Set<String> = []
  var lastEngine: JSON?

  init(port: UInt16) throws { conn = try TcpConnection.connect(host: "127.0.0.1", port: port) }

  func close() { conn.close() }

  func next() -> UInt32 {
    seq += 1
    return seq
  }

  func sendRaw(_ type: Msg, seq: UInt32, _ data: Data) throws { try conn.send(type, seq: seq, data: data) }

  /// The reply to `seq`, dealing with progress and engine updates on the way.
  func expect(_ type: Msg, seq: UInt32, timeout: Double = 30) throws -> (UInt16, Data) {
    let end = monotonic() + timeout
    while true {
      let m = try conn.receive(timeout: max(0.01, end - monotonic()))
      let data = m.data
      if m.header.msgType == type.rawValue && m.seq == seq { return (m.header.msgType, data) }
      try note(m, data)
    }
  }

  private func note(_ m: Message, _ data: Data) throws {
    switch m.type {
    case .progress: if let s = try JSON.parse(data)["stage"]?.string { progressStages.insert(s) }
    case .engineResp: lastEngine = try JSON.parse(data)
    case .error: throw EngineError("server error: \(String(decoding: data, as: UTF8.self))")
    default: break
    }
  }

  func hello() throws -> JSON {
    let s = next()
    try conn.sendJSON(.helloReq, seq: s, ["client": ["nonce": "abcd1234", "name": "tests"]])
    return try JSON.parse(expect(.helloResp, seq: s).1)
  }

  func engineReq(_ sha: String, _ nbytes: Int) throws -> JSON {
    let s = next()
    try conn.sendJSON(.engineReq, seq: s, ["sha256": .string(sha), "nbytes": .int(nbytes), "frame_skip": 4])
    let r = try JSON.parse(expect(.engineResp, seq: s).1)
    lastEngine = r
    return r
  }

  func upload(offset: Int, _ bytes: Data) throws {
    var header = UInt64(offset).littleEndian
    let off = Data(bytes: &header, count: 8)
    try off.withUnsafeBytes { o in
      try bytes.withUnsafeBytes { b in try conn.send(.uploadChunk, seq: next(), parts: [o, b]) }
    }
  }

  func uploadDone() throws -> JSON {
    let s = next()
    try conn.sendJSON(.uploadDone, seq: s, [:])
    let r = try JSON.parse(expect(.engineResp, seq: s, timeout: 60).1)
    lastEngine = r
    return r
  }

  /// Waits for an engine update that says ready, as ensure_engine does.
  func awaitReady(timeout: Double = 120) throws -> JSON {
    let end = monotonic() + timeout
    while monotonic() < end {
      if let e = lastEngine, e["state"]?.string == "ready" { return e }
      if let e = lastEngine, e["state"]?.string == "failed" { throw EngineError("engine failed: \(e)") }
      do {
        let m = try conn.receive(timeout: 1)
        try note(m, m.data)
      } catch let e as LinkError where e.isTimeout {
        continue
      }
    }
    throw EngineError("engine not ready in \(timeout) s")
  }

  func infer(frameID: UInt32, reset: Bool, warped: Data, packed: Data) throws -> (UInt32, UInt32, [Float]) {
    let s = next()
    var req = Data(count: 8)
    req.withUnsafeMutableBytes {
      $0.storeBytes(of: frameID.littleEndian, toByteOffset: 0, as: UInt32.self)
      $0.storeBytes(of: (reset ? Flag.resetQueues.rawValue : 0).littleEndian, toByteOffset: 4, as: UInt32.self)
    }
    try req.withUnsafeBytes { r in
      try warped.withUnsafeBytes { w in
        try packed.withUnsafeBytes { p in try conn.send(.inferReq, seq: s, parts: [r, w, p]) }
      }
    }
    let (_, data) = try expect(.inferResp, seq: s)
    let (fid, status): (UInt32, UInt32) = data.withUnsafeBytes {
      (UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self)), UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: 4, as: UInt32.self)))
    }
    let out: [Float] = data.dropFirst(Wire.inferRespSize).withUnsafeBytes { raw in
      (0..<(raw.count / 4)).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: 4 * $0, as: UInt32.self))) }
    }
    return (status, fid, out)
  }

  func ping() throws -> UInt16 {
    let s = next()
    try conn.send(.ping, seq: s)
    let m = try conn.receive(timeout: 5)
    XCTAssertEqual(m.seq, s)
    return m.header.msgType
  }

  func state() throws -> JSON {
    let s = next()
    try conn.sendJSON(.stateReq, seq: s, [:])
    return try JSON.parse(expect(.stateResp, seq: s).1)
  }

  func shutdown() throws -> JSON {
    let s = next()
    try conn.sendJSON(.shutdownReq, seq: s, ["reason": "tests"])
    return try JSON.parse(expect(.shutdownResp, seq: s).1)
  }
}

final class BenchmarkTests: XCTestCase {
  func testABenchmarkRunsPacedAndCanBeStopped() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("jetlink-bench-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let cache = try EngineCache(root: root)
    let host = EngineHost(cache: cache, preference: .cpu)
    defer { host.close() }
    XCTAssertThrowsError(try host.benchmark(seconds: 1, run: BenchmarkRun()) { _ in }, "nothing loaded yet")

    let model = try Data(contentsOf: XCTUnwrap(Bundle.module.url(forResource: "tiny_model.onnx.bin", withExtension: nil, subdirectory: "Fixtures")))
    let (sha, n) = (hex(SHA256Hash(model)), model.count)
    try model.write(to: cache.modelPath(sha))
    let ready = expectation(description: "ready")
    host.subscribe { kind, payload in
      if kind == "engine", payload["state"]?.string == "ready" { ready.fulfill() }
    }
    host.request(try EngineRequest(sha256: sha, nbytes: n, frameSkip: 4), session: nil)
    wait(for: [ready], timeout: 60)

    let report = try host.benchmark(seconds: 2, run: BenchmarkRun()) { _ in }
    // 20 Hz for 2 s, give or take a frame at either end
    XCTAssertTrue((38...42).contains(report.frames), "\(report.frames) frames")
    XCTAssertEqual(report.units, .cpu)
    XCTAssertFalse(report.windows.isEmpty)
    XCTAssertGreaterThan(report.frame.mean, 0)
    XCTAssertTrue(report.text.contains("frames at 20 Hz"))

    let run = BenchmarkRun()
    DispatchQueue.global().asyncAfter(deadline: .now() + 0.5) { run.cancel() }
    let stopped = try host.benchmark(seconds: 60, run: run) { _ in }
    XCTAssertTrue(stopped.cancelled)
    XCTAssertLessThan(stopped.frames, 40)
  }
}

import CryptoKit

func SHA256Hash(_ d: Data) -> [UInt8] { Array(CryptoKit.SHA256.hash(data: d)) }
