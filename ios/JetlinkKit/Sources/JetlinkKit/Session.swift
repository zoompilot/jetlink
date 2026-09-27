// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// The request loop for one connection: Session in jetlink/server/session.py.
//
// One client at a time, one message at a time. Builds run on the host's job
// thread and report through here, so sends are serialized. The inference path
// neither allocates nor logs: the queues read the frame where it landed in
// the receive buffer and gather straight into the engine's input buffers.

import Foundation

public final class Session: @unchecked Sendable {
  let conn: TcpConnection
  let host: EngineHost
  private let sendLock = NSLock()
  private let stateLock = NSLock()
  private let log = JLogger("jetlink.server")

  private(set) var client = ""  // who said hello
  private var lastSeq: UInt32 = 0
  private var request: EngineRequest?
  private(set) var frames = 0

  // The frame path's fixed buffers.
  private var packedScratch: UnsafeMutablePointer<Float>
  private var packedCapacity: Int
  private var respBuffer: UnsafeMutableRawPointer
  private var respCapacity: Int

  /// Frames past this server-side are worth a log line (the Python's
  /// SLOW_FRAME_US); the comma's own line fires at 80 ms end to end.
  static let slowFrameUs = 60_000

  public init(conn: TcpConnection, host: EngineHost) {
    self.conn = conn
    self.host = host
    packedCapacity = 1
    packedScratch = .allocate(capacity: 1)
    respCapacity = Wire.inferRespSize
    respBuffer = .allocate(byteCount: respCapacity, alignment: 16)
  }

  deinit {
    packedScratch.deallocate()
    respBuffer.deallocate()
  }

  var currentRequest: EngineRequest? {
    stateLock.lock()
    defer { stateLock.unlock() }
    return request
  }

  private func reset(_ seq: UInt32) {
    stateLock.lock()
    lastSeq = seq
    request = nil
    frames = 0
    stateLock.unlock()
  }

  // MARK: plumbing

  private func send(_ type: Msg, _ seq: UInt32, parts: [UnsafeRawBufferPointer] = []) throws {
    sendLock.lock()
    defer { sendLock.unlock() }
    try conn.send(type, seq: seq, parts: parts)
  }

  private func sendJSON(_ type: Msg, _ seq: UInt32, _ obj: JSON) throws {
    let data = obj.data
    try data.withUnsafeBytes { try send(type, seq, parts: [$0]) }
  }

  private func sendError(_ seq: UInt32, _ error: String, _ detail: String = "") throws {
    log.error("\(error): \(detail)")
    try sendJSON(.error, seq, ["error": .string(error), "detail": .string(detail)])
  }

  /// Build progress, from the job thread. The comma may have given up and
  /// fallen back; the build carries on either way.
  func progress(_ stage: String, _ frac: Double, _ msg: String) {
    try? sendJSON(.progress, 0, ["stage": .string(stage), "frac": .double(pyRound(frac, 4)), "msg": .string(msg)])
  }

  /// The job finished. Tell the client that is here now, whoever it is.
  func engineUpdate() {
    let (sha, skip) = wanted()
    let st = host.status(sha, skip)
    try? sendJSON(.engineResp, 0, st)
  }

  /// The connection is gone. The engine stays: see EngineHost.
  public func close() {
    host.lock.lock()
    if host.session === self { host.session = nil }
    host.lock.unlock()
  }

  /// Serves until the link fails or `stop` says so; checks `stop` at least
  /// once a second.
  public func serve(stop: @escaping () -> Bool) throws {
    while !stop() {
      let msg: Message
      do {
        msg = try conn.receive(timeout: 1.0)
      } catch let e as LinkError where e.isTimeout {
        continue
      }
      do {
        try handle(msg)
      } catch let e as LinkError {
        throw e
      } catch {
        // A bad request must not take the server down.
        log.error("handler failed: \(error)")
        try sendError(msg.seq, String(describing: type(of: error)), String(describing: error))
      }
    }
  }

  func handle(_ msg: Message) throws {
    let type = msg.type
    if type == .helloReq {
      // A hello means a new client process, and is answered whatever the
      // seq says: a client always starts at seq 1.
      greet(msg)
      try onHello(msg)
      return
    }
    // Seqs never repeat on a connection, so anything at or below the last one
    // is a replay; running it would push the same image into the queues twice.
    stateLock.lock()
    if msg.seq <= lastSeq {
      let last = lastSeq
      stateLock.unlock()
      log.warning(
        "dropping replayed message type=\(msg.header.msgType) seq=\(msg.seq) (last \(last)) from \(client.isEmpty ? "an unnamed client" : client)")
      return
    }
    lastSeq = msg.seq
    stateLock.unlock()
    switch type {
    case .inferReq: try onInfer(msg)
    case .ping: try send(.pong, msg.seq)
    case .engineReq: try onEngineReq(msg)
    case .uploadChunk: try onUploadChunk(msg)
    case .uploadDone: try onUploadDone(msg)
    case .stateReq: try onState(msg)
    case .shutdownReq: try onShutdown(msg)
    default: try sendError(msg.seq, "unknown_message", "type \(msg.header.msgType)")
    }
  }

  private func wanted() -> (String?, Int?) {
    stateLock.lock()
    defer { stateLock.unlock() }
    return (request?.sha256, request?.frameSkip)
  }

  private func greet(_ msg: Message) {
    // json.loads(payload or b'{}').get('client') or {}, then name/nonce with
    // defaults; anything unparseable leaves the client unnamed.
    var who = ""
    let parsed = msg.payload.count == 0 ? JSON.object([:]) : (try? JSON.parse(msg.payload))
    if let d = parsed?.object {
      let c = d["client"]
      if c == nil || c == .null || c?.object != nil {
        let name = c?["name"]?.string.flatMap { $0.isEmpty ? nil : $0 } ?? "client"
        let nonce = c?["nonce"]?.string.flatMap { $0.isEmpty ? nil : $0 } ?? "?"
        who = "\(name)/\(nonce)"
      }
    }
    if !client.isEmpty && who != client {
      log.info("session handed from \(client) to \(who.isEmpty ? "an unnamed client" : who)")
    }
    client = who
    reset(msg.seq)
    log.info("hello from \(who.isEmpty ? "an unnamed client" : who) (seq \(msg.seq))")
  }

  private func onHello(_ msg: Message) throws {
    let (sha, skip) = wanted()
    var resp = host.describe()
    resp["protocol"] = .int(Int(Wire.version))
    resp["engine_state"] = host.status(sha, skip)["state"] ?? "none"
    resp["loaded"] = .str(host.loadedSha())
    resp["frames_served"] = .int(frames)
    resp["cached_models"] = .array(host.inventory().map(JSON.string))
    // The phone reports no sensors: nothing is what the comma is told, rather
    // than zeros that would read as a cold idle board.
    resp["telemetry"] = .object([:])
    // A phone never suspends under the comma, so the comma holds the link.
    resp["sleep_after"] = .double(0)
    try sendJSON(.helloResp, msg.seq, .object(resp))
  }

  private func onEngineReq(_ msg: Message) throws {
    let d = try JSON.parse(msg.payload)
    guard let sha = d["sha256"]?.string, let nbytes = d["nbytes"]?.int else {
      throw SpecError("ENGINE_REQ needs sha256 and nbytes")
    }
    let req = try EngineRequest(sha256: sha, nbytes: nbytes, frameSkip: d["frame_skip"]?.int ?? defaultFrameSkip)
    stateLock.lock()
    request = req
    stateLock.unlock()
    try sendJSON(.engineResp, msg.seq, host.request(req, session: self))
  }

  private func onUploadChunk(_ msg: Message) throws {
    guard let req = currentRequest else { return try sendError(msg.seq, "no_model", "send ENGINE_REQ first") }
    guard msg.payload.count >= 8 else { return try sendError(msg.seq, "bad_upload", "missing chunk offset") }
    let offset = Int(UInt64(littleEndian: msg.payload.loadUnaligned(as: UInt64.self)))
    let data = UnsafeRawBufferPointer(rebasing: msg.payload[8...])
    guard offset >= 0, offset + data.count <= req.nbytes else {
      return try sendError(msg.seq, "bad_upload", "chunk exceeds declared model size")
    }
    let path = try host.cache.modelPath(req.sha256)
    // 'r+b' after the first chunk, 'wb' on the first: offset 0 starts the file over.
    let fd = open(path.path, offset == 0 ? (O_WRONLY | O_CREAT | O_TRUNC) : (O_WRONLY | O_CREAT), 0o644)
    if fd < 0 { throw EngineError("cannot write \(path.lastPathComponent): \(String(cString: strerror(errno)))") }
    defer { Darwin.close(fd) }
    var written = 0
    while written < data.count {
      let n = pwrite(fd, data.baseAddress! + written, data.count - written, off_t(offset + written))
      if n < 0 {
        if errno == EINTR { continue }
        throw EngineError("cannot write \(path.lastPathComponent): \(String(cString: strerror(errno)))")
      }
      written += n
    }
    // Silent on purpose: the client streams chunks without reading between them.
  }

  private func onUploadDone(_ msg: Message) throws {
    guard let req = currentRequest else { return try sendError(msg.seq, "no_model", "send ENGINE_REQ first") }
    let path = try host.cache.modelPath(req.sha256)
    let sha = (try? sha256File(path.path).0) ?? ""
    if sha != req.sha256 {
      try? FileManager.default.removeItem(at: path)
      try sendJSON(
        .engineResp, msg.seq,
        ["state": "failed", "detail": "sha256 mismatch after upload", "sha256": .string(req.sha256), "chunk": .int(uploadChunk)])
      return
    }
    progress("upload", 1.0, "verified")
    try sendJSON(.engineResp, msg.seq, host.request(req, session: self))
  }

  private func onState(_ msg: Message) throws {
    let (sha, skip) = wanted()
    let st = host.status(sha, skip)
    try sendJSON(
      .stateResp, msg.seq,
      [
        "engine_state": st["state"] ?? "none", "detail": st["detail"] ?? "", "loaded": .str(host.loadedSha()),
        "frames_served": .int(frames),
      ])
  }

  private func onShutdown(_ msg: Message) throws {
    let reason = (try? JSON.parse(msg.payload))?["reason"]?.string ?? ""
    log.warning("the comma asked to power off: \(reason.isEmpty ? "no reason given" : reason)")
    // A phone is not the comma's to power off; say so rather than pretend.
    try sendJSON(.shutdownResp, msg.seq, ["ok": false, "detail": "an iPhone server cannot be powered off by the comma"])
    host.emit("shutdown_requested", ["reason": .string(reason)])
  }

  // MARK: the hot path

  private func onInfer(_ msg: Message) throws {
    let (sha, skip) = wanted()
    host.lock.lock()
    defer { host.lock.unlock() }
    guard let loaded = host.loaded, loaded.sha256 == sha, loaded.spec.frameSkip == skip, !host.benchmarking else {
      return try respond(msg.seq, frameID: 0, status: .notReady)
    }
    try infer(loaded, msg)
  }

  private func respond(_ seq: UInt32, frameID: UInt32, status: InferStatus) throws {
    let r = InferWire.resp(frameID: frameID, status: status)
    try r.withUnsafeBytes { try send(.inferResp, seq, parts: [$0]) }
  }

  private func infer(_ loaded: Loaded, _ msg: Message) throws {
    let t0 = monotonic()
    let spec = loaded.spec
    if msg.payload.count != spec.inferReqNbytes {
      // The offsets below come from the spec, not the wire: a client on another
      // model would have its scalars read out of the image.
      return try respond(msg.seq, frameID: 0, status: .badShape)
    }
    let base = msg.payload.baseAddress!
    let (frameID, flags) = InferWire.decodeReq(base)
    if Flag(rawValue: flags).contains(.resetQueues) { loaded.queues.reset() }

    let warped = base.advanced(by: Wire.inferReqSize).assumingMemoryBound(to: UInt8.self)
    // The floats follow an odd number of image bytes' worth of offset in the
    // receive buffer, so they are copied out to aligned memory first.
    let n = spec.packedNelem
    if packedCapacity < n {
      packedScratch.deallocate()
      packedScratch = .allocate(capacity: n)
      packedCapacity = n
    }
    UnsafeMutableRawPointer(packedScratch).copyMemory(
      from: base.advanced(by: Wire.inferReqSize + spec.warpedNbytes), byteCount: n * 4)
    loaded.queues.step(warped: warped, packed: packedScratch, into: loaded.hostInputs)
    let queueUs = Int((monotonic() - t0) * 1e6)

    try loaded.engine.run()
    let out = loaded.engine.output()
    // openpilot drops to the small model on a non-finite output either way;
    // checking here saves the comma rescanning the whole vector.
    let finite = Convert.allFinite(out.baseAddress!, count: out.count, fromFloat16: loaded.engine.outputIsFloat16)
    let totalUs = Int((monotonic() - t0) * 1e6)
    let gpuUs = loaded.engine.lastRunUs

    InferWire.encodeResp(
      into: respBuffer, frameID: frameID, status: finite ? .ok : .notFinite, gpuUs: wireMicros(gpuUs),
      queueUs: wireMicros(queueUs), totalUs: wireMicros(totalUs))
    var parts = [
      UnsafeRawBufferPointer(start: respBuffer, count: Wire.inferRespSize),
      UnsafeRawBufferPointer(start: out.baseAddress, count: out.count * 4),
    ]
    let sendStarted = monotonic()
    if Flag(rawValue: flags).contains(.wantState) {
      // No sensors on a phone: the telemetry is an empty object.
      try Session.emptyState.withUnsafeBytes { s in
        parts.append(s)
        try send(.inferResp, msg.seq, parts: parts)
      }
    } else {
      try send(.inferResp, msg.seq, parts: parts)
    }
    let sendUs = Int((monotonic() - sendStarted) * 1e6)
    stateLock.lock()
    frames += 1
    stateLock.unlock()
    if totalUs > Session.slowFrameUs || sendUs > 10_000 {
      log.warning(
        String(
          format: "slow frame %u: gpu %.1f queue %.1f total %.1f send %.1f ms", frameID, Double(gpuUs) / 1e3,
          Double(queueUs) / 1e3, Double(totalUs) / 1e3, Double(sendUs) / 1e3))
    }
    host.frameStats.record(totalUs: totalUs, gpuUs: gpuUs)
  }

  private static let emptyState = [UInt8]("{}".utf8)
}
