import Foundation
import JetlinkKit
import JetlinkRegistry

/// One client's connection: the request loop, the Swift form of
/// `session.Session`. One message at a time; builds run on the host's job
/// thread so progress keeps flowing. The inference path neither allocates
/// nor logs on the way to the reply.
final class Session: @unchecked Sendable {
  private let transport: any MessageLink
  private let host: EngineHost
  private let log = ServerLog(category: "session")
  private var client = ""
  /// Has a client said hello on this connection? A comma pinging a session
  /// with neither a hello nor a model request joined an earlier one.
  private var greeted = false
  private var lastSeq: UInt32 = 0
  private(set) var request: Request?
  /// The add-on model this session asked for last, by interface: its uploads go
  /// there until the next ENGINE_REQ, which never replaces `request`.
  private var addonRequest: (request: Request, kind: [UInt8], name: String?)?
  private(set) var frames = 0
  /// The frames this session served, for its summary when it ends.
  private var trips = SessionTrips()
  /// Has this connection reported a link? At once for a connection someone
  /// made, on the first message over USB (`MessageLink.connectsOnOpen`). A
  /// session that ends unannounced was a gadget nobody on the comma served.
  private(set) var announced = false
  /// How the link is carried: the transport's view until the comma's hello
  /// says better.
  private(set) var medium: LinkMedium?
  /// The name the comma's hello gave, for the link event.
  private var helloName: String?
  /// Hears the link event when the session announces it (`first`), and
  /// again when a later hello changes its medium or name.
  var onLink: ((_ event: LinkEvent, _ first: Bool) -> Void)?
  /// Hears every message that arrives, before it is handled, and the push of
  /// a finished build (on the job's thread).
  var onMessage: (() -> Void)?

  /// The reply's float32 outputs, reused every frame.
  private var outputBuffer: UnsafeMutablePointer<Float>
  private var outputCapacity: Int
  /// The reply's fixed head, packed in place every frame.
  private let responseHead: UnsafeMutableRawPointer
  /// The reply's parts: head, the outputs either side of hidden_state, and
  /// the telemetry when asked for.
  private let parts: UnsafeMutablePointer<UnsafeRawBufferPointer>
  private static let maxParts = 4

  init(transport: any MessageLink, host: EngineHost) {
    self.transport = transport
    medium = transport.medium
    self.host = host
    outputCapacity = 18_452
    outputBuffer = .allocate(capacity: outputCapacity)
    responseHead = .allocate(byteCount: Wire.inferRespSize, alignment: 8)
    parts = .allocate(capacity: Session.maxParts)
    parts.initialize(repeating: UnsafeRawBufferPointer(start: nil, count: 0), count: Session.maxParts)
  }

  deinit {
    outputBuffer.deallocate()
    responseHead.deallocate()
    parts.deallocate()
  }

  var peer: String { transport.peer }
  /// Who said hello here, as the log names them.
  var who: String { client.isEmpty ? "an unnamed client" : client }

  /// One line on what this session served, or nil when it served nothing.
  var summary: String? { trips.summary }

  // MARK: plumbing

  private func send(_ type: Wire.Msg, seq: UInt32, parts: [UnsafeRawBufferPointer] = [], flags: Wire.Flag = []) throws {
    try transport.send(type, seq: seq, parts: parts, flags: flags)
  }

  private func sendJSON(_ type: Wire.Msg, seq: UInt32, _ object: [String: Any]) throws {
    try transport.sendJSON(type, seq: seq, object)
  }

  private func error(_ seq: UInt32, _ error: String, _ detail: String = "") throws {
    log.error("\(error): \(detail)")
    try sendJSON(.error, seq: seq, ["error": error, "detail": detail])
  }

  /// Sent from the job thread. The comma may have given up and fallen back;
  /// the build continues either way.
  func progress(_ stage: String, _ frac: Double, _ msg: String) {
    try? sendJSON(.progress, seq: 0, ["stage": stage, "frac": pythonRound(frac, 4), "msg": msg])
  }

  /// The worker finished. Tell the client that is here now, whoever it is.
  /// A comma waiting on a build sends nothing for minutes, so the push counts
  /// as hearing from it: its link is not back at stock for the swap it waits for.
  func engineUpdate() {
    onMessage?()
    try? respondEngine(0)
  }

  /// An add-on's build ended: its ENGINE_RESP, which the comma tells from the driving model's by sha256.
  func addonUpdate(_ state: [String: Any]) {
    onMessage?()
    try? sendJSON(.engineResp, seq: 0, state)
  }

  /// The connection is gone. The engine stays.
  func close() {
    host.lock.lock()
    if host.session === self {
      host.session = nil
    }
    host.lock.unlock()
    transport.close()
  }

  /// Wakes the request loop from another thread, which then returns.
  func interrupt() {
    transport.shutdown()
  }

  // MARK: the loop

  var linkEvent: LinkEvent {
    LinkEvent(state: .connected, detail: "", peer: peer, medium: medium?.rawValue, client: helloName)
  }

  private func announce() {
    announced = true
    onLink?(linkEvent, true)
  }

  /// Serves until the link fails, and returns why.
  func serveForever() -> String {
    if transport.connectsOnOpen {
      announce()
    }
    while true {
      let message: Message
      do {
        message = try transport.recv()
        if !announced {
          // Over USB the first message is the hello: the link is announced
          // once, with what it says, rather than twice.
          if message.msgType == Wire.Msg.helloReq.rawValue {
            _ = adopt(hello: Session.helloClient(message))
          }
          announce()
        }
      } catch let error as LinkError {
        if case .timedOut = error { continue }
        log.info("link closed: \(error.description)")
        return error.description
      } catch {
        return String(describing: error)
      }
      onMessage?()
      do {
        try handle(message)
      } catch let error as LinkError {
        return error.description
      } catch {
        // A bad request must not take the server down.
        log.error("handler failed: \(String(describing: error))")
        if (try? self.error(message.seq, String(describing: type(of: error)), String(describing: error))) == nil {
          return "could not report an error to the client"
        }
      }
    }
  }

  func handle(_ message: Message) throws {
    guard let type = Wire.Msg(rawValue: message.msgType) else {
      try error(message.seq, "unknown_message", "type \(message.msgType)")
      return
    }
    if message.viaDatagram {
      // Only frames come as datagrams, and those never repeat (FrameDatagrams
      // drops a seq at or below the last it finished). Their seqs are left out
      // of lastSeq: a TCP message sent before a frame can arrive after it.
      guard type == .inferReq else { return }
      try onInfer(message)
      return
    }
    if type == .helloReq {
      // A hello means "a new client process", answered whatever the seq says.
      try onHello(message, client: greet(message))
      return
    }
    // Seqs never repeat on a connection, so anything at or below the last one
    // is a replay; running it would push the same image into the queues twice.
    if message.seq <= lastSeq {
      log.warning("dropping replayed message type=\(message.msgType) seq=\(message.seq) (last \(lastSeq))")
      return
    }
    lastSeq = message.seq
    switch type {
    case .inferReq: try onInfer(message)
    case .ping: try onPing(message)
    case .engineReq: try onEngineReq(message)
    case .uploadChunk: try onUploadChunk(message)
    case .uploadDone: try onUploadDone(message)
    case .stateReq: try onState(message)
    case .shutdownReq: try onShutdown(message)
    case .leave: onLeave(message)
    default: try error(message.seq, "unknown_message", "type \(message.msgType)")
    }
  }

  private func wanted() -> (String?, Int?) {
    guard let request else { return (nil, nil) }
    return (request.sha256, request.frameSkip)
  }

  /// A hello's `client`: who is calling and what its link is.
  private static func helloClient(_ message: Message) -> [String: Any]? {
    JSONLine.decode(message.payload)?["client"] as? [String: Any]
  }

  /// Takes a hello's name, and its link's medium (`adopt(_:)`). Whether
  /// either changed.
  private func adopt(hello client: [String: Any]?) -> Bool {
    let name = (client?["name"] as? String).flatMap { $0.isEmpty ? nil : $0 }
    let renamed = name != helloName
    helloName = name
    return adopt(LinkMedium(link: client?["link"] as? [String: Any])) || renamed
  }

  /// Takes the hello's word on the link where it says more: the cable's
  /// speed over TCP, say. A USB speed the host read off the bus stands
  /// against a hello that names none. Whether the medium changed.
  private func adopt(_ said: LinkMedium?) -> Bool {
    guard let said, said != medium else { return false }
    if said == .usb, let medium, [.usb3, .usb2, .usb1].contains(medium) { return false }
    medium = said
    return true
  }

  /// Starts the session over for the hello's client, which it returns.
  private func greet(_ message: Message) -> [String: Any]? {
    let hello = Session.helloClient(message)
    if adopt(hello: hello), announced {
      onLink?(linkEvent, false)
    }
    var who = ""
    if let hello {
      who = "\(helloName ?? "client")/\(hello["nonce"].map { "\($0)" } ?? "?")"
    }
    let previous = client
    client = who
    if !previous.isEmpty && previous != client {
      log.info("session handed from \(previous) to \(self.who)")
    }
    greeted = true
    lastSeq = message.seq
    request = nil
    addonRequest = nil
    frames = 0
    host.lock.lock()
    host.loaded?.staging.newClient()
    host.lastSeenAt = ProcessInfo.processInfo.systemUptime
    host.lock.unlock()
    log.info("hello from \(self.who) (seq \(message.seq))")
    return hello
  }

  private func onHello(_ message: Message, client hello: [String: Any]?) throws {
    let (sha, skip) = wanted()
    var response: [String: Any] = [
      "protocol": Int(Wire.version),
      "engine_state": host.status(sha, frameSkip: skip)["state"] ?? "none",
      "loaded": host.loadedSHA() ?? NSNull(),
      "frames_served": frames,
      "cached_models": host.cache.inventory(),
      "telemetry": host.telemetry.read(),
      // 0 unless the host really suspends: the comma then holds the gadget
      // for the whole park instead of letting go for a box that never sleeps.
      "sleep_after": host.hooks.sleepAfter,
      // The newer of this server's pick and the comma's; a comma whose own
      // pick is older takes this one (PickStore).
      "pick": PickStore.wire(host.picks.meet(hello?["pick"] as? [String: Any])),
      // Whether a model neither end has is downloaded here when the comma asks:
      // a comma parked without this server fetches a new pick itself only if not.
      "fetches": host.fetcher?.canFetch ?? false,
    ]
    for (key, value) in host.backend.describe() {
      response[key] = value
    }
    // its presence says add-on ENGINE_REQs are taken: an older server would replace its driving model with one
    if let addons = host.addons {
      response["addons"] = addons.helloState()
    }
    // Over a phone's cable, frames may come as datagrams: a new token each
    // hello, so pieces of the last session's frames count for nothing
    if let offer = transport.offerDatagrams() {
      response["frame_port"] = Int(offer.port)
      response["frame_token"] = Int(offer.token)
    }
    try sendJSON(.helloResp, seq: message.seq, response)
  }

  /// PONG to a client this session knows: one that said hello here or asked
  /// for a model. A comma waiting for a window to swap only pings, so after a
  /// server restart, or a USB session the server reopened under it, a PONG
  /// would say all is well until its first frame came back NOT_READY and
  /// cost a demotion. The error makes it rejoin now: a hello, then its model
  /// request.
  private func onPing(_ message: Message) throws {
    guard greeted || request != nil else {
      try error(message.seq, "no_hello", "no hello on this connection; the server restarted or reopened the link since, so say hello again")
      return
    }
    try send(.pong, seq: message.seq)
  }

  private func onEngineReq(_ message: Message) throws {
    guard let d = JSONLine.decode(message.payload), let sha = d["sha256"] as? String,
      let nbytes = (d["nbytes"] as? NSNumber)?.int64Value
    else {
      try error(message.seq, "bad_request", "ENGINE_REQ needs sha256 and nbytes")
      return
    }
    if let addon = d["addon"] as? [String: Any] {
      try onAddonReq(message, sha: sha, nbytes: nbytes, addon: addon)
      return
    }
    addonRequest = nil
    let skip = (d["frame_skip"] as? NSNumber)?.intValue ?? ModelConstants.defaultFrameSkip
    let request = try Request(sha256: sha, nbytes: nbytes, frameSkip: skip, fetch: (d["fetch"] as? Bool) ?? false, ref: d["ref"] as? String)
    self.request = request
    try sendJSON(.engineResp, seq: message.seq, host.request(request, session: self))
  }

  /// An add-on model, for its interface: never the driving model's request, nor its engine.
  private func onAddonReq(_ message: Message, sha: String, nbytes: Int64, addon: [String: Any]) throws {
    guard let service = host.addons else {
      try error(message.seq, "no_addons", "this server serves no add-ons")
      return
    }
    guard let kind = (addon["interface"] as? String).flatMap(AddonWire.interfaceClass) else {
      try error(message.seq, "bad_request", "an add-on needs its interface, like ff/56/41")
      return
    }
    let request = try Request(sha256: sha, nbytes: nbytes, frameSkip: ModelConstants.defaultFrameSkip)
    let name = addon["name"] as? String
    addonRequest = (request, kind, name)
    try sendJSON(.engineResp, seq: message.seq, service.request(sha256: sha, nbytes: nbytes, interface: kind, name: name, session: self))
  }

  /// Where an upload goes: the add-on asked for last, else the driving model.
  private var uploading: Request? { addonRequest?.request ?? request }

  private func respondEngine(_ seq: UInt32) throws {
    let (sha, skip) = wanted()
    try sendJSON(.engineResp, seq: seq, host.status(sha, frameSkip: skip))
  }

  private func onUploadChunk(_ message: Message) throws {
    guard let request = uploading else {
      try error(message.seq, "no_model", "send ENGINE_REQ first")
      return
    }
    guard message.payload.count >= 8 else {
      try error(message.seq, "bad_upload", "missing chunk offset")
      return
    }
    let offset = UInt64(littleEndian: message.payload.loadUnaligned(as: UInt64.self))
    let data = UnsafeRawBufferPointer(rebasing: message.payload[8...])
    if offset + UInt64(data.count) > UInt64(request.nbytes) {
      try error(message.seq, "bad_upload", "chunk exceeds declared model size")
      return
    }
    let path = host.cache.modelPath(request)
    if offset == 0 || !FileManager.default.fileExists(atPath: path.path) {
      FileManager.default.createFile(atPath: path.path, contents: nil)
    }
    let handle = try FileHandle(forWritingTo: path)
    defer { try? handle.close() }
    if offset == 0 {
      try handle.truncate(atOffset: 0)
    }
    try handle.seek(toOffset: offset)
    try handle.write(contentsOf: data)
    // Silent on purpose: the client streams chunks without reading between them.
  }

  private func onUploadDone(_ message: Message) throws {
    guard let request = uploading else {
      try error(message.seq, "no_model", "send ENGINE_REQ first")
      return
    }
    let path = host.cache.modelPath(request)
    let digest = (try? Registry.hashFile(path))?.0
    if digest != request.sha256 {
      try? FileManager.default.removeItem(at: path)
      try sendJSON(
        .engineResp, seq: message.seq,
        [
          "state": "failed", "detail": "sha256 mismatch after upload", "sha256": request.sha256, "chunk": ModelConstants.chunk,
        ])
      return
    }
    progress("upload", 1, "verified")
    if let addon = addonRequest {
      guard let service = host.addons else {
        try error(message.seq, "no_addons", "this server serves no add-ons")
        return
      }
      try sendJSON(
        .engineResp, seq: message.seq,
        service.request(sha256: request.sha256, nbytes: request.nbytes, interface: addon.kind, name: addon.name, session: self))
      return
    }
    try sendJSON(.engineResp, seq: message.seq, host.request(request, session: self))
  }

  // MARK: the hot path

  /// Runs the frame under the host's lock, then replies outside it: over USB
  /// the write waits for the comma to read, and nothing else that needs the
  /// host (status, a build's progress, the control channel) should wait with it.
  private func onInfer(_ message: Message) throws {
    let (sha, skip) = wanted()
    host.addonGate.frameStarted()
    host.lock.lock()
    guard let loaded = host.loaded, loaded.sha256 == sha, loaded.spec.frameSkip == skip, !host.benchmarking else {
      host.lock.unlock()
      host.addonGate.frameEnded()
      try respond(message.seq, InferReply(status: .notReady))
      return
    }
    let reply = infer(loaded, message)
    host.lock.unlock()
    host.addonGate.frameEnded()

    let state: Data? = reply.wantsState ? host.telemetry.json() : nil
    let sendStarted = DispatchTime.now().uptimeNanoseconds
    do {
      // After the reply, so the comma hears INFER_FAILED rather than a
      // timeout, and after a reply that could not be written too (the comma
      // already gave up): the device stays broken for the next session.
      defer {
        if let failure = reply.failure, (failure as? any FatalEngineError)?.isFatal == true {
          log.error("the engine cannot recover from this: \(String(describing: failure))")
          host.hooks.fatal?(failure)
        }
      }
      try respond(message.seq, reply, state: state)
    }
    guard reply.ran else { return }
    let sendUs = microseconds(since: sendStarted)
    frames += 1
    trips.record(totalUs: reply.totalUs)
    if reply.totalUs > FrameStats.slowUs || sendUs > 10_000 {
      log.warning(
        "slow frame \(reply.frameID): gpu \(Double(reply.gpuUs) / 1000) queue \(Double(reply.queueUs) / 1000) total \(Double(reply.totalUs) / 1000) send \(Double(sendUs) / 1000) ms"
      )
    }
    host.frameStats.record(totalUs: reply.totalUs, gpuUs: reply.gpuUs, queueUs: reply.queueUs, sendUs: sendUs)
  }

  /// What an INFER_RESP says; the outputs are in `outputBuffer`.
  private struct InferReply {
    var frameID: UInt32 = 0
    var status: Wire.Status
    var gpuUs: UInt32 = 0
    var queueUs: UInt32 = 0
    var totalUs: UInt32 = 0
    /// Floats of `outputBuffer` the reply carries, 0 for none.
    var outputCount = 0
    /// Left out of those, unless the comma asked for them.
    var hidden: Range<Int>?
    var wantsState = false
    /// The model ran, so the frame counts and is timed.
    var ran = false
    /// Why it failed, when it did.
    var failure: (any Error)?
  }

  /// INFER_RESP: the head, the output buffer less hidden_state, and the
  /// telemetry, in one write from buffers this session owns. The outputs go
  /// as the runs either side of hidden_state, so leaving it out copies nothing.
  private func respond(_ seq: UInt32, _ reply: InferReply, state: Data? = nil) throws {
    Wire.packInferResp(
      frameID: reply.frameID, status: reply.status, gpuUs: reply.gpuUs, queueUs: reply.queueUs, totalUs: reply.totalUs, into: responseHead)
    parts[0] = UnsafeRawBufferPointer(start: responseHead, count: Wire.inferRespSize)
    var count = 1
    if let hidden = reply.hidden {
      parts[1] = UnsafeRawBufferPointer(start: outputBuffer, count: hidden.lowerBound * 4)
      parts[2] = UnsafeRawBufferPointer(start: outputBuffer + hidden.upperBound, count: (reply.outputCount - hidden.upperBound) * 4)
      count = 3
    } else if reply.outputCount > 0 {
      parts[1] = UnsafeRawBufferPointer(start: outputBuffer, count: reply.outputCount * 4)
      count = 2
    }
    if let state {
      try state.withUnsafeBytes { bytes in
        parts[count] = bytes
        try transport.sendParts(.inferResp, seq: seq, parts: UnsafeBufferPointer(start: parts, count: count + 1), flags: [])
      }
    } else {
      try transport.sendParts(.inferResp, seq: seq, parts: UnsafeBufferPointer(start: parts, count: count), flags: [])
    }
  }

  /// The frame itself: stage, run, read the output back. Caller holds `host.lock`.
  private func infer(_ loaded: Loaded, _ message: Message) -> InferReply {
    let started = DispatchTime.now().uptimeNanoseconds
    host.lastSeenAt = ProcessInfo.processInfo.systemUptime
    let layout = loaded.staging.layout
    guard message.payload.count == layout.requestBytes else {
      // The offsets below come from the spec, not the wire: a client on
      // another model would have its scalars read out of the image.
      return InferReply(status: .badShape)
    }
    let base = message.payload.baseAddress!
    let frameID = UInt32(littleEndian: base.loadUnaligned(as: UInt32.self))
    let flags = Wire.Flag(rawValue: UInt32(littleEndian: base.loadUnaligned(fromByteOffset: 4, as: UInt32.self)))
    if flags.contains(.resetQueues) {
      loaded.staging.reset()
    }
    let warped = base + Wire.inferReqSize

    var status = Wire.Status.ok
    var queueUs: UInt32 = 0
    var failure: (any Error)?
    do {
      // The packed floats stay where they arrived: every cast and copy of
      // them reads unaligned, so they need no aligned copy first.
      try loaded.staging.stage(warped: warped, packed: warped + layout.warpedBytes)
      queueUs = microseconds(since: started)
      try loaded.engine.run()
    } catch {
      log.error("inference failed: \(String(describing: error))")
      status = .inferFailed
      failure = error
    }

    let count = layout.outputCount
    if count > outputCapacity {
      outputBuffer.deallocate()
      outputCapacity = count
      outputBuffer = .allocate(capacity: count)
    }
    if status == .ok, let type = layout.outputType, let out = layout.output {
      // float32 on the wire whatever the graph says; openpilot drops to the
      // small model on a non-finite output either way, so say so here.
      switch type {
      case .float:
        outputBuffer.update(from: out.assumingMemoryBound(to: Float.self), count: count)
      case .float16:
        Convert.f16ToF32(out, outputBuffer, count: count)
      default:
        status = .inferFailed
      }
      if status == .ok && !Convert.allFinite(outputBuffer, count: count) {
        status = .notFinite
      }
      if status == .ok {
        loaded.staging.keep(outputs: out, type: type)
      }
    } else if status == .ok {
      status = .inferFailed
    }

    let replied = status == .ok || status == .notFinite
    return InferReply(
      frameID: frameID, status: status, gpuUs: loaded.engine.lastGpuUs, queueUs: queueUs, totalUs: microseconds(since: started),
      outputCount: replied ? count : 0, hidden: replied && !flags.contains(.wantHidden) ? layout.hidden : nil,
      wantsState: flags.contains(.wantState), ran: true, failure: failure)
  }

  private func onShutdown(_ message: Message) throws {
    let reason = (JSONLine.decode(message.payload)?["reason"] as? String) ?? ""
    log.warning("shutdown requested by the client: \(reason.isEmpty ? "no reason given" : reason)")
    if let powerOff = host.hooks.shutdown?(reason) {
      try sendJSON(.shutdownResp, seq: message.seq, ["ok": true, "detail": "powering off"])
      powerOff()
      return
    }
    // A phone does not power itself off for the comma; say so rather than
    // pretend, and let the app tell the person the comma asked.
    try sendJSON(.shutdownResp, seq: message.seq, ["ok": false, "detail": "this server cannot power its device off"])
    host.emit(.shutdownRequested(reason: reason))
  }

  /// The comma says why it stops using the link, and what it measured over
  /// it: the whole frame as modeld waited on it, which this end never sees.
  /// Logged next to this session's own numbers; nothing is sent back.
  private func onLeave(_ message: Message) {
    let d = JSONLine.decode(message.payload) ?? [:]
    let reason: String
    switch (d["reason"] as? String).flatMap(LeaveReason.init(rawValue:)) {
    case .behind: reason = "modeld fell behind the large model"
    case .lost: reason = "the comma lost the link"
    case .stopped: reason = "modeld stopped"
    case .provisioned: reason = "the provisioning run finished"
    case nil: reason = d["reason"] as? String ?? "no reason given"
    }
    var line = "\(who) is leaving: \(reason)"
    func number(_ key: String) -> Double? { (d[key] as? NSNumber)?.doubleValue }
    if let frames = number("frames") {
      line += "; the comma measured \(Int(frames)) frames"
      if let span = number("span_s") { line += " in \(span) s" }
      if let p50 = number("p50_ms"), let p99 = number("p99_ms"), let max = number("max_ms") {
        line += ": whole frame p50 \(p50) p99 \(p99) max \(max) ms"
      }
      if let server = number("server_ms") { line += ", server \(server) ms" }
      if let over = number("over") { line += ", \(Int(over)) over 50 ms" }
      // a frame whose reply was late published the one before it again; a
      // shadowed frame went out with the small model driving, never waited for
      if let held = number("held"), held > 0 { line += ", \(Int(held)) held" }
      if let shadowed = number("shadowed"), shadowed > 0 { line += "; \(Int(shadowed)) more ran with the small model driving" }
    }
    if let drops = number("drops"), let lags = number("lags") {
      line += "; handed back for a lost link \(Int(drops)) and for lag \(Int(lags)) times this drive"
    }
    log.warning(line)
    if let summary { log.info("this session served \(summary)") }
  }

  private func onState(_ message: Message) throws {
    let (sha, skip) = wanted()
    let status = host.status(sha, frameSkip: skip)
    var response = host.telemetry.read()
    response["engine_state"] = status["state"] ?? "none"
    response["detail"] = status["detail"] ?? ""
    response["loaded"] = host.loadedSHA() ?? NSNull()
    response["frames_served"] = frames
    try sendJSON(.stateResp, seq: message.seq, response)
  }
}

/// The frames one session served: a count, the slow ones, and a 1 ms
/// histogram of the server's total, so p50 and p99 cost a bucket increment a
/// frame and nothing on the hot path allocates.
/// Why a comma stops using the link, as its LEAVE says it: the Python's
/// protocol.LEAVE_REASONS, pinned as Pinned.leaveReasons.
enum LeaveReason: String, CaseIterable {
  case behind, lost, stopped, provisioned
}

struct SessionTrips {
  private var buckets = [UInt32](repeating: 0, count: 1001)
  private var count = 0
  private var slow = 0
  private var maxUs: UInt32 = 0
  private let started = ProcessInfo.processInfo.systemUptime

  mutating func record(totalUs: UInt32) {
    count += 1
    if totalUs > FrameStats.slowUs { slow += 1 }
    if totalUs > maxUs { maxUs = totalUs }
    buckets[min(1000, Int(totalUs / 1000))] += 1
  }

  /// The value at `fraction` of the sorted frames, as FrameStats takes p99.
  private func percentile(_ fraction: Double) -> Int {
    let wanted = Int(fraction * Double(count - 1)) + 1
    var seen = 0
    for (ms, n) in buckets.enumerated() {
      seen += Int(n)
      if seen >= wanted { return ms }
    }
    return 1000
  }

  var summary: String? {
    guard count > 0 else { return nil }
    let held = Int((ProcessInfo.processInfo.systemUptime - started).rounded())
    return "\(count) frames in \(held) s, total p50 \(percentile(0.5)) p99 \(percentile(0.99)) max \(maxUs / 1000) ms, \(slow) slow"
  }
}

@inline(__always)
func microseconds(since start: UInt64) -> UInt32 {
  UInt32(min(UInt64(UInt32.max), (DispatchTime.now().uptimeNanoseconds - start) / 1000))
}
