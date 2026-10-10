import Foundation
import JetlinkKit
import JetlinkLog
import JetlinkORT
import JetlinkTestSupport
import Testing

@testable import JetlinkServer

/// The whole server, over a real socket, on onnxruntime's CPU provider: the
/// upload, the Swift preparation, the build, the load, the queues or the state
/// loop, and the reply, checked against what the Python server's parts
/// compute for the same frames (bit for bit on Apple; see `CommaClient.replay`).
@Suite("Server", .serialized)
struct ServerTests {
  @Test("A comma is served the outputs the Python server computes", arguments: ["tiny_queued", "tiny_stateful"])
  func servesGoldenFrames(_ name: String) throws {
    let golden = try Golden(name)
    try serve { server, client in
      let (hello, count) = try client.replay(golden)
      #expect(hello["protocol"] as? Int == Int(Wire.version))
      #expect(hello["backend"] as? String == "ort")
      #expect((hello["device"] as? String)?.hasPrefix("cpu-") == true)
      #expect(count == 8)
      #expect(eventually { server.framesServed == count })

      // onnxruntime's artifact: a manifest naming the session and its unit, and a sidecar.
      let engines = server.cache.layout.engines
      let artifacts = try FileManager.default.contentsOfDirectory(atPath: engines.path).filter { $0.hasSuffix(".ortcache") }
      #expect(artifacts.count == 1)
      let artifact = engines.appending(path: artifacts[0])
      let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: artifact.appending(path: "sessions.json"))) as? [[String: Any]]
      #expect(manifest?.map { $0["unit"] as? String } == ["cpu"])
      let meta = Artifact.sidecar(artifact)
      #expect(meta["prepare"] as? Int == OrtProfile.cpu.prepareVersion)
      #expect(meta["preparer"] as? String == "swift")
      #expect(((meta["artifact_bytes"] as? NSNumber)?.int64Value ?? 0) > 0)
    }
  }

  @Test("The link names its medium: TCP until the comma's hello says it is a USB cable")
  func linkMedium() throws {
    let links = Recorded<LinkEvent>()
    try serve { server, client in
      server.host.subscribe { if case .link(let link) = $0 { links.append(link) } }
      try client.send(.stateReq)
      _ = try client.recv(.stateResp)
      #expect(server.currentLink.linkMedium == .tcp)
      _ = try client.hello(name: "modeld", link: ["kind": "cable", "usb_speed": "high-speed"])
      #expect(server.currentLink.linkMedium == .usb2)
    }
    let media = links.all.filter { $0.state == .connected }.compactMap(\.linkMedium)
    #expect(media.last == .usb2)
    #expect(media.dropLast().allSatisfy { $0 == .tcp })
    // one "client connected" line for the session, and the hello's word on its own
    let peer = try #require(links.all.first { $0.state == .connected }?.peer)
    let lines = LogRing.shared.lines()
    #expect(lines.filter { $0.contains("jetlink.server: client connected from \(peer) over TCP") }.count == 1)
    #expect(!lines.contains { $0.contains("client connected from \(peer) over USB") })
    #expect(lines.contains { $0.contains("jetlink.server: the comma's hello says its link is USB 2") })
  }

  @Test("The link names who said hello: nobody on a held call, then each hello's name")
  func linkClient() throws {
    try serve { server, client in
      try client.send(.stateReq)
      _ = try client.recv(.stateResp)
      #expect(server.currentLink.state == .connected)
      #expect(server.currentLink.client == nil)
      _ = try client.hello(name: "provision")
      #expect(server.currentLink.client == "provision")
      #expect(!server.currentLink.isDrive)
      // the same medium, a new name: announced again
      _ = try client.hello(name: "modeld")
      #expect(server.currentLink.isDrive)
    }
  }

  @Test("Progress is throttled within a stage, never across one")
  func progressStages() throws {
    try serve { server, _ in
      let seen = Recorded<String>()
      server.host.subscribe { if case .progress(let stage, _, let msg) = $0 { seen.append("\(stage) \(msg)") } }
      server.host.progress("patch", 0, "preparing")
      server.host.progress("patch", 0.5, "halfway")
      server.host.progress("compile", 0, "compiling")
      server.host.progress("compile", 0.1, "still compiling")
      server.host.progress("compile", 1, "compiled")
      #expect(seen.all == ["patch preparing", "compile compiling", "compile compiled"])
    }
  }

  @Test("A replayed request is dropped, and pings are answered")
  func dropsReplays() throws {
    try serve { _, client in
      _ = try client.hello()
      let seq = try client.send(.ping)
      #expect(try client.recv().type == Wire.Msg.pong.rawValue)
      try client.send(.ping, seq: seq)  // a replay: no answer
      let next = try client.send(.ping)
      let reply = try client.recv()
      #expect(reply.type == Wire.Msg.pong.rawValue)
      #expect(reply.seq == next)
    }
  }

  @Test("A ping before a hello is an error, and answered once the client says hello")
  func pingNeedsHello() throws {
    try serve { _, client in
      let early = try client.send(.ping)
      let refused = try client.recv()
      #expect(refused.type == Wire.Msg.error.rawValue && refused.seq == early)
      #expect(refused.json["error"] as? String == "no_hello")
      _ = try client.hello()
      let next = try client.send(.ping)
      let reply = try client.recv()
      #expect(reply.type == Wire.Msg.pong.rawValue && reply.seq == next)
    }
  }

  @Test("A frame with no engine is answered NOT_READY, not dropped")
  func notReady() throws {
    try serve { _, client in
      try client.send(.inferReq, Data(count: 64))
      let reply = try client.recv(.inferResp)
      #expect(reply.status == Wire.Status.notReady.rawValue)
    }
  }

  @Test("A model that does not hash to its name is refused")
  func refusesACorruptUpload() throws {
    let golden = try Golden("tiny_queued")
    try serve { server, client in
      let bytes = try Data(contentsOf: golden.model)
      try client.sendJSON(.engineReq, ["sha256": golden.sha256, "nbytes": bytes.count, "frame_skip": 4])
      #expect(try client.recv(.engineResp).json["state"] as? String == "need_upload")
      var payload = withUnsafeBytes(of: UInt64(0).littleEndian) { Data($0) }
      payload.append(Data(repeating: 7, count: bytes.count))
      try client.send(.uploadChunk, payload)
      try client.sendJSON(.uploadDone, ["sha256": golden.sha256])
      let reply = try client.recv(.engineResp).json
      #expect(reply["state"] as? String == "failed")
      #expect(try !FileManager.default.fileExists(atPath: server.cache.modelPath(golden.sha256).path))
    }
  }

  @Test("A new connection takes over from the one being served")
  func newConnectionTakesOver() throws {
    try serve { server, first in
      _ = try first.hello()
      try first.send(.ping)
      #expect(try first.recv().type == Wire.Msg.pong.rawValue)
      let second = try TestClient(port: server.port!)
      defer { second.close() }
      _ = try second.hello()
      try second.send(.ping)
      #expect(try second.recv().type == Wire.Msg.pong.rawValue)
      #expect(throws: LinkError.self) { try first.recv() }
    }
  }

  @Test("A benchmark runs the loaded engine at the comma's pace and reports in windows")
  func benchmarks() throws {
    let golden = try Golden("tiny_queued")
    try serve { server, client in
      _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
      // Refused with a comma on the line.
      #expect(throws: HostError.self) { try server.host.benchmark(seconds: 1, run: BenchmarkRun()) }
      client.close()
      let deadline = Date().addingTimeInterval(5)
      while server.host.lock.withLock({ server.host.session != nil }) && Date() < deadline {
        Thread.sleep(forTimeInterval: 0.02)
      }
      let events = Recorded<BenchmarkEvent>()
      server.host.subscribe { event in
        if case .benchmark(let value) = event { events.append(value) }
      }
      let report = try server.host.benchmark(seconds: 2, run: BenchmarkRun())
      #expect(report.sha256 == golden.sha256)
      #expect(report.device == server.backend.deviceTag())
      #expect(report.frames >= 30 && report.frames <= 45, "\(report.frames) frames in 2 s at 20 Hz")
      #expect(report.frame.mean > 0 && report.frame.p99 >= report.frame.p50 && report.frame.max >= report.frame.p99)
      #expect(report.windows.count == 1)
      #expect(report.windows[0].startSecond == 0)
      #expect(report.over50 <= report.over35)
      #expect(!report.cancelled)
      #expect(report.build.contains("CPU keep-warm off"))
      #expect(report.thermalAtStart != "")
      #expect(report.text.contains("frame        mean"))
      let seen = events.all
      #expect(seen.first?.state == "running")
      #expect(seen.last?.state == "done")
      #expect(seen.last?.report == report)
      #expect(!server.host.lock.withLock { server.host.benchmarking })
    }
  }

  @Test("A benchmark can be cancelled")
  func cancelsABenchmark() throws {
    let golden = try Golden("tiny_stateful")
    try serve { server, client in
      _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
      client.close()
      let deadline = Date().addingTimeInterval(5)
      while server.host.lock.withLock({ server.host.session != nil }) && Date() < deadline {
        Thread.sleep(forTimeInterval: 0.02)
      }
      let run = BenchmarkRun()
      Thread {
        Thread.sleep(forTimeInterval: 0.6)
        run.cancel()
      }.start()
      let report = try server.host.benchmark(seconds: 60, run: run)
      #expect(report.cancelled)
      #expect(report.seconds < 5)
    }
  }

  @Test(
    "Between sessions an engine that cools runs on zeros, and the next comma still gets the Python outputs",
    arguments: ["tiny_queued", "tiny_stateful"])
  func keptWarmBetweenSessions(_ name: String) throws {
    let golden = try Golden(name)
    try serve(backend: FlakyBackend(coolsWhenIdle: true)) { server, client in
      _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
      let host = server.host
      let seen = ProcessInfo.processInfo.systemUptime
      #expect(!host.warmIfIdle(now: seen + 1), "a comma seen a second ago may be about to send a frame")
      #expect(host.warmIfIdle(now: seen + EngineHost.warmIdleAfter + 1))
      #expect(!host.warmIfIdle(now: seen + EngineHost.warmFor + 5), "long after the comma was last seen")
      // the warm run left nothing in the history the comma's frames meet
      #expect(try client.replay(golden).frames == 8)
      #expect(host.warmIfIdle(now: ProcessInfo.processInfo.systemUptime + EngineHost.warmIdleAfter + 1))
      #expect(try client.replay(golden).frames == 8)
    }
  }

  @Test("An engine that keeps warm by itself is never run between sessions")
  func onlyCoolingEnginesAreWarmed() throws {
    let golden = try Golden("tiny_stateful")
    try serve { server, client in
      _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
      #expect(!server.host.warmIfIdle(now: ProcessInfo.processInfo.systemUptime + EngineHost.warmIdleAfter + 1))
    }
  }

  @Test("A warm run that fails stops the keeping warm until the next load")
  func failedWarmStops() throws {
    let golden = try Golden("tiny_stateful")
    let backend = FlakyBackend(coolsWhenIdle: true)
    try serve(backend: backend) { server, client in
      _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
      let idle = ProcessInfo.processInfo.systemUptime + EngineHost.warmIdleAfter + 1
      backend.failRuns(with: HostError.failed("test"))
      #expect(!server.host.warmIfIdle(now: idle))
      backend.failRuns(with: nil)
      #expect(!server.host.warmIfIdle(now: idle + 1))
    }
  }

  @Test("The engine outlives the connection, and the next comma gets it loaded")
  func engineOutlivesConnection() throws {
    let golden = try Golden("tiny_stateful")
    try serve { server, client in
      _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
      client.close()
      let again = try TestClient(port: server.port!)
      defer { again.close() }
      let bytes = try Data(contentsOf: golden.model)
      try again.sendJSON(.engineReq, ["sha256": golden.sha256, "nbytes": bytes.count, "frame_skip": 4])
      #expect(try again.recv(.engineResp).json["state"] as? String == "ready")
    }
  }
}

/// The listener beside the sessions: it heals, and stopping keeps the engine.
@Suite("Server lifecycle", .serialized)
struct ServerLifecycleTests {
  func makeServer(_ cache: TemporaryDirectory, dial: DialTarget? = nil) throws -> Server {
    try Server(
      configuration: Server.Configuration(host: "127.0.0.1", port: 0, cacheRoot: cache.url, preload: false, dial: dial), backend: cpuBackend())
  }

  @Test("stop keeps the engine loaded; shutdown releases it")
  func stopKeepsTheEngine() throws {
    let golden = try Golden("tiny_queued")
    let cache = try TemporaryDirectory()
    let server = try makeServer(cache)
    try server.start()
    let client = try TestClient(port: server.port!)
    _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
    client.close()
    server.stop()
    #expect(server.host.loadedSHA() == golden.sha256)
    #expect(server.port == nil)
    server.shutdown()
    #expect(server.host.loadedSHA() == nil)
  }

  @Test("The log sink hears what the server logs")
  func logSink() throws {
    let lines = Recorded<String>()
    Log.sink = { level, category, message in lines.append("\(level.rawValue) \(category): \(message)") }
    defer { Log.sink = nil }
    let cache = try TemporaryDirectory()
    let server = try makeServer(cache)
    try server.start()
    server.shutdown()
    #expect(lines.all.contains { $0.hasPrefix("info server: listening on 127.0.0.1:") })
  }

  @Test("A listener that goes away is opened again")
  func listenerHeals() throws {
    let cache = try TemporaryDirectory()
    let server = try makeServer(cache)
    try server.start()
    let first = server.port!
    // The accept loop's listener ends as iOS ends it: closed under the loop.
    server.listener?.close()
    let deadline = Date().addingTimeInterval(5)
    while server.isListening && Date() < deadline {
      Thread.sleep(forTimeInterval: 0.05)
    }
    #expect(!server.isListening)
    while !server.isListening && Date() < deadline {
      Thread.sleep(forTimeInterval: 0.05)
    }
    guard let port = server.port else { throw TestError("the server did not listen again after \(first) went away") }
    let client = try TestClient(port: port)
    defer { client.close(); server.shutdown() }
    _ = try client.hello()
    try client.send(.ping)
    #expect(try client.recv().type == Wire.Msg.pong.rawValue)
  }

  @Test("A dialed connection is served like an accepted one, and dialed again after it ends")
  func dials() throws {
    let cache = try TemporaryDirectory()
    // The comma's end: a listener the server dials.
    let comma = try TCPListener(host: "127.0.0.1", port: 0)
    let server = try makeServer(cache, dial: DialTarget(host: "127.0.0.1", port: comma.port))
    try server.start()
    defer { server.shutdown(); comma.close() }
    for round in 0..<2 {
      guard let transport = comma.accept() else { throw TestError("no dial in round \(round)") }
      transport.setReceiveTimeout(10)
      try transport.sendJSON(.helloReq, seq: 1, ["client": ["name": "comma"]])
      #expect(try transport.recv().msgType == Wire.Msg.helloResp.rawValue)
      try transport.send(.ping, seq: UInt32(round + 2))
      let reply = try transport.recv()
      #expect(reply.msgType == Wire.Msg.pong.rawValue)
      #expect(reply.seq == UInt32(round + 2))
      transport.close()
    }
    // A listener keeps accepting beside the dialing.
    let client = try TestClient(port: server.port!)
    defer { client.close() }
    _ = try client.hello()
    try client.send(.ping)
    #expect(try client.recv().type == Wire.Msg.pong.rawValue)
    server.setDial(nil)
    #expect(server.dialTarget == nil)
  }

  @Test("A dial target that does not answer is retried until it does")
  func dialsUntilAnswered() throws {
    let cache = try TemporaryDirectory()
    // A port with nobody on it, until the listener opens below.
    let probe = try TCPListener(host: "127.0.0.1", port: 0)
    let port = probe.port
    probe.close()
    let server = try makeServer(cache)
    try server.start()
    defer { server.shutdown() }
    server.setDial(DialTarget(host: "127.0.0.1", port: port))
    Thread.sleep(forTimeInterval: 0.7)
    let comma = try TCPListener(host: "127.0.0.1", port: port)
    defer { comma.close() }
    guard let transport = comma.accept() else { throw TestError("never dialed") }
    transport.setReceiveTimeout(10)
    try transport.sendJSON(.helloReq, seq: 1, ["client": ["name": "comma"]])
    #expect(try transport.recv().msgType == Wire.Msg.helloResp.rawValue)
    try transport.send(.ping, seq: 2)
    #expect(try transport.recv().msgType == Wire.Msg.pong.rawValue)
    transport.close()
  }
  /// Two loops taking over at once, as the accept, dial and USB loops can:
  /// the first link's interrupt waits for a second one, so two takeovers
  /// that both read it as the session being served are both inside the swap
  /// together, and each would start a session of its own.
  @Test("Takeovers from two loops at once serve one session at a time")
  func takeoversAreSerialized() throws {
    let cache = try TemporaryDirectory()
    let server = try Server(
      configuration: Server.Configuration(host: "127.0.0.1", port: 0, cacheRoot: cache.url, preload: false, listen: false),
      backend: cpuBackend())
    try server.start()
    defer { server.shutdown() }
    let live = LiveCount()
    let interrupts = Recorded<String>()
    server.takeover(ParkedLink("a", live: live, interrupts: interrupts, holdFirstInterrupt: 1.0))
    #expect(eventually { live.current == 1 })
    let group = DispatchGroup()
    for name in ["b", "c"] {
      DispatchQueue.global().async(group: group) {
        server.takeover(ParkedLink(name, live: live, interrupts: interrupts))
      }
    }
    group.wait()
    #expect(eventually { live.current == 1 })
    Thread.sleep(forTimeInterval: 0.2)
    #expect(live.most == 1, "\(live.most) sessions served at once")
    #expect(interrupts.all.filter { $0 == "a" }.count == 1, "\(interrupts.all)")
  }

  /// Over USB the gadget's interface opens whether or not anything on the
  /// comma serves it, so a claim that hears nothing must not interrupt the
  /// session being served: a loan handover's bounce does exactly that, with
  /// nobody on the comma behind the re-appeared gadget.
  @Test("An unserved gadget claim leaves the session being served alone")
  func unservedGadgetDoesNotTakeOver() throws {
    let cache = try TemporaryDirectory()
    let server = try Server(
      configuration: Server.Configuration(host: "127.0.0.1", port: 0, cacheRoot: cache.url, preload: false, listen: false),
      backend: cpuBackend())
    try server.start()
    defer { server.shutdown() }
    let live = LiveCount()
    let interrupts = Recorded<String>()
    server.takeover(ParkedLink("tcp", live: live, interrupts: interrupts))
    #expect(eventually { live.current == 1 })
    let (done, session) = server.takeoverWhenAnnounced(UnservedGadget())
    done.wait()
    #expect(!session.announced)
    #expect(live.current == 1, "the session being served was interrupted")
    #expect(interrupts.all.isEmpty, "\(interrupts.all)")
  }

  /// The other half of the same rule: a gadget the comma does serve speaks,
  /// and the session it announces takes over from the one being served, as
  /// the comma's own reconnect must.
  @Test("A gadget that speaks takes over from the session being served")
  func speakingGadgetTakesOver() throws {
    let cache = try TemporaryDirectory()
    let server = try Server(
      configuration: Server.Configuration(host: "127.0.0.1", port: 0, cacheRoot: cache.url, preload: false, listen: false),
      backend: cpuBackend())
    try server.start()
    defer { server.shutdown() }
    let live = LiveCount()
    let interrupts = Recorded<String>()
    server.takeover(ParkedLink("tcp", live: live, interrupts: interrupts))
    #expect(eventually { live.current == 1 })
    let (done, session) = server.takeoverWhenAnnounced(SpeakingGadget())
    done.wait()
    #expect(session.announced)
    #expect(interrupts.all == ["tcp"], "\(interrupts.all)")
  }
}

/// How many links are being read at once, and the most there ever were.
final class LiveCount: @unchecked Sendable {
  private let lock = NSLock()
  private var now = 0
  private var peak = 0

  var current: Int { lock.withLock { now } }
  var most: Int { lock.withLock { peak } }

  func enter() {
    lock.withLock {
      now += 1
      peak = max(peak, now)
    }
  }

  func leave() {
    lock.withLock { now -= 1 }
  }
}

/// A link with a comma that never sends: its session reads until the link
/// is shut down. The first shutdown can wait for a second, up to
/// `holdFirstInterrupt`.
final class ParkedLink: MessageLink, @unchecked Sendable {
  let peer: String
  private let live: LiveCount
  private let interrupts: Recorded<String>
  private let hold: TimeInterval
  private let condition = NSCondition()
  private var shut = false

  init(_ peer: String, live: LiveCount, interrupts: Recorded<String>, holdFirstInterrupt: TimeInterval = 0) {
    self.peer = peer
    self.live = live
    self.interrupts = interrupts
    hold = holdFirstInterrupt
  }

  var medium: LinkMedium? { .tcp }
  var connectsOnOpen: Bool { true }

  func recv() throws -> Message {
    live.enter()
    defer { live.leave() }
    condition.withLock {
      while !shut { condition.wait() }
    }
    throw LinkError.closed("shut down")
  }

  func sendParts(_ type: Wire.Msg, seq: UInt32, parts: UnsafeBufferPointer<UnsafeRawBufferPointer>, flags: Wire.Flag) throws {}

  func shutdown() {
    interrupts.append(peer)
    if hold > 0 {
      _ = interrupts.wait(timeout: hold) { seen in seen.filter { $0 == self.peer }.count >= 2 }
    }
    condition.withLock {
      shut = true
      condition.broadcast()
    }
  }

  func close() {}
}

/// A gadget nobody on the comma serves: its reads fail at once, and no
/// message ever arrives. The shape the USB loop sees between runs and across
/// a loan handover.
final class UnservedGadget: MessageLink, @unchecked Sendable {
  let peer = "usb"
  var medium: LinkMedium? { .usb }
  var connectsOnOpen: Bool { false }

  func recv() throws -> Message {
    throw LinkError.closed("usb bulk read aborted")
  }

  func sendParts(_ type: Wire.Msg, seq: UInt32, parts: UnsafeBufferPointer<UnsafeRawBufferPointer>, flags: Wire.Flag) throws {}

  func shutdown() {}

  func close() {}
}

/// A gadget the comma serves: one message — enough to announce — and then
/// the link is gone.
final class SpeakingGadget: MessageLink, @unchecked Sendable {
  let peer = "usb"
  var medium: LinkMedium? { .usb3 }
  var connectsOnOpen: Bool { false }
  private var said = false

  func recv() throws -> Message {
    defer { said = true }
    if said {
      throw LinkError.closed("link closed")
    }
    return Message(msgType: Wire.Msg.ping.rawValue, seq: 1, flags: 0, payload: UnsafeRawBufferPointer(_empty: ()))
  }

  func sendParts(_ type: Wire.Msg, seq: UInt32, parts: UnsafeBufferPointer<UnsafeRawBufferPointer>, flags: Wire.Flag) throws {}

  func shutdown() {}

  func close() {}
}
