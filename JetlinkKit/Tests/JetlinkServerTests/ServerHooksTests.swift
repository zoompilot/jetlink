import Foundation
import JetlinkKit
import JetlinkTestSupport
import Testing

@testable import JetlinkServer

/// A CUDA error as the TensorRT backend will throw it: sticky or not.
struct DeviceError: FatalEngineError, CustomStringConvertible {
  let sticky: Bool
  var isFatal: Bool { sticky }
  var description: String { sticky ? "CUDA_ERROR_ILLEGAL_ADDRESS" : "CUDA_ERROR_OUT_OF_MEMORY" }
}

/// A link whose frame replies cannot be written: the comma already hung up.
final class HungUpLink: MessageLink, @unchecked Sendable {
  var peer: String { "hung-up" }
  var medium: LinkMedium? { .tcp }
  var connectsOnOpen: Bool { true }
  func recv() throws -> Message { throw LinkError.closed("nothing to read") }
  func sendParts(_ type: Wire.Msg, seq: UInt32, parts: UnsafeBufferPointer<UnsafeRawBufferPointer>, flags: Wire.Flag) throws {
    if type == .inferResp { throw LinkError.closed("peer went away during send") }
  }
  func shutdown() {}
  func close() {}
}

/// A gadget that is off the bus until `plug` and back off after the comma
/// unplugs, recording when the USB loop looked.
final class ComingAndGoingGadget: GadgetSource, @unchecked Sendable {
  private let lock = NSLock()
  private var pipes: FakeUsbfs?
  private var looked: [TimeInterval] = []

  func plug(_ pipes: FakeUsbfs) {
    lock.withLock { self.pipes = pipes }
  }

  func unplug() {
    lock.withLock {
      pipes?.unplug()
      pipes = nil
    }
  }

  var looks: [TimeInterval] { lock.withLock { looked } }

  func present() -> Bool {
    lock.withLock {
      looked.append(ProcessInfo.processInfo.systemUptime)
      return pipes != nil
    }
  }

  func open() throws -> any MessageLink {
    guard let pipes = lock.withLock({ pipes }) else { throw LinkError.closed("no gadget") }
    return USBTransport(pipes: UsbfsPipes(device: UsbfsDevice(kernel: pipes), inEndpoint: 0x81, outEndpoint: 0x01), medium: .usb3)
  }
}

/// Each of `ServerHooks`, as the Linux daemon will use it, and the defaults
/// the apps keep.
@Suite("Server hooks", .serialized)
struct ServerHooksTests {
  /// One frame of `golden`'s model, zeros throughout.
  func frame(_ client: TestClient, _ golden: Golden, flags: Wire.Flag = []) throws -> Reply {
    let spec = try ModelSpec.from(golden.spec)
    var request = Data(count: Wire.inferReqSize)
    request.withUnsafeMutableBytes {
      $0.storeBytes(of: UInt32(1).littleEndian, as: UInt32.self)
      $0.storeBytes(of: flags.rawValue.littleEndian, toByteOffset: 4, as: UInt32.self)
    }
    request.append(Data(count: spec.warpedBytes + spec.packedBytes))
    try client.send(.inferReq, request)
    return try client.recv(.inferResp)
  }

  @Test("The hello carries Python's keys, the host's sleep_after and what the backend adds")
  func hello() throws {
    let python: Set = [
      "protocol", "backend", "runtime_version", "device", "engine_state", "loaded", "frames_served", "cached_models", "telemetry", "sleep_after",
      "pick", "fetches",
    ]
    try serve(hooks: ServerHooks()) { _, client in
      let hello = try client.hello()
      #expect(Set(hello.keys) == python)
      #expect(hello["sleep_after"] as? Double == 0)
      // nothing picked yet, and no control layer to download for the comma
      #expect(hello["pick"] is NSNull)
      #expect(hello["fetches"] as? Bool == false)
    }
    let trt = FlakyBackend(describing: ["trt_version": "10.3.0"])
    try serve(hooks: ServerHooks(sleepAfter: 900), backend: trt) { _, client in
      let hello = try client.hello()
      #expect(Set(hello.keys) == python.union(["trt_version"]))
      #expect(hello["trt_version"] as? String == "10.3.0")
      #expect(hello["sleep_after"] as? Double == 900)
    }
  }

  @Test("Telemetry goes out in the hello, the state reply, and a frame that asks for it")
  func telemetry() throws {
    let golden = try Golden("tiny_stateful")
    let hooks = ServerHooks(telemetry: { ["temp_c": 51.5, "power_w": 12.25] })
    func readings(_ object: Any?) -> [String: Double]? {
      (object as? [String: Any])?.compactMapValues { ($0 as? NSNumber)?.doubleValue }
    }
    try serve(hooks: hooks) { server, client in
      // The server samples the host's sensors on a thread of its own, when
      // asked, and a sample more than a second old is none: each check waits
      // for a fresh one, since a slow emulator can take longer between asks.
      func fresh() -> Bool { eventually(timeout: 10) { !server.host.telemetry.read().isEmpty } }
      #expect(fresh())
      #expect(readings(try client.hello()["telemetry"]) == ["temp_c": 51.5, "power_w": 12.25])
      try client.send(.stateReq)
      let state = try client.recv(.stateResp).json
      #expect(state["temp_c"] as? Double == 51.5 && state["engine_state"] as? String == "none")
      _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
      let plain = try frame(client, golden)
      let spec = try ModelSpec.from(golden.spec)
      #expect(plain.payload.count == spec.inferRespBytes)
      #expect(fresh())
      let asked = try frame(client, golden, flags: .wantState)
      #expect(asked.status == Wire.Status.ok.rawValue)
      try #require(asked.payload.count > spec.inferRespBytes)
      let piggyback = asked.payload[spec.inferRespBytes...]
      #expect(readings(try JSONSerialization.jsonObject(with: piggyback)) == ["temp_c": 51.5, "power_w": 12.25])
    }
  }

  @Test("A benchmark reports the host's thermal state")
  func thermal() throws {
    let golden = try Golden("tiny_stateful")
    try serve(hooks: ServerHooks(thermal: { "serious" })) { server, client in
      _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
      client.close()
      #expect(eventually(timeout: 10) { server.host.lock.withLock { server.host.session == nil } })
      let report = try server.host.benchmark(seconds: 1, run: BenchmarkRun())
      #expect(report.thermalAtStart == "serious" && report.thermalAtEnd == "serious")
      #expect(report.windows.allSatisfy { $0.thermal == "serious" })
    }
  }

  @Test("A probe's frames are the same every run, from a reset, and it waits for the comma to go")
  func probe() throws {
    let golden = try Golden("tiny_stateful")
    try serve(hooks: ServerHooks()) { server, client in
      _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
      #expect(throws: (any Error).self) { try server.host.probe() }
      client.close()
      #expect(eventually(timeout: 10) { server.host.lock.withLock { server.host.session == nil } })
      let first = try server.host.probe(frames: 6)
      let again = try server.host.probe(frames: 6)
      #expect(first.outputs.count == 6)
      #expect(first.outputs == again.outputs)
      // frames differ, and the history carries: the same frame gives another answer later
      #expect(first.outputs[0] != first.outputs[1])
      let spec = try ModelSpec.from(golden.spec)
      #expect(first.outputs.allSatisfy { $0.count == spec.outputCount })
      #expect(Set(first.slices.map(\.name)) == Set(spec.outputSlices.map(\.name)))
      let same = OutputParity.compare(candidate: again.outputs, reference: first.outputs, slices: first.slices)
      #expect(same.passed, "\(same.text)")
      #expect(!server.host.lock.withLock { server.host.benchmarking })
    }
  }

  /// The sleeper's input: touched while the gadget is on the bus and at each
  /// connection's start and end, idle only while it is gone, and a look for
  /// the gadget at once after a sleep.
  @Test("The gadget's comings and goings reach the host, and a wake looks for the comma at once")
  func gadgetIdle() throws {
    let gadget = ComingAndGoingGadget()
    let events = Recorded<GadgetIdleEvent>()
    let slept = Recorded<TimeInterval>()
    let hooks = ServerHooks(gadgetIdle: { event in
      events.append(event)
      guard event == .absent, slept.all.isEmpty else { return false }
      slept.append(ProcessInfo.processInfo.systemUptime)
      return true
    })
    let cache = try TemporaryDirectory()
    let server = try Server(
      configuration: Server.Configuration(host: "127.0.0.1", port: 0, cacheRoot: cache.url, preload: false, listen: false, usb: true),
      backend: cpuBackend(), gadget: gadget, hooks: hooks)
    try server.start()
    defer { server.shutdown() }
    #expect(events.wait { $0.filter { $0 == .absent }.count >= 3 })
    // The first idle slept: the next look came at once, not a poll later.
    let woke = try #require(slept.all.first)
    let next = try #require(gadget.looks.first { $0 > woke })
    #expect(next - woke < Server.usbPoll / 2, "looked again \(next - woke) s after waking")

    let comma = FakeUsbfs()
    gadget.plug(comma)
    let client = GadgetClient(comma)
    _ = try client.hello(name: "modeld")
    gadget.unplug()
    #expect(events.wait { $0.last == .absent })
    let seen = events.all.drop { $0 == .absent }
    #expect(Array(seen.prefix(3)) == [.present, .connected, .disconnected], "\(seen)")
    #expect(seen.dropFirst(3).allSatisfy { $0 == .absent }, "\(seen)")
  }

  @Test("No idle while a comma is served over TCP, though the gadget is off the bus")
  func noIdleWhileServed() throws {
    let gadget = ComingAndGoingGadget()
    let events = Recorded<GadgetIdleEvent>()
    let hooks = ServerHooks(gadgetIdle: { event in
      events.append(event)
      return false
    })
    try serve(hooks: hooks, gadget: gadget) { _, client in
      _ = try client.hello()
      try client.send(.ping)
      _ = try client.recv(.pong)
      #expect(events.wait { $0.contains(.connected) })
      let connected = events.all.count
      Thread.sleep(forTimeInterval: Server.usbPoll * 3)
      #expect(events.all.count == connected, "\(events.all)")
      client.close()
      #expect(events.wait { $0.last == .absent })
      #expect(events.all[connected...].first == .disconnected)
    }
  }

  @Test("Without a shutdown hook, or when it declines, the reply is ok:false and the app hears of it")
  func shutdownRefused() throws {
    for hooks in [ServerHooks(), ServerHooks(shutdown: { _ in nil })] {
      try serve(hooks: hooks) { server, client in
        let heard = Recorded<String>()
        server.host.subscribe { if case .shutdownRequested(let reason) = $0 { heard.append(reason) } }
        try client.sendJSON(.shutdownReq, ["reason": "car battery"])
        let reply = try client.recv(.shutdownResp).json
        #expect(reply["ok"] as? Bool == false)
        #expect(heard.wait { $0 == ["car battery"] })
      }
    }
  }

  @Test("A shutdown hook that accepts gets its reply ok:true written before its action runs")
  func shutdownAccepted() throws {
    let comma = FakeUsbfs()
    let reasons = Recorded<String>()
    let repliedFirst = Recorded<Bool>()
    let hooks = ServerHooks(shutdown: { reason in
      reasons.append(reason)
      return {
        let frames = (try? HostFrames.parse(comma.written)) ?? []
        repliedFirst.append(frames.contains { $0.type == Wire.Msg.shutdownResp.rawValue })
      }
    })
    let cache = try TemporaryDirectory()
    let server = try Server(
      configuration: Server.Configuration(host: "127.0.0.1", port: 0, cacheRoot: cache.url, preload: false, listen: false, usb: true),
      backend: cpuBackend(), gadget: FakeGadget([comma]), hooks: hooks)
    try server.start()
    defer { server.shutdown() }
    let client = GadgetClient(comma)
    try client.sendJSON(.shutdownReq, ["reason": "car battery"])
    let reply = try client.recv(.shutdownResp).json
    #expect(reply["ok"] as? Bool == true)
    #expect(reply["detail"] as? String == "powering off")
    #expect(repliedFirst.wait { $0 == [true] })
    #expect(reasons.all == ["car battery"])
  }

  @Test("A fatal engine error is answered INFER_FAILED, then handed to the host; another is only answered")
  func fatal() throws {
    let golden = try Golden("tiny_stateful")
    let backend = FlakyBackend()
    let fatal = Recorded<String>()
    try serve(hooks: ServerHooks(fatal: { fatal.append(String(describing: $0)) }), backend: backend) { _, client in
      _ = try client.ensureEngine(model: golden.model, sha256: golden.sha256)
      #expect(try frame(client, golden).status == Wire.Status.ok.rawValue)
      backend.failRuns(with: DeviceError(sticky: false))
      #expect(try frame(client, golden).status == Wire.Status.inferFailed.rawValue)
      backend.failRuns(with: HostError.failed("not an engine fault"))
      #expect(try frame(client, golden).status == Wire.Status.inferFailed.rawValue)
      #expect(fatal.all.isEmpty)
      backend.failRuns(with: DeviceError(sticky: true))
      #expect(try frame(client, golden).status == Wire.Status.inferFailed.rawValue)
      #expect(fatal.wait { $0 == ["CUDA_ERROR_ILLEGAL_ADDRESS"] })
    }
  }

  @Test("A fatal engine error while preparing the engine fails the job, then is handed to the host; another only fails it")
  func fatalWhilePreparing() throws {
    let golden = try Golden("tiny_stateful")
    let backend = FlakyBackend()
    let fatal = Recorded<String>()
    try serve(hooks: ServerHooks(fatal: { fatal.append(String(describing: $0)) }), backend: backend) { _, client in
      backend.failLoads(with: DeviceError(sticky: false))
      #expect(throws: TestError.self) { try client.ensureEngine(model: golden.model, sha256: golden.sha256) }
      #expect(fatal.all.isEmpty)
      // Built by the first try: this one only loads.
      backend.failLoads(with: DeviceError(sticky: true))
      #expect(throws: TestError.self) { try client.ensureEngine(model: golden.model, sha256: golden.sha256) }
      #expect(fatal.wait { $0 == ["CUDA_ERROR_ILLEGAL_ADDRESS"] })
    }
  }

  @Test("A fatal engine error is handed to the host even when its INFER_FAILED cannot be written")
  func fatalWhenTheReplyFails() throws {
    let golden = try Golden("tiny_stateful")
    let backend = FlakyBackend()
    let fatal = Recorded<String>()
    let tmp = try TemporaryDirectory()
    let cache = try ServerCache(root: tmp.url, backend: backend)
    let host = EngineHost(cache: cache, hooks: ServerHooks(fatal: { fatal.append(String(describing: $0)) }))
    defer { host.close() }
    let model = try Data(contentsOf: golden.model)
    try model.write(to: cache.modelPath(golden.sha256))
    let session = Session(transport: HungUpLink(), host: host)
    func handle(_ type: Wire.Msg, seq: UInt32, _ payload: Data) throws {
      try payload.withUnsafeBytes { try session.handle(Message(msgType: type.rawValue, seq: seq, flags: 0, payload: $0)) }
    }
    let ask: [String: Any] = ["sha256": golden.sha256, "nbytes": model.count, "frame_skip": 4]
    try handle(.engineReq, seq: 1, JSONSerialization.data(withJSONObject: ask))
    #expect(host.settles() && host.snapshot().state == .ready)
    let spec = try ModelSpec.from(golden.spec)
    backend.failRuns(with: DeviceError(sticky: true))
    #expect(throws: LinkError.self) {
      try handle(.inferReq, seq: 2, Data(count: Wire.inferReqSize + spec.warpedBytes + spec.packedBytes))
    }
    #expect(fatal.all == ["CUDA_ERROR_ILLEGAL_ADDRESS"])
  }
}
