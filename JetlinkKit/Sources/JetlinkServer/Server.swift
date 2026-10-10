import Foundation
import JetlinkKit
import JetlinkRegistry

#if canImport(Darwin)
  import Darwin
#elseif canImport(Android)
  import Android
#endif

/// Where a server dials to serve: the comma's end of a USB network link,
/// which listens for the phone (192.168.60.1:5599 on the composite gadget).
public struct DialTarget: Sendable, Equatable, CustomStringConvertible {
  public let host: String
  public let port: UInt16

  public init(host: String, port: UInt16 = Wire.defaultPort) {
    self.host = host
    self.port = port
  }

  /// "host:port", or "host" for the default port.
  public init?(_ text: String) {
    let parts = text.split(separator: ":", maxSplits: 1).map(String.init)
    guard let host = parts.first, !host.isEmpty else { return nil }
    var port = Wire.defaultPort
    if parts.count == 2 {
      guard let parsed = UInt16(parts[1]) else { return nil }
      port = parsed
    }
    self.init(host: host, port: port)
  }

  public var description: String { "\(host):\(port)" }
}

/// The jetlink server over TCP: a listener, one comma at a time, and the
/// engine host they share. The Swift form of `server/main.py`'s `_serve`.
///
/// One difference from the Python: a new connection takes over from the one
/// being served instead of waiting behind it. A comma only ever has one
/// connection open, so a second one means the first is dead (a pulled cable
/// the keepalive has not noticed yet), and the reconnect must not wait for it.
/// Over USB an open interface is not a connection: the gadget sits on the bus
/// while nothing on the comma serves it, so the gadget's session takes over
/// when the comma first speaks on it, not when the interface opens.
///
/// A connection comes from the listener, from dialing, or from the comma's
/// USB gadget. Over a USB network link the comma listens and the phone dials
/// it, so an accepted connection on the comma is the proof of a phone. A
/// dialed connection is served the same way, and the listener stays open
/// beside it for benches and a Mac on the LAN. The server can be the USB host
/// instead, as the Python server is with `--transport usb`: it opens the
/// gadget's vendor interface whenever the comma is on the bus.
///
/// The host builds the backend and the gadget and passes them in, so nothing
/// here knows which runtime runs the model or how a platform reaches USB.
public final class Server: @unchecked Sendable {
  public struct Configuration: Sendable {
    public var host: String
    public var port: UInt16
    public var cacheRoot: URL
    /// Start loading the engine that was loaded last, before a comma asks.
    public var preload: Bool
    /// Dial this end and serve the connection; `setDial` changes it later.
    public var dial: DialTarget?
    /// Listen on `host:port`. Off for a Mac serving the comma over USB, which
    /// has no reason to open a port.
    public var listen: Bool
    /// Be the USB host: open the comma's gadget whenever it is on the bus.
    /// Needs the gadget the host passes to `init`.
    public var usb: Bool
    /// Built engines kept: one per registry entry on a Mac or a Jetson, where
    /// a rebuild costs minutes and 1 to 2 GB; a phone's disk holds two.
    public var keepPlans: Int

    public init(
      host: String = "0.0.0.0", port: UInt16 = Wire.defaultPort, cacheRoot: URL, preload: Bool = true, dial: DialTarget? = nil,
      listen: Bool = true, usb: Bool = false, keepPlans: Int = CacheLayout.keepArtifacts
    ) {
      self.host = host
      self.port = port
      self.cacheRoot = cacheRoot
      self.preload = preload
      self.dial = dial
      self.listen = listen
      self.usb = usb
      self.keepPlans = keepPlans
    }
  }

  public static let statsInterval: TimeInterval = 1.0
  /// A dial that takes longer has no comma behind it.
  public static let dialTimeout: TimeInterval = 1.0
  public static let dialInterval: TimeInterval = 0.5
  /// Between attempts to listen again after the socket went away.
  static let relistenBackoff: ClosedRange<TimeInterval> = 0.5...5.0
  /// How often to look for the gadget, and to retry one that is on the bus but
  /// not served: the comma's owner holds it between runs, and a run's hello
  /// waits on this end reading. Quick for the first few, then every 2 s as
  /// the Python server does, so a comma parked with nothing to run does not
  /// have its interface opened and closed twice a second all night.
  static let usbPoll: TimeInterval = 0.5
  static let usbQuietRetry: TimeInterval = 2.0
  static let usbQuickRetries = 5

  public let configuration: Configuration
  public let host: EngineHost
  public let cache: ServerCache
  public let backend: any EngineBackend
  /// What the host passed in besides the backend and the gadget.
  public var hooks: ServerHooks { host.hooks }

  private let log = ServerLog(category: "server")
  private let lock = NSLock()
  /// Internal so a test can end it under the accept loop.
  var listener: TCPListener?
  /// The session being served, and the latch its end releases.
  private var current: (session: Session, done: Latch)?
  /// Held by `takeover` across the whole swap, the wait for the old session
  /// to end included: that session's `serve` takes only `lock`, so the wait
  /// always ends.
  private let takeoverLock = NSLock()
  private var link: LinkEvent = .waiting
  private var ticker: Ticker?
  private var started = false
  private var stopped = false
  private var dial: DialTarget?
  private var dialing = false
  /// Where the USB loop finds the comma: IOKit's `USBGadget` on a Mac, the
  /// Android app's descriptor in a `UsbfsGadget`. Internal so a test can hand
  /// it a fake before `start`.
  var gadget: (any GadgetSource)?

  /// A server on the host's `backend` (JetlinkORT's CoreML or QNN backend, or
  /// the tests' own), its `gadget` if it serves USB, and its `hooks`.
  public init(
    configuration: Configuration, backend: any EngineBackend, gadget: (any GadgetSource)? = nil, hooks: ServerHooks = ServerHooks()
  ) throws {
    self.configuration = configuration
    self.dial = configuration.dial
    self.backend = backend
    self.gadget = gadget
    cache = try ServerCache(root: configuration.cacheRoot, backend: backend, keep: configuration.keepPlans)
    host = EngineHost(cache: cache, hooks: hooks)
    // A write to a socket the comma closed must be an error, not a signal
    // that kills the app.
    signal(SIGPIPE, SIG_IGN)
  }

  /// Listens, dials and opens the USB gadget as configured, and serves until
  /// `stop`. Returns once listening.
  public func start() throws {
    if configuration.listen {
      try listen()
    }
    lock.lock()
    started = true
    lock.unlock()
    if configuration.preload {
      host.preload()
    }
    ticker = Ticker(interval: Server.statsInterval) { [weak self] _ in self?.tick() }
    startDialing()
    startUSB()
  }

  private func listen() throws {
    let listener = try TCPListener(host: configuration.host, port: configuration.port)
    lock.lock()
    if self.listener != nil {
      // Listened again from two sides at once; the first one stands.
      lock.unlock()
      listener.close()
      return
    }
    self.listener = listener
    lock.unlock()
    log.info("listening on \(configuration.host):\(listener.port)")
    if currentLink.state != .connected {
      setLink(LinkEvent(state: .waiting, detail: "listening on \(configuration.host):\(listener.port)", peer: nil))
    }
    let thread = Thread { [self] in
      acceptLoop(listener)
      listenerEnded(listener)
    }
    thread.name = "jetlink-accept"
    // The first receive of a new connection starts on this thread's priority.
    thread.qualityOfService = .userInteractive
    thread.start()
  }

  /// Whether the listener is still accepting. iOS takes a suspended app's
  /// listening socket away, so the app asks this when it comes back.
  public var isListening: Bool {
    lock.lock()
    defer { lock.unlock() }
    return listener != nil
  }

  /// Listens again after the old socket was taken away. The engine stays
  /// loaded. A server configured not to listen stays that way.
  public func reopenListener() throws {
    guard configuration.listen else { return }
    lock.lock()
    let old = listener
    let isStopped = stopped
    listener = nil
    lock.unlock()
    guard !isStopped else { return }
    old?.close()
    try listen()
  }

  /// The accept loop ended on its own: iOS reclaimed the socket, or accept
  /// failed. Listen again, backing off while the system refuses, until the
  /// server stops or something else has listened meanwhile.
  private func listenerEnded(_ ended: TCPListener) {
    lock.lock()
    let wasCurrent = listener === ended
    if wasCurrent { listener = nil }
    let isStopped = stopped
    lock.unlock()
    guard wasCurrent && !isStopped else { return }
    log.warning("the listening socket went away; listening again")
    ended.close()
    var backoff = Server.relistenBackoff.lowerBound
    while true {
      Thread.sleep(forTimeInterval: backoff)
      lock.lock()
      let idle = listener == nil && !stopped
      lock.unlock()
      guard idle else { return }
      do {
        try listen()
        return
      } catch {
        log.warning("cannot listen yet: \(String(describing: error))")
        backoff = min(backoff * 2, Server.relistenBackoff.upperBound)
      }
    }
  }

  public var port: UInt16? {
    lock.lock()
    defer { lock.unlock() }
    return listener?.port
  }

  public var currentLink: LinkEvent {
    lock.lock()
    defer { lock.unlock() }
    return link
  }

  /// Frames the connected comma has been served.
  public var framesServed: Int {
    lock.lock()
    defer { lock.unlock() }
    return current?.session.frames ?? 0
  }

  /// The frames served over the last `window` seconds, as one summary: the
  /// dashboard's headline, steadier than the once-a-second event.
  public func recentStats(window: TimeInterval) -> StatsEvent? {
    host.frameStats.summary(window: window, framesTotal: framesServed)
  }

  /// Stops listening, dialing and serving. The engine stays loaded, for a
  /// server that will be started again; `shutdown()` releases it.
  public func stop() {
    lock.lock()
    stopped = true
    let listener = self.listener
    let session = current?.session
    self.listener = nil
    endServing()  // a session that ends after a stop says nothing
    lock.unlock()
    ticker?.stop()
    listener?.close()
    session?.interrupt()
  }

  /// `stop()`, release the engine and close the gadget: the process is ending.
  public func shutdown() {
    stop()
    host.close()
    gadget?.close()
  }

  // MARK: connections

  private func acceptLoop(_ listener: TCPListener) {
    while let transport = listener.accept() {
      takeover(transport)
    }
  }

  /// Serves `transport` on a thread of its own, after the session being
  /// served, if any, has been interrupted and has ended. Returns the latch
  /// the new session's end releases, and the session. Internal so a test can
  /// race two.
  @discardableResult
  func takeover(_ transport: any MessageLink) -> (done: Latch, session: Session) {
    let session = Session(transport: transport, host: host)
    let done = Latch()
    attach(session, done: done, takesOverOnAnnounce: false)
    displaceCurrent(with: (session, done))
    start(session, done: done)
    return (done, session)
  }

  /// Serves `transport` without touching the session being served until this
  /// one speaks. Over USB an open gadget is not a client: the comma's owner
  /// keeps the gadget on the bus while nothing on the comma serves it —
  /// between runs, and across a loan handover, whose re-bind alone used to
  /// interrupt whoever was connected here. The swap happens on the session's
  /// first message; a session that ends without one, the unserved gadget,
  /// leaves the session being served alone.
  @discardableResult
  func takeoverWhenAnnounced(_ transport: any MessageLink) -> (done: Latch, session: Session) {
    let session = Session(transport: transport, host: host)
    let done = Latch()
    attach(session, done: done, takesOverOnAnnounce: true)
    start(session, done: done)
    return (done, session)
  }

  /// Wires the session's callbacks. `takesOverOnAnnounce` moves the swap into
  /// the first link event, for a transport whose openness proves nothing (the
  /// gadget); the swap then runs on the session's own thread, inside the
  /// announce, before the link it reports replaces the one being served.
  private func attach(_ session: Session, done: Latch, takesOverOnAnnounce: Bool) {
    session.onLink = { [weak self] event, first in
      guard let self else { return }
      if first, takesOverOnAnnounce {
        displaceCurrent(with: (session, done))
      }
      let medium = event.linkMedium?.title ?? "an unknown link"
      if first {
        // install.sh waits for this line
        log.info("client connected from \(event.peer ?? "?") over \(medium)")
      } else {
        log.info("the comma's hello says its link is \(medium)")
      }
      setLink(event)
      if first {
        _ = hooks.gadgetIdle?(.connected)
        gadget?.sessionStarted()
      }
    }
    if let gadget {
      session.onMessage = { gadget.sessionHeard() }
    }
  }

  /// One swap at a time, from reading `current` to replacing it: the accept,
  /// dial and USB loops each take over, and two that saw the same session
  /// would each serve one of their own on the engine's one set of queues.
  private func displaceCurrent(with new: (session: Session, done: Latch)) {
    takeoverLock.lock()
    defer { takeoverLock.unlock() }
    lock.lock()
    let previous = current
    lock.unlock()
    if let previous {
      log.info("a new connection from \(new.session.peer) takes over from \(previous.session.peer)")
      previous.session.interrupt()
      previous.done.wait()
    }
    lock.lock()
    current = new
    lock.unlock()
  }

  private func start(_ session: Session, done: Latch) {
    let thread = Thread { [self] in
      serve(session)
      done.release()
    }
    thread.name = "jetlink-session"
    thread.qualityOfService = .userInteractive
    thread.start()
  }

  private func serve(_ session: Session) {
    let reason = session.serveForever()
    session.close()
    lock.lock()
    let isCurrent = current?.session === session
    if isCurrent {
      current = nil
    }
    let isStopped = stopped
    lock.unlock()
    // A gadget nobody on the comma was serving never connected, so it does
    // not disconnect either: the USB loop says so once, and retries quietly.
    guard session.announced else { return }
    _ = hooks.gadgetIdle?(.disconnected)
    gadget?.sessionEnded()
    log.info("client disconnected: \(reason)")
    if let summary = session.summary {
      log.info("the session with \(session.who) served \(summary)")
    }
    if isCurrent && !isStopped {
      setLink(LinkEvent(state: .disconnected, detail: reason, peer: nil))
    }
  }

  // MARK: dialing

  /// Where the server dials, if anywhere.
  public var dialTarget: DialTarget? {
    lock.lock()
    defer { lock.unlock() }
    return dial
  }

  /// Dial `target` from now on, or stop dialing with nil. A connection
  /// being served is left alone either way.
  public func setDial(_ target: DialTarget?) {
    lock.lock()
    let changed = dial != target
    dial = target
    lock.unlock()
    if changed, let target {
      log.info("dialing \(target.description)")
    } else if changed {
      log.info("no longer dialing")
    }
    startDialing()
  }

  /// Starts the dial thread if there is a target and none is running.
  private func startDialing() {
    lock.lock()
    guard started, !stopped, dial != nil, !dialing else {
      lock.unlock()
      return
    }
    dialing = true
    lock.unlock()
    let thread = Thread { [self] in dialLoop() }
    thread.name = "jetlink-dial"
    thread.qualityOfService = .userInteractive
    thread.start()
  }

  /// Connects, serves, and dials again when the session ends; retries
  /// every `dialInterval` while the target stands and the server runs.
  private func dialLoop() {
    var failed: DialTarget?
    while true {
      lock.lock()
      guard !stopped, let target = dial else {
        dialing = false
        lock.unlock()
        return
      }
      lock.unlock()
      do {
        let transport = try TCPTransport.connect(host: target.host, port: target.port, timeout: Server.dialTimeout)
        failed = nil
        log.info("dialed \(transport.peer)")
        takeover(transport).done.wait()
      } catch {
        // Once per outage, not twice a second.
        if failed != target {
          failed = target
          log.info("cannot reach \(target.description) yet: \(String(describing: error))")
        }
      }
      Thread.sleep(forTimeInterval: Server.dialInterval)
    }
  }

  // MARK: USB

  /// Starts the USB loop if the configuration asks for one.
  private func startUSB() {
    guard configuration.usb else { return }
    guard let gadget else {
      log.warning("serving over USB needs a gadget this platform can open; this server only listens and dials")
      return
    }
    setLink(LinkEvent(state: .waiting, detail: Server.usbWaiting, peer: nil))
    let thread = Thread { [self] in usbLoop(gadget) }
    thread.name = "jetlink-usb"
    thread.qualityOfService = .userInteractive
    thread.start()
  }

  static let usbWaiting = String(format: "waiting for a jetlink gadget at %04x:%04x", Pinned.usbVendorID, Pinned.usbProductID)

  /// Opens the gadget whenever it is on the bus and serves it, one session
  /// at a time, until the server stops. The Swift form of the Python
  /// server's `_usb_opener` in its serve loop.
  ///
  /// Presence is not readiness: the comma's owner keeps the gadget on the
  /// bus while no process on the comma is serving it, and then a read fails
  /// within milliseconds. Such a session never reported a connection, so
  /// it is retried without a link event, and said once in the log. Nor is
  /// presence a client: the claim takes over only when the comma speaks on
  /// it, so a session being served over TCP or a dial survives the gadget's
  /// idle periods and the loan handover's bounce.
  private func usbLoop(_ gadget: any GadgetSource) {
    var waiting = WaitLog(log: log)
    var quiet = 0
    while !isStopped {
      guard gadget.present() else {
        quiet = 0
        waiting.say(Server.usbWaiting)
        let link = currentLink.state
        if link == .disconnected {
          setLink(LinkEvent(state: .waiting, detail: Server.usbWaiting, peer: nil))
        }
        // A comma served over TCP meanwhile keeps the host up as a gadget does.
        if link != .connected, hooks.gadgetIdle?(.absent) == true {
          continue
        }
        Thread.sleep(forTimeInterval: Server.usbPoll)
        continue
      }
      _ = hooks.gadgetIdle?(.present)
      let transport: any MessageLink
      do {
        transport = try gadget.open()
      } catch {
        // Retried as an unserved gadget is: after a handover, a bounce or a
        // glitch the device's URBs die before its sysfs entry goes, so the
        // first opens claim the stale device and fail, and the comma is back
        // a poll later, not two seconds.
        quiet += 1
        waiting.say("could not open the gadget: \(String(describing: error))")
        Thread.sleep(forTimeInterval: quiet <= Server.usbQuickRetries ? Server.usbPoll : Server.usbQuietRetry)
        continue
      }
      let (done, session) = takeoverWhenAnnounced(transport)
      done.wait()
      if session.announced {
        // The comma closes the link between runs; the next run's hello is
        // already on its way, so open again at once.
        waiting.reset()
        quiet = 0
        continue
      }
      quiet += 1
      waiting.say("the comma's gadget is on the bus, but nothing on the comma is serving it yet")
      Thread.sleep(forTimeInterval: quiet <= Server.usbQuickRetries ? Server.usbPoll : Server.usbQuietRetry)
    }
  }

  private var isStopped: Bool {
    lock.lock()
    defer { lock.unlock() }
    return stopped
  }

  private func setLink(_ event: LinkEvent) {
    lock.lock()
    link = event
    if event.state == .connected {
      beginServing()
    } else {
      endServing()
    }
    lock.unlock()
    host.emit(.link(event))
  }

  /// Held while a comma is connected. The system throttles an app nobody is
  /// looking at (App Nap on a Mac), and an in-process server with it: behind a
  /// locked screen the Mac app ran the model at a p50 of 75 ms and dropped 45%
  /// of frames, where the same server as a command-line process ran at 36 ms.
  /// Not held while idle, and idle sleep stays allowed; frames are not user input.
  /// On Android the app's foreground service does this job.
  #if canImport(Darwin)
    private var servingActivity: NSObjectProtocol?
  #endif

  /// Under `lock`.
  private func beginServing() {
    #if canImport(Darwin)
      guard servingActivity == nil else { return }
      servingActivity = ProcessInfo.processInfo.beginActivity(
        options: [.userInitiatedAllowingIdleSystemSleep, .latencyCritical], reason: "serving the comma's model")
    #endif
  }

  /// Under `lock`.
  private func endServing() {
    #if canImport(Darwin)
      guard let held = servingActivity else { return }
      ProcessInfo.processInfo.endActivity(held)
      servingActivity = nil
    #endif
  }

  private var lastTick = ProcessInfo.processInfo.systemUptime

  /// A frame summary a second while a comma is connected and sending, and
  /// between sessions a warm run for an engine that cools. The window is the
  /// time since the last tick, so no frame falls between two.
  private func tick() {
    let now = ProcessInfo.processInfo.systemUptime
    let window = now - lastTick
    lastTick = now
    host.warmIfIdle(now: now)
    lock.lock()
    let connected = link.state == .connected
    let frames = current?.session.frames ?? 0
    lock.unlock()
    guard connected, let stats = host.frameStats.summary(window: window, framesTotal: frames) else { return }
    host.emit(.stats(stats))
  }
}

/// Says why there is no comma once, then keeps quiet about it: the USB loop
/// polls every half second, and a Mac parked overnight would otherwise fill
/// its log with one line. A new reason gets its own line; a session resets it.
struct WaitLog {
  let log: ServerLog
  private var last: String?

  init(log: ServerLog) {
    self.log = log
  }

  mutating func say(_ message: String) {
    guard message != last else { return }
    last = message
    log.warning(message)
  }

  mutating func reset() {
    last = nil
  }
}
