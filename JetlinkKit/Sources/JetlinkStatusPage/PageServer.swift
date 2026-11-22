#if canImport(Darwin) || canImport(Glibc)
  import Foundation
  import JetlinkKit
  import JetlinkLog
  import JetlinkServer
  #if canImport(CryptoKit)
    import CryptoKit
  #else
    import Crypto
  #endif
  #if canImport(Glibc)
    import Glibc
  #else
    import Darwin
  #endif

  /// The web page, served by the daemon itself for a phone on the comma's
  /// hotspot: `/` the page, `/events` a stream of Server-Sent Events, `/logs`
  /// the last lines this process logged, and under `/api/` the sign-in and
  /// the controls. With a password file (`WebAuth`) everything but the page
  /// and the sign-in needs the cookie; without one the page is read-only
  /// and nothing here sends the server a command.
  ///
  /// The server comes first: every thread here runs at the lowest priority,
  /// the one lock a server thread could meet is behind `PageFeed`'s handoff
  /// queue, which never makes it wait, and while the car drives nothing that
  /// restarts, builds, downloads or deletes may start.
  public final class PageServer: @unchecked Sendable {
    /// The request line and headers, with the blank line after them.
    static let headerBytes = 8 * 1024
    /// A command's or a setting's JSON.
    static let bodyBytes = 16 * 1024
    /// How long a command may take to answer: a download's pointer lookup.
    static let commandTimeout: TimeInterval = 30
    /// The controller's commands a signed-in page may send. Imports name a
    /// path on this computer, and shutdown and unload are the apps'.
    static let commands: Set<String> = ["status", "catalog", "download", "cancel_download", "prepare", "forget", "inventory", "benchmark", "cancel_benchmark"]
    /// What takes the GPU, the disk or the network from the frames.
    static let heavy: Set<String> = ["download", "prepare", "forget", "benchmark"]
    static let parkedOnly = PageResponse.error(409, "The car is driving on the big model. Try again when it is parked.")
    static let needsPassword = PageResponse.error(403, "Set a password for the web page first. On the Jetson: sudo jetlink password")
    static let signInFirst = PageResponse.error(401, "Sign in first.")
    /// Connections at once of any kind; more are closed unanswered.
    static let connections = 32
    /// A page that stops reading is dropped after this long.
    static let writeTimeout: TimeInterval = 10
    static let logLines = 300

    /// Where it listens; the one asked for, or the one the system gave for 0.
    public let port: UInt16

    let feed: PageFeed
    /// A connection that has not sent its request by then is dropped.
    private let headerTimeout: TimeInterval
    /// Quiet before a comment on an event stream, so a phone or a proxy
    /// keeps it open.
    private let keepalive: TimeInterval
    /// The page, and the 304 for a browser that has this build's.
    private let page: (whole: Data, unchanged: Data, etag: String)
    private let logs: @Sendable () -> [String]
    private let auth: WebAuth?
    private let system: (any PageSystem)?
    /// The add-on server's state for the Models tab, where there is one.
    private let addons: (@Sendable () -> [String: Any])?
    private let log = ServerLog(category: "page")
    private let listener: TCPListener
    private let condition = NSCondition()
    private var stopped = false
    private var threads = 0
    private var connections = 0
    private var detach: (() -> Void)?
    private var controller: ServerController?

    /// The page on every address at `port` (0: any free port), fed by
    /// `controller`, signed in with `auth`'s file and changing `system`.
    /// Throws `StatusPage.Unavailable` when the page's bundle is missing, or
    /// why it cannot listen.
    public static func start(
      port: Int, controller: ServerController, version: String, hardware: (any PageHardwareSource)?, auth: WebAuth? = nil,
      system: (any PageSystem)? = nil, addons: (@Sendable () -> [String: Any])? = nil
    ) throws -> PageServer {
      let server = try PageServer(
        port: port, page: StatusPage.page(), feed: PageFeed(hardware: hardware), auth: auth, system: system, addons: addons
      ) {
        LogRing.shared.lines()
      }
      server.watch(controller, version: version)
      return server
    }

    init(
      port: Int, page: Data, feed: PageFeed, auth: WebAuth? = nil, system: (any PageSystem)? = nil,
      addons: (@Sendable () -> [String: Any])? = nil, headerTimeout: TimeInterval = 30,
      keepalive: TimeInterval = 15, logs: @escaping @Sendable () -> [String]
    ) throws {
      guard let port = UInt16(exactly: port) else { throw HostError.invalid("\(port) is not a port") }
      // IPv6 with IPv4 mapped in, since a phone may resolve the Jetson's name
      // to either
      listener = try TCPListener(port: port, dualStack: true)
      self.port = listener.port
      let etag = "\"" + SHA256.hash(data: page).prefix(8).map { String(format: "%02x", $0) }.joined() + "\""
      let replies = PageResponse.page(page, etag: etag)
      self.page = (replies.whole, replies.unchanged, etag)
      self.feed = feed
      self.headerTimeout = headerTimeout
      self.keepalive = keepalive
      self.logs = logs
      self.auth = auth
      self.system = system
      self.addons = addons
      threads = 1
      PageThread.start("jetlink-page") { [self] in acceptLoop() }
    }

    /// Relays `controller`'s events from now on, after the hello and what the
    /// controller knows already. Called on the daemon's thread: the state is
    /// read here, never from a page's thread, because it takes the host's
    /// lock, which a frame holds.
    func watch(_ controller: ServerController, version: String) {
      feed.publish(.hello(HelloEvent(version: version)))
      let id = controller.observe { [feed] in feed.publish($0) }
      feed.seed(controller.currentState())
      condition.lock()
      detach = { controller.stopObserving(id) }
      self.controller = controller
      condition.unlock()
    }

    /// Tells the open pages the server is stopping, closes them and stops
    /// listening. Waits a second at most for the page's threads.
    public func stop() {
      condition.lock()
      let first = !stopped
      stopped = true
      let detach = self.detach
      self.detach = nil
      controller = nil
      condition.unlock()
      guard first else { return }
      detach?()
      feed.stop()
      listener.close()
      let deadline = Date(timeIntervalSinceNow: 1)
      condition.lock()
      while threads > 0 && condition.wait(until: deadline) {}
      condition.unlock()
    }

    private var isStopped: Bool {
      condition.lock()
      defer { condition.unlock() }
      return stopped
    }

    private func threadEnded(connection: Bool) {
      condition.lock()
      threads -= 1
      if connection { connections -= 1 }
      condition.broadcast()
      condition.unlock()
    }

    // MARK: connections

    private func acceptLoop() {
      defer { threadEnded(connection: false) }
      while let (client, _, host) = listener.acceptConnection() {
        condition.lock()
        let admitted = !stopped && connections < PageServer.connections
        if admitted {
          connections += 1
          threads += 1
        }
        condition.unlock()
        guard admitted else {
          Sys.close(client)
          continue
        }
        PageThread.start("jetlink-page-client") { [self] in
          serve(client, address: host.hasPrefix("::ffff:") ? String(host.dropFirst(7)) : host)
          Sys.close(client)
          threadEnded(connection: true)
        }
      }
    }

    /// `address`, the client's host without its port: the sign-in throttle
    /// counts by it, and every connection has a port of its own.
    private func serve(_ client: Int32, address: String) {
      PageServer.tune(client)
      let reply: Data
      guard let (request, bytes) = readRequest(client) else { return }
      switch request {
      case .request(let head):
        guard let routed = route(head, client, bytes, address: address) else { return }
        reply = routed
      case .bad:
        reply = PageResponse.text(400, "bad request\n")
      case .tooLarge:
        reply = PageResponse.text(431, "request too large\n")
      case .incomplete:
        reply = PageResponse.text(408, "no request\n")
      }
      if write(client, reply) {
        linger(client)
      }
    }

    // MARK: routes

    /// The reply to `head`, or nil when the connection became an event
    /// stream or went away while its body was read.
    private func route(_ head: PageHead, _ client: Int32, _ bytes: [UInt8], address: String) -> Data? {
      // read once: the file may change under a running server
      let file = auth?.file
      let session = file.flatMap { auth?.session(cookieHeader: head.headers["cookie"], file: $0) }
      // Without a password file the page is the read-only one it always was.
      let mayWatch = file == nil || session != nil
      switch (head.method, head.path) {
      case ("GET", "/"):
        return head.headers["if-none-match"] == page.etag ? page.unchanged : page.whole
      case ("GET", "/api/session"):
        let stale = file != nil && session == nil && WebAuth.cookieValue(head.headers["cookie"]) != nil
        return PageResponse.json(
          200, ["auth": file == nil ? "none" : "required", "signed_in": session != nil],
          cookie: session?.renewal ?? (stale ? WebAuth.clearCookie : nil))
      case ("GET", "/events"):
        guard mayWatch else { return PageServer.signInFirst }
        stream(client)
        return nil
      case ("GET", "/logs"):
        guard mayWatch else { return PageServer.signInFirst }
        let lines = logs().suffix(PageServer.logLines)
        return PageResponse.text(200, lines.isEmpty ? "" : lines.joined(separator: "\n") + "\n", cache: "no-store")
      case ("GET", "/api/addons"):
        guard mayWatch else { return PageServer.signInFirst }
        return PageResponse.json(200, addons?() ?? ["served": false])
      case ("GET", "/api/system"):
        if let refusal = refusal(file, session) { return refusal }
        return PageResponse.json(200, system?.info() ?? ["installed": false])
      case ("GET", "/api/task"):
        if let refusal = refusal(file, session) { return refusal }
        return PageResponse.json(200, system?.task() ?? ["state": "none"])
      case ("POST", let path) where path.hasPrefix("/api/"):
        if let refusal = crossSite(head) { return refusal }
        let body: [String: Any]
        switch readBody(head, client, bytes) {
        case .object(let object): body = object
        case .reply(let reply): return reply
        case .gone: return nil
        }
        if path == "/api/logout" { return PageResponse.json(200, ["ok": true], cookie: WebAuth.clearCookie) }
        guard let auth, let file else { return PageServer.needsPassword }
        if path == "/api/login" { return login(body, auth, file, address: address) }
        guard session != nil else { return PageServer.signInFirst }
        return post(path, body, auth, file, address: address)
      default:
        return PageResponse.text(404, "not found\n")
      }
    }

    /// Why a control may not be used: no password file, or no sign-in.
    private func refusal(_ file: AuthFile?, _ session: WebAuth.Session?) -> Data? {
      file == nil ? PageServer.needsPassword : session == nil ? PageServer.signInFirst : nil
    }

    /// A POST must come from this page: the `X-Jetlink` header, which another
    /// site cannot add without a preflight this server never answers, and
    /// an Origin, when the browser sends one, that names this host.
    private func crossSite(_ head: PageHead) -> Data? {
      let refused = PageResponse.error(403, "Requests come from the page.")
      guard head.headers["x-jetlink"] == "1" else { return refused }
      if let site = head.headers["sec-fetch-site"], site != "same-origin", site != "none" { return refused }
      if let origin = head.headers["origin"] {
        guard let host = head.headers["host"], origin == "http://\(host)" || origin == "https://\(host)" else { return refused }
      }
      return nil
    }

    private enum Body {
      case object([String: Any])
      case reply(Data)
      case gone
    }

    /// The JSON object a POST carries, read to its Content-Length.
    private func readBody(_ head: PageHead, _ client: Int32, _ bytes: [UInt8]) -> Body {
      guard (head.headers["content-type"] ?? "").lowercased().hasPrefix("application/json") else {
        return .reply(PageResponse.error(415, "Send JSON."))
      }
      guard let length = head.contentLength else { return .reply(PageResponse.error(411, "Say how long the body is.")) }
      guard length >= 0 else { return .reply(PageResponse.error(400, "Content-Length is not a number.")) }
      guard length <= PageServer.bodyBytes else { return .reply(PageResponse.error(413, "The body is too large.")) }
      var body = Array(bytes.dropFirst(head.length).prefix(length))
      let deadline = Date(timeIntervalSinceNow: 10)
      while body.count < length {
        switch receive(client, until: deadline, max: length - body.count) {
        case .bytes(let more): body += more
        case .timedOut: return .reply(PageResponse.error(408, "The body did not arrive."))
        case .gone: return .gone
        }
      }
      guard let object = (try? JSONSerialization.jsonObject(with: Data(body))) as? [String: Any] else {
        return .reply(PageResponse.error(400, "The body is not a JSON object."))
      }
      return .object(object)
    }

    private func login(_ body: [String: Any], _ auth: WebAuth, _ file: AuthFile, address: String) -> Data {
      guard let password = body["password"] as? String, !password.isEmpty else { return PageResponse.error(400, "Type the password.") }
      switch auth.login(password, file: file, address: address) {
      case .signedIn(let cookie):
        log.info("web page: signed in from \(address)")
        return PageResponse.json(200, ["ok": true], cookie: cookie)
      case .refused(let check):
        if case .wrong = check { log.warning("web page: a wrong password from \(address)") }
        return PageServer.refused(check)
      }
    }

    /// A password check that did not pass: 401, or 429 while held up, with
    /// the seconds to wait.
    private static func refused(_ check: WebAuth.Check, wrong: String = "That password is not right.") -> Data {
      switch check {
      case .wait(let wait):
        let seconds = Int(wait.rounded(.up))
        return PageResponse.json(429, ["error": "Too many tries. Wait \(seconds) s.", "retry_after": seconds])
      case .wrong(let wait):
        var object: [String: Any] = ["error": wrong]
        if let wait { object["retry_after"] = Int(wait.rounded(.up)) }
        return PageResponse.json(401, object)
      case .right:
        return PageResponse.error(500, "A right password was refused.")
      }
    }

    /// The signed-in POSTs.
    private func post(_ path: String, _ body: [String: Any], _ auth: WebAuth, _ file: AuthFile, address: String) -> Data {
      switch path {
      case "/api/logout-all":
        do {
          let cookie = try auth.signOutEverywhere(file: file)
          log.info("web page: signed out every other device")
          return PageResponse.json(200, ["ok": true], cookie: cookie)
        } catch {
          return PageResponse.error(500, "Could not sign the others out: \(error.localizedDescription)")
        }
      case "/api/password":
        guard let current = body["current"] as? String, let new = body["new"] as? String else {
          return PageResponse.error(400, "Send the current password and the new one.")
        }
        switch auth.changePassword(current: current, new: new, file: file, address: address) {
        case .changed(let cookie):
          log.info("web page: the password was changed from \(address)")
          return PageResponse.json(200, ["ok": true], cookie: cookie)
        case .refused(let check):
          return PageServer.refused(check, wrong: "The current password is not right.")
        case .rejected(let why):
          return PageResponse.error(400, why)
        case .failed(let why):
          return PageResponse.error(500, why)
        }
      case "/api/command":
        return command(body)
      case "/api/settings":
        guard let system else { return PageResponse.error(409, "Settings are for an installed Jetlink.") }
        guard let settings = body as? [String: String] else { return PageResponse.error(400, "Each setting is a word, as the installer takes it.") }
        if feed.isDriving { return PageServer.parkedOnly }
        do {
          try system.apply(settings)
          return PageResponse.json(200, ["ok": true, "task": system.task()])
        } catch {
          return PageResponse.error(error.status, error.description)
        }
      case "/api/action":
        guard let system else { return PageResponse.error(409, "These are for an installed Jetlink.") }
        guard let action = (body["action"] as? String).flatMap(PageAction.init(rawValue:)) else { return PageResponse.error(400, "Say which action.") }
        if action.interruptsComma && feed.isDriving { return PageServer.parkedOnly }
        do {
          var reply = try system.perform(action, seconds: (body["seconds"] as? NSNumber)?.intValue)
          reply["ok"] = true
          return PageResponse.json(200, reply)
        } catch {
          return PageResponse.error(error.status, error.description)
        }
      default:
        return PageResponse.text(404, "not found\n")
      }
    }

    /// One of the controller's commands, as the apps send them.
    private func command(_ body: [String: Any]) -> Data {
      let command: ControlCommand
      do {
        command = try ControlCommand(object: body)
      } catch {
        return PageResponse.error(400, "\(error)")
      }
      guard PageServer.commands.contains(command.name) else { return PageResponse.error(403, "The page does not send \(command.name).") }
      if PageServer.heavy.contains(command.name) && feed.isDriving { return PageServer.parkedOnly }
      condition.lock()
      let controller = self.controller
      condition.unlock()
      guard let controller else { return PageResponse.error(503, "The server is not ready.") }
      guard let reply = blocking(timeout: PageServer.commandTimeout, { await controller.handle(command) }) else {
        return PageResponse.error(504, "The server took too long to answer.")
      }
      var object = ControlEvent.reply(reply).payload()
      object["id"] = nil
      return PageResponse.json(reply.ok ? 200 : 409, object)
    }

    /// Closing with unread bytes waiting resets the connection, and a reset
    /// can throw away the reply before the client reads it: the end of an
    /// oversized request, say. So stop sending, and read what is left for up
    /// to a second, or until the server stops: a browser's idle spare socket
    /// never answers, and the daemon's exit waits for this thread.
    private func linger(_ client: Int32) {
      _ = shutdown(client, Int32(SHUT_WR))
      let deadline = Date(timeIntervalSinceNow: 1)
      var sink = [UInt8](repeating: 0, count: 4096)
      var drained = 0
      while drained < 1 << 16 && !isStopped {
        let left = deadline.timeIntervalSinceNow
        if left <= 0 { return }
        var poller = pollfd(fd: client, events: Int16(POLLIN), revents: 0)
        let ready = poll(&poller, 1, Int32(min(left, 0.1) * 1000) + 1)
        if ready < 0 && errno != EINTR { return }
        if ready <= 0 { continue }
        let n = sink.withUnsafeMutableBytes { recv(client, $0.baseAddress, $0.count, 0) }
        if n <= 0 { return }
        drained += n
      }
    }

    /// A page's socket: replies go out at once, and a phone that vanished
    /// without closing (it left the hotspot, its radio slept) frees its
    /// stream in about half a minute rather than once the send buffer
    /// fills. Keepalive probes after 15 s of silence, 5 s apart, 3 of them,
    /// find an idle one; on Linux, TCP_USER_TIMEOUT drops one whose events
    /// go unacknowledged for 30 s. Loopback never loses a packet, so the
    /// tests check the options, not a vanishing phone.
    static func tune(_ fd: Int32) {
      Sys.tune(fd, keepalive: (15, 5, 3), sendTimeout: writeTimeout)
      #if canImport(Glibc)
        Sys.set(fd, Int32(IPPROTO_TCP), TCP_USER_TIMEOUT, 30_000)
      #endif
    }

    /// The request, or `.incomplete` once the header timeout passes; nil
    /// when the client went away first, or the server stops. A browser opens
    /// a spare connection it may never send on, and answering that at a
    /// stop would only hold the exit up in `linger`.
    private func readRequest(_ client: Int32) -> (PageRequest, [UInt8])? {
      let deadline = Date(timeIntervalSinceNow: headerTimeout)
      var bytes: [UInt8] = []
      while true {
        // never more than the cap holds: past it the answer is 431 anyway
        switch receive(client, until: deadline, max: max(1, PageServer.headerBytes - bytes.count)) {
        case .bytes(let more): bytes += more
        case .timedOut: return (.incomplete, bytes)
        case .gone: return nil
        }
        let request = PageRequest.parse(bytes, limit: PageServer.headerBytes)
        if request != .incomplete { return (request, bytes) }
      }
    }

    private enum Received {
      case bytes(ArraySlice<UInt8>)
      case timedOut
      /// The client closed, or the server stops.
      case gone
    }

    /// Up to `max` bytes, as soon as some come, waiting in short polls so a
    /// stop is noticed.
    private func receive(_ client: Int32, until deadline: Date, max: Int) -> Received {
      var chunk = [UInt8](repeating: 0, count: min(max, 4096))
      while !isStopped {
        let left = deadline.timeIntervalSinceNow
        if left <= 0 { return .timedOut }
        var poller = pollfd(fd: client, events: Int16(POLLIN), revents: 0)
        let ready = poll(&poller, 1, Int32(min(left, 0.25) * 1000) + 1)
        if ready < 0 && errno != EINTR { return .gone }
        if ready <= 0 { continue }
        let n = chunk.withUnsafeMutableBytes { recv(client, $0.baseAddress, $0.count, 0) }
        if n < 0 && (errno == EINTR || errno == EAGAIN) { continue }
        if n <= 0 { return .gone }
        return .bytes(chunk[..<n])
      }
      return .gone
    }

    private func stream(_ client: Int32) {
      guard let stream = feed.open() else {
        _ = write(client, PageResponse.text(503, "too many pages open\n"))
        return
      }
      defer { feed.close(stream) }
      // retry: how soon a browser's own reconnect comes; the page backs off
      // on its own when that fails.
      guard write(client, PageResponse.head(200, type: "text/event-stream") + Data("retry: 2000\n\n".utf8)) else { return }
      while let frames = feed.next(stream, timeout: keepalive) {
        if peerClosed(client) { return }
        guard write(client, frames.isEmpty ? PageFeed.keepalive : Data(frames.joined())) else { return }
      }
    }

    /// A page that closed its end: a reload, a phone locking. Found before
    /// the next write rather than by it.
    private func peerClosed(_ client: Int32) -> Bool {
      var byte: UInt8 = 0
      let n = recv(client, &byte, 1, Int32(MSG_PEEK | MSG_DONTWAIT))
      return n == 0 || n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR
    }

    private func write(_ client: Int32, _ data: Data) -> Bool {
      data.withUnsafeBytes { raw in
        var offset = 0
        while offset < raw.count {
          let n = send(client, raw.baseAddress! + offset, raw.count - offset, Sys.sendFlags)
          if n < 0 && errno == EINTR { continue }
          // EAGAIN here is SO_SNDTIMEO: a page that stopped reading.
          if n <= 0 { return false }
          offset += n
        }
        return true
      }
    }
  }
#endif
