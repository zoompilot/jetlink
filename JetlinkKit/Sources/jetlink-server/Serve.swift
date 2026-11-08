#if os(macOS) || os(Linux)
  import ArgumentParser
  import Dispatch
  import Foundation
  import JetlinkKit
  import JetlinkLog
  import JetlinkRegistry
  import JetlinkServer
  import JetlinkStatusPage
  #if os(Linux)
    import JetlinkLinux
    import JetlinkTRT
  #endif

  extension Log.Level: ExpressibleByArgument {}

  /// --cache, which every command that touches the cache takes.
  struct CacheArguments: ParsableArguments {
    @Option(
      help: "Where models and engines live. Default: $JETLINK_CACHE, /mnt/data/jetlink on a Jetson, /var/lib/jetlink as root, else the user's cache."
    )
    var cache: String?

    var root: URL { cache.map { URL(fileURLWithPath: $0, isDirectory: true) } ?? defaultCache() }
  }

  /// `jetlink-server serve`, the default: serves the comma until SIGINT or
  /// SIGTERM, then exits 0.
  struct Serve: ParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Serve the comma (the default).")
    /// Said once, when the server has started; install.sh matches it as it is.
    static let servingLine = "jetlink-server is serving"

    @OptionGroup var chosen: BackendArguments
    @Flag(help: "Be the USB host for the comma's gadget. No TCP listener then, unless --listen too.")
    var usb = false
    @Flag(help: "Listen on TCP, as without --usb.")
    var listen = false
    @Option(help: "The address to listen on.")
    var host = "0.0.0.0"
    @Option(help: "The TCP port to listen on.")
    var port = Wire.defaultPort
    @Option(help: ArgumentHelp("Also dial this end and serve it, as the phone dials the comma over a USB network link.", valueName: "host[:port]"))
    var dial: String?
    @OptionGroup var cache: CacheArguments
    @Option(help: "Suspend after this many seconds with no gadget (Linux, with --usb); 0 never.")
    var sleepAfter = 0.0
    @Option(help: "Serve the web page on this port; 0 is off.")
    var statusPort = 0
    @Option(
      help: ArgumentHelp(
        "The web page's password file (web-password writes it). With it the page signs in and has the controls; without it, or without the file, it is read-only.",
        valueName: "file"))
    var webAuth: String?
    @Flag(help: "Power this machine off when the comma asks (Linux). Without it the comma is told ok and the machine stays up.")
    var poweroff = false
    @Option(help: "A Jetson's clocks: on holds them at the power mode's top while a comma is connected or an engine builds, off leaves them to NVIDIA's scaling. Without it they stay as they are.")
    var clocks: String?
    @Option(help: "The nvpmodel power mode whose limits --clocks keeps to (MAXN_SUPER).")
    var powerMode: String?
    @Flag(name: .customLong("no-preload"), help: "Do not load the engine loaded last before a comma asks.")
    var noPreload = false
    @Flag(name: .customLong("no-keepalive"), help: "Do not keep the GPU clocked up between frames (a Mac's onnxruntime).")
    var noKeepAlive = false
    @Flag(name: .customLong("no-cpu-keepwarm"), help: "Do not keep a CPU core busy between Neural Engine frames (a Mac's onnxruntime).")
    var noCPUKeepWarm = false
    @Option(help: "debug, info, warning or error.")
    var logLevel = Log.Level.info

    func validate() throws {
      if let dial, DialTarget(dial) == nil {
        throw ValidationError("--dial wants HOST or HOST:PORT, not \(dial)")
      }
      guard sleepAfter >= 0 else { throw ValidationError("--sleep-after cannot be negative") }
      // A TCP listener never sees the comma go, so nothing would say when to sleep.
      guard sleepAfter == 0 || usb else { throw ValidationError("--sleep-after needs --usb") }
      guard (0...65535).contains(statusPort) else { throw ValidationError("--status-port \(statusPort) is not a port") }
      guard clocks == nil || (["on", "off"].contains(clocks) && powerMode != nil) else {
        throw ValidationError("--clocks takes on or off, with --power-mode")
      }
    }

    func run() throws {
      holdStopSignals()
      setUpLogging(logLevel)
      let log = ServerLog(category: "main")
      let backend = try chosen.pick(keepAlive: !noKeepAlive, keepCPUWarm: !noCPUKeepWarm)
      let root = cache.root

      var hooks = ServerHooks()
      var gadget: (any GadgetSource)?
      #if os(macOS)
        gadget = USBGadget()
      #elseif os(Linux)
        // The gadget through sysfs, and telemetry, the sleeper and poweroff
        // as hooks; NVML reads the GPU TensorRT runs on.
        gadget = SysfsGadget()
        let telemetry = LinuxHost.telemetry(gpu: (backend as? TrtBackend)?.trt.device ?? 0)
        hooks = LinuxHost.hooks(sleepAfter: sleepAfter, poweroff: poweroff, telemetry: telemetry)
        let jetsonClocks = powerMode.flatMap { JetsonClocks(modeName: $0) }
      #endif
      hooks.fatal = exitOnFatal

      let server: Server
      let controller: ServerController
      do {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        server = try Server(
          configuration: Server.Configuration(
            host: host, port: port, cacheRoot: root, preload: !noPreload, dial: dial.flatMap { DialTarget($0) }, listen: listen || !usb, usb: usb),
          backend: backend, gadget: gadget, hooks: hooks)
        #if os(Linux)
          // the clocks follow the comma and the builds, or go back to NVIDIA's scaling
          if clocks == "on" { jetsonClocks?.follow(server.host) } else if clocks == "off" { jetsonClocks?.release() }
        #endif
        // Made before the server starts, so it hears the first link event:
        // a comma on the bus at boot connects at once. With the page off too:
        // it downloads a model the comma asks for and has not got either.
        controller = ServerController(server: server, registry: Registry(layout: server.cache.layout))
        log.info("backend \(backend.name) \(backend.runtimeVersion) on \(backend.deviceTag()), cache \(root.path)")
        try server.start()
        // The installer waits for this line. It is said whatever the comma is
        // doing: a comma on the bus that nothing on it serves yet gets no
        // line of its own that says the server is up.
        log.info(Serve.servingLine)
      } catch {
        log.error("cannot serve: \(error)")
        throw ExitCode.failure
      }
      var hardware: (any PageHardwareSource)?
      #if os(Linux)
        hardware = PageHardware(cache: root, gpu: telemetry)
      #endif
      let page = statusPort > 0 ? startPage(controller, hardware: hardware, log: log) : nil
      stopOnSignals { signal in
        log.info("stopping on \(signal)")
        // The comma's server, its engine and gadget first, so a frame in
        // flight is answered or cut before anything else goes, then the
        // status page, which shows the server stopping until the end.
        server.shutdown()
        log.info("stopped the server")
        if let page {
          page.stop()
          log.info("stopped the web page")
        }
        // held to the end: the host keeps its fetcher weakly
        withExtendedLifetime(controller) {}
      }
    }

    /// The web page, on its own low-priority threads. It observes
    /// `controller`, and sends it a command only for a signed-in page, so an
    /// open page never starts a catalog fetch, a download or a build by
    /// itself. Without its page it stays off: the comma matters more than a
    /// page.
    private func startPage(_ controller: ServerController, hardware: (any PageHardwareSource)?, log: ServerLog) -> PageServer? {
      let auth = webAuth.map { WebAuth(url: URL(fileURLWithPath: $0)) }
      var system: (any PageSystem)?
      #if os(Linux)
        system = PageHost()
      #endif
      do {
        let page = try PageServer.start(
          port: statusPort, controller: controller, version: productVersion(), hardware: hardware, auth: auth, system: system)
        let signIn = auth == nil ? "read-only" : auth!.isConfigured ? "with sign-in" : "read-only until \(auth!.url.path) is written"
        log.info("web page on port \(page.port), \(signIn)")
        return page
      } catch {
        log.warning("no web page: \(error)")
        return nil
      }
    }
  }

  /// Log lines on standard error, from `threshold` up: journald adds the time
  /// under systemd. Linux's loggers write there already; a Mac's go to the
  /// unified log, so they are copied out.
  func setUpLogging(_ threshold: Log.Level) {
    Log.threshold = threshold
    #if os(macOS)
      Log.sink = { level, category, message in
        FileHandle.standardError.write(Data((EmbeddedServer.logLine(level, category, message) + "\n").utf8))
      }
    #endif
  }

  /// Exits 3 after an engine error nothing recovers from: only a new process
  /// gets a working device back, and systemd starts one, which preloads the
  /// engine.
  @Sendable func exitOnFatal(_ error: any Error) {
    ServerLog(category: "main").error("exiting on an engine error nothing recovers from: \(error)")
    exitNow(3)
  }

  /// Flushes every output stream, then ends the process without exit()'s
  /// atexit handlers and C++ static destructors: a TensorRT build may still
  /// be running on its own thread, and tearing TensorRT's statics down under
  /// it aborts or faults, a crash in the journal after a stop. fflush(nil)
  /// rather than stdout and stderr, which Swift 6 on Linux will not touch.
  func exitNow(_ code: Int32) -> Never {
    fflush(nil)
    _exit(code)
  }

  let stopSignals = [(SIGINT, "SIGINT"), (SIGTERM, "SIGTERM")]

  /// Keeps SIGINT and SIGTERM pending for `stopOnSignals` rather than
  /// delivered: on Linux the dispatch source reads them from a signalfd, which
  /// only sees a signal no thread took, and a thread that has it unblocked
  /// takes and drops it. Called before any thread starts, so all inherit it.
  func holdStopSignals() {
    #if os(Linux)
      var held = sigset_t()
      sigemptyset(&held)
      for (number, _) in stopSignals {
        sigaddset(&held, number)
      }
      pthread_sigmask(SIG_BLOCK, &held, nil)
    #endif
  }

  /// Runs `stop` on the main queue at SIGINT or SIGTERM, then exits 0 by
  /// `exitNow`. Never returns.
  func stopOnSignals(_ stop: @escaping @Sendable (String) -> Void) -> Never {
    let sources = stopSignals.map { number, name in
      #if os(macOS)
        // Handled by the source alone; a shell that starts this in the
        // background hands it SIGINT ignored, which the source still hears.
        signal(number, SIG_IGN)
      #endif
      // Not on Linux, where the signals are held: ignoring one drops it if
      // it is already pending, a SIGTERM that came during start-up.
      let source = DispatchSource.makeSignalSource(signal: number, queue: .main)
      source.setEventHandler {
        stop(name)
        exitNow(0)
      }
      source.resume()
      return source
    }
    withExtendedLifetime(sources) { dispatchMain() }
  }

  /// $JETLINK_CACHE, else Linux's rule (JetlinkLinux's Platform), else a
  /// Mac's user cache directory.
  func defaultCache(environment: [String: String] = ProcessInfo.processInfo.environment) -> URL {
    if let named = environment["JETLINK_CACHE"], !named.isEmpty {
      return URL(fileURLWithPath: named, isDirectory: true)
    }
    #if os(Linux)
      return Platform.defaultCache(environment: environment)
    #else
      return FileManager.default.homeDirectoryForCurrentUser.appending(path: "Library/Caches/jetlink", directoryHint: .isDirectory)
    #endif
  }
#endif
