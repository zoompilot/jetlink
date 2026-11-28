#if os(Android)
  import Android
  import Foundation
  import JetlinkKit
  import JetlinkLiteRT
  import JetlinkLog
  import JetlinkORT
  import JetlinkServer

  // The Android app's way into the server: plain JNI functions on
  // io.zoompilot.jetlink.server.Native, strings in and out.
  //
  // Kotlin draws and Swift decides. The app sends control commands in the
  // control protocol's JSON (docs/control-protocol.md), and reads one snapshot
  // of everything its screens show, built here from the same events and the
  // same ModelRowBuilder the iPhone and Mac apps use. So the Android app holds
  // no server logic of its own.
  //
  // Every call that waits (command, snapshot) is made from a Kotlin background
  // thread. Nothing here calls back into Java, and nothing runs on the main
  // actor, which a library loaded through JNI never drains.

  // MARK: JNI strings

  public typealias Env = UnsafeMutablePointer<JNIEnv?>

  func string(_ env: Env, _ value: jstring?) -> String {
    guard let value, let chars = env.pointee!.pointee.GetStringUTFChars(env, value, nil) else { return "" }
    defer { env.pointee!.pointee.ReleaseStringUTFChars(env, value, chars) }
    return String(cString: chars)
  }

  /// JSON the app parses, as a Java string. JNI takes modified UTF-8, so
  /// anything outside the Basic Multilingual Plane goes as a \u escape pair.
  func jstring(_ env: Env, _ text: String) -> jstring? {
    // A four-byte UTF-8 sequence starts 0xF0 or above; JSON has escaped any NUL.
    guard text.utf8.contains(where: { $0 >= 0xF0 || $0 == 0 }) else {
      return text.withCString { env.pointee!.pointee.NewStringUTF(env, $0) }
    }
    var safe = ""
    safe.reserveCapacity(text.utf8.count)
    for scalar in text.unicodeScalars {
      if scalar.value > 0xFFFF {
        let v = scalar.value - 0x10000
        safe += String(format: "\\u%04x\\u%04x", 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF))
      } else if scalar.value == 0 {
        safe += "\\u0000"
      } else {
        safe.unicodeScalars.append(scalar)
      }
    }
    return safe.withCString { env.pointee!.pointee.NewStringUTF(env, $0) }
  }

  func json(_ object: Any) -> String {
    ControlJSON.data(object).map { String(decoding: $0, as: UTF8.self) } ?? "{}"
  }

  func parse(_ text: String) -> [String: Any] {
    (try? JSONSerialization.jsonObject(with: Data(text.utf8))) as? [String: Any] ?? [:]
  }

  // MARK: the calls

  /// `config` is JSON: cache (a directory), backend ("ort", the default, or
  /// "litert") and its device ("htp", "htp-whole", "gpu", "cpu" for
  /// onnxruntime; "gpu", "cpu", "npu" for LiteRT), keep_alive, keep_cpu_warm,
  /// listen, port, usb, preload, chip (Build.SOC_MODEL), firmware
  /// (Build.FINGERPRINT) and native_library_dir. Returns nil, or why the
  /// server could not start.
  @_cdecl("Java_io_zoompilot_jetlink_server_Native_start")
  public func nativeStart(_ env: Env, _ cls: jclass?, _ config: jstring?) -> jstring? {
    do {
      try Host.shared.start(parse(string(env, config)))
      return nil
    } catch {
      return jstring(env, String(describing: error))
    }
  }

  /// Stops the server and releases the engine.
  @_cdecl("Java_io_zoompilot_jetlink_server_Native_stop")
  public func nativeStop(_ env: Env, _ cls: jclass?) {
    Host.shared.stop()
  }

  /// Runs one control command, `{"cmd": "prepare", "sha256": ...}`, and returns
  /// its reply, `{"ok": true, ...}`. Blocks until the server replies.
  @_cdecl("Java_io_zoompilot_jetlink_server_Native_command")
  public func nativeCommand(_ env: Env, _ cls: jclass?, _ command: jstring?) -> jstring? {
    jstring(env, json(Host.shared.command(parse(string(env, command)))))
  }

  /// The app's state once its version is past `after`, or after `timeoutMs`;
  /// empty when nothing changed.
  @_cdecl("Java_io_zoompilot_jetlink_server_Native_snapshot")
  public func nativeSnapshot(_ env: Env, _ cls: jclass?, _ after: jlong, _ timeoutMs: jint) -> jstring? {
    let server = Host.shared.running?.server
    let snapshot = Host.shared.state.snapshot(after: Int(after), timeout: Double(timeoutMs) / 1000, port: server?.port.map { Int($0) }) {
      server?.recentStats(window: AppSnapshot.recentWindow)
    }
    return jstring(env, snapshot.map(json) ?? "")
  }

  /// Log lines numbered past `after`: `{"next": n, "lines": [...]}`.
  @_cdecl("Java_io_zoompilot_jetlink_server_Native_logs")
  public func nativeLogs(_ env: Env, _ cls: jclass?, _ after: jlong) -> jstring? {
    let (next, lines) = LogRing.shared.lines(after: Int(after))
    return jstring(env, json(["next": next, "lines": lines]))
  }

  /// The comma's gadget is open: `fd` from UsbDeviceConnection with the vendor
  /// interface claimed, and its bulk endpoints' addresses. Nil or an error.
  @_cdecl("Java_io_zoompilot_jetlink_server_Native_usbAttach")
  public func nativeUSBAttach(_ env: Env, _ cls: jclass?, _ fd: jint, _ inEndpoint: jint, _ outEndpoint: jint) -> jstring? {
    do {
      try Host.shared.gadget.attach(
        fd: fd, inEndpoint: UInt8(truncatingIfNeeded: inEndpoint), outEndpoint: UInt8(truncatingIfNeeded: outEndpoint))
      return nil
    } catch {
      return jstring(env, String(describing: error))
    }
  }

  /// The gadget is going. Returns once nothing uses the descriptor, so the app
  /// can close the connection after.
  @_cdecl("Java_io_zoompilot_jetlink_server_Native_usbDetach")
  public func nativeUSBDetach(_ env: Env, _ cls: jclass?) {
    Host.shared.gadget.detach()
  }

  /// "nominal", "fair", "serious" or "critical", from PowerManager, for the
  /// benchmark's reports.
  @_cdecl("Java_io_zoompilot_jetlink_server_Native_reportThermal")
  public func nativeReportThermal(_ env: Env, _ cls: jclass?, _ label: jstring?) {
    Host.shared.thermal = string(env, label)
  }

  /// The onnxruntime version, for Settings before the server has said it.
  @_cdecl("Java_io_zoompilot_jetlink_server_Native_runtimeVersion")
  public func nativeRuntimeVersion(_ env: Env, _ cls: jclass?) -> jstring? {
    jstring(env, OrtRuntime.version)
  }

  // MARK: the server

  /// The one server the app runs, and what it has said.
  final class Host: @unchecked Sendable {
    static let shared = Host()

    let state = AppSnapshot()
    /// The comma's descriptor, from the app's USB permission flow.
    let gadget = UsbfsGadget()
    private let lock = NSLock()
    private var embedded: EmbeddedServer?
    private var thermalLabel = "unknown"

    var running: EmbeddedServer? {
      lock.withLock { embedded }
    }

    var thermal: String {
      get { lock.withLock { thermalLabel } }
      set { lock.withLock { thermalLabel = newValue } }
    }

    func start(_ config: [String: Any]) throws {
      stop()
      if let dir = config["native_library_dir"] as? String, !dir.isEmpty {
        // Where the QNN provider finds the NPU's skel libraries: the app's own,
        // then the system's DSP directories.
        setenv("ADSP_LIBRARY_PATH", "\(dir);/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp", 1)
      }
      guard let cache = config["cache"] as? String, !cache.isEmpty else {
        throw HostFailure("no cache directory")
      }
      let backend = try Host.backend(config)
      let configuration = Server.Configuration(
        port: UInt16(clamping: (config["port"] as? NSNumber)?.intValue ?? Int(Wire.defaultPort)),
        cacheRoot: URL(fileURLWithPath: cache, isDirectory: true),
        preload: (config["preload"] as? Bool) ?? true,
        listen: (config["listen"] as? Bool) ?? true,
        usb: (config["usb"] as? Bool) ?? true,
        keepPlans: 2)

      // logcat too, where `adb logcat -s jetlink` finds it
      Log.sink = { level, category, message in
        _ = __android_log_write(level.logcatPriority, "jetlink", EmbeddedServer.logLine(level, category, message))
      }
      let server: EmbeddedServer
      do {
        server = try EmbeddedServer(
          configuration: configuration, backend: backend, gadget: gadget, hooks: ServerHooks(thermal: { Host.shared.thermal }))
        state.reset()
        let events = server.events
        let state = self.state
        Task.detached {
          for await event in events {
            state.apply(event)
          }
        }
        try server.start()
      } catch {
        Log.sink = nil
        throw error
      }
      lock.withLock { embedded = server }
      state.serverStarted()
    }

    /// The backend and device the config names. A config without a backend
    /// is onnxruntime's.
    static func backend(_ config: [String: Any]) throws -> any EngineBackend {
      let chip = config["chip"] as? String ?? ""
      let device = config["device"] as? String
      switch config["backend"] as? String ?? "ort" {
      case "litert":
        guard let profile = LiteRtProfile(rawValue: device ?? LiteRtProfile.gpu.rawValue) else {
          throw HostFailure("LiteRT has no device \(device ?? "")")
        }
        let libraries = (config["native_library_dir"] as? String).flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: $0, isDirectory: true) }
        let backend = LiteRtBackend(
          profile: profile, preparer: ONNXPreparer(), libraries: libraries, chip: chip, firmware: config["firmware"] as? String)
        // libLiteRt.so, its GPU accelerator and for the NPU Google Tensor's
        // libraries, from the app's own and the system's: a phone LiteRT
        // cannot run on says so now, not at the first build.
        try backend.open()
        return backend
      case "ort":
        guard let profile = OrtProfile(rawValue: device ?? OrtProfile.htp.rawValue), OrtProfile.available.contains(profile) else {
          throw HostFailure("onnxruntime has no device \(device ?? "")")
        }
        return OrtBackend(
          profile: profile, preparer: ONNXPreparer(), keepAlive: (config["keep_alive"] as? Bool) ?? true,
          keepCPUWarm: (config["keep_cpu_warm"] as? Bool) ?? false, chip: chip,
          qnn: try QnnPolicy(configuration: config))
      case let other:
        throw HostFailure("unknown backend \(other)")
      }
    }

    func stop() {
      let server = lock.withLock {
        defer { embedded = nil }
        return embedded
      }
      server?.stop(releasingEngine: true)
      Log.sink = nil
      if server != nil {
        state.serverStopped()
      }
    }

    func command(_ request: [String: Any]) -> Any {
      guard let server = running else {
        return ["ok": false, "error": "The server is not running."]
      }
      let command: ControlCommand
      do {
        command = try ControlCommand(object: request)
      } catch {
        return ["ok": false, "error": String(describing: error)]
      }
      let reply = try? blocking { await server.handle(command) }
      return reply.map { ControlEvent.reply($0).payload() } ?? NSNull()
    }
  }

  struct HostFailure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
  }

  extension Log.Level {
    /// ANDROID_LOG_DEBUG, INFO, WARN and ERROR.
    var logcatPriority: Int32 {
      switch self {
      case .debug: 3
      case .info: 4
      case .warning: 5
      case .error: 6
      }
    }
  }
#endif
