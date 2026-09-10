import Foundation
import os

enum BackendChoice: String, CaseIterable, Codable, Sendable {
  case auto, coreml, ane, tinygrad

  /// What `--backend` gets.
  var backendArgument: String {
    switch self {
    case .auto: return "auto"
    case .coreml, .ane: return "ort"
    case .tinygrad: return "tinygrad"
    }
  }

  /// What `--device` gets, or nil when the flag is left off.
  var deviceArgument: String? {
    switch self {
    case .auto: return nil
    case .coreml: return "coreml"
    case .ane: return "ane"
    case .tinygrad: return "METAL"
    }
  }

  /// What the choice comes to on a Mac, before the server has said so itself.
  var title: String {
    switch self {
    case .auto, .coreml: return "CoreML on the GPU"
    case .ane: return "CoreML with the Neural Engine"
    case .tinygrad: return "tinygrad on Metal"
    }
  }
}

enum TransportChoice: String, CaseIterable, Codable, Sendable {
  case usb, tcp
}

struct ServerConfiguration: Sendable, Equatable {
  let python: URL
  let backend: BackendChoice
  let transport: TransportChoice
  let tcpPort: Int
  let cacheDirectory: URL
  let controlSocket: URL
  let logLevel: String
  let logFile: URL

  init(python: URL, backend: BackendChoice, transport: TransportChoice, tcpPort: Int, cacheDirectory: URL, controlSocket: URL, logLevel: String, logFile: URL) {
    self.python = python
    self.backend = backend
    self.transport = transport
    self.tcpPort = tcpPort
    self.cacheDirectory = cacheDirectory
    self.controlSocket = controlSocket
    self.logLevel = logLevel
    self.logFile = logFile
  }
}

@MainActor
final class ServerProcess {
  private var process: Process?
  private let log = Logger(subsystem: "io.zoompilot.jetlink", category: "process")

  /// One line of the server's merged stdout and stderr, without the newline.
  var onLine: (@Sendable (String) -> Void)?
  /// The exit status and reason, delivered once per run.
  var onExit: (@Sendable (Int32, Process.TerminationReason) -> Void)?

  var isRunning: Bool { process?.isRunning ?? false }
  var processIdentifier: Int32? { process?.processIdentifier }

  // MARK: pure argument and environment building

  nonisolated static func arguments(for configuration: ServerConfiguration) -> [String] {
    var out = ["-m", "jetlink.server.main"]
    out += ["--backend", configuration.backend.backendArgument]
    if let device = configuration.backend.deviceArgument {
      out += ["--device", device]
    }
    out += ["--transport", configuration.transport.rawValue]
    if configuration.transport == .tcp {
      out += ["--host", "0.0.0.0", "--port", String(configuration.tcpPort)]
    }
    out += ["--cache", configuration.cacheDirectory.path(percentEncoded: false)]
    out += ["--control-socket", configuration.controlSocket.path(percentEncoded: false)]
    out += ["--parent-pid", String(ProcessInfo.processInfo.processIdentifier)]
    out += ["--log-level", configuration.logLevel]
    return out
  }

  nonisolated static func environment(for configuration: ServerConfiguration) -> [String: String] {
    let inherited = ProcessInfo.processInfo.environment
    var out: [String: String] = [
      "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
      "LANG": "en_US.UTF-8",
      "PYTHONNOUSERSITE": "1",
      "PYTHONDONTWRITEBYTECODE": "1",
      "PYTHONUNBUFFERED": "1",
      "PYTHONIOENCODING": "utf-8",
    ]
    if let home = inherited["HOME"] { out["HOME"] = home }
    if let user = inherited["USER"] { out["USER"] = user }
    out["TMPDIR"] = inherited["TMPDIR"] ?? NSTemporaryDirectory()
    return out
  }

  // MARK: running

  func start(_ configuration: ServerConfiguration) throws {
    if isRunning { return }
    try FileManager.default.createDirectory(at: configuration.cacheDirectory, withIntermediateDirectories: true)

    let process = Process()
    process.executableURL = configuration.python
    process.arguments = ServerProcess.arguments(for: configuration)
    process.environment = ServerProcess.environment(for: configuration)
    process.currentDirectoryURL = configuration.cacheDirectory

    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe
    process.standardInput = FileHandle.nullDevice

    let sink = LineSink(deliver: onLine)
    pipe.fileHandleForReading.readabilityHandler = { handle in
      let data = handle.availableData
      if data.isEmpty {
        sink.flush()
        handle.readabilityHandler = nil
        return
      }
      sink.ingest(data)
    }

    let exitHandler = onExit
    process.terminationHandler = { finished in
      if finished.terminationReason == .uncaughtSignal {
        // The server leads its own process group (--parent-pid), so this
        // takes the ORT worker down instead of leaving it orphaned with the
        // engine half built. ESRCH when there is no such group, harmless.
        kill(-finished.processIdentifier, SIGKILL)
      }
      exitHandler?(finished.terminationStatus, finished.terminationReason)
    }

    try process.run()
    self.process = process
    log.info("started the server, pid \(process.processIdentifier)")
  }

  /// SIGINT, then SIGTERM after 10 s, then SIGKILL after another 5 s.
  func stop() async {
    guard let process, process.isRunning else {
      self.process = nil
      return
    }
    let pid = process.processIdentifier
    process.interrupt()
    if await waitForExit(process, seconds: 10) {
      log.info("server \(pid) exited after SIGINT")
      self.process = nil
      return
    }
    log.notice("server \(pid) ignored SIGINT, sending SIGTERM")
    process.terminate()
    if await waitForExit(process, seconds: 5) {
      log.info("server \(pid) exited after SIGTERM")
      self.process = nil
      return
    }
    log.error("server \(pid) ignored SIGTERM, sending SIGKILL")
    kill(pid, SIGKILL)
    kill(-pid, SIGKILL)
    _ = await waitForExit(process, seconds: 5)
    self.process = nil
  }

  private func waitForExit(_ process: Process, seconds: Int) async -> Bool {
    let steps = seconds * 10
    for _ in 0..<steps {
      if !process.isRunning { return true }
      try? await Task.sleep(for: .milliseconds(100))
    }
    return !process.isRunning
  }
}

/// Splits an arbitrary byte stream into lines. The pipe's readability handler
/// runs on a private queue, so the partial line is only touched under `lock`.
private final class LineSink: @unchecked Sendable {
  private let lock = NSLock()
  private var partial = Data()
  private let deliver: (@Sendable (String) -> Void)?

  init(deliver: (@Sendable (String) -> Void)?) {
    self.deliver = deliver
  }

  func ingest(_ data: Data) {
    let lines: [String] = lock.withLock {
      partial.append(data)
      var out: [String] = []
      while let index = partial.firstIndex(of: 0x0A) {
        let slice = partial[partial.startIndex..<index]
        partial = partial[partial.index(after: index)...]
        out.append(String(decoding: slice, as: UTF8.self))
      }
      return out
    }
    for line in lines { deliver?(line) }
  }

  func flush() {
    let line: String? = lock.withLock {
      guard !partial.isEmpty else { return nil }
      let text = String(decoding: partial, as: UTF8.self)
      partial = Data()
      return text
    }
    if let line { deliver?(line) }
  }
}
