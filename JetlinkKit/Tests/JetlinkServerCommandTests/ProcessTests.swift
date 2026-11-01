#if os(macOS) || os(Linux)
  import Foundation
  import JetlinkKit
  import JetlinkServer
  import JetlinkTestSupport
  import Testing

  #if canImport(Glibc)
    import Glibc
  #endif

  /// The built binary beside this test bundle, or $JETLINK_SERVER_BIN.
  enum Binary {
    static let url: URL? = {
      if let named = ProcessInfo.processInfo.environment["JETLINK_SERVER_BIN"], !named.isEmpty {
        return URL(fileURLWithPath: named)
      }
      #if os(Linux)
        let directory = (try? FileManager.default.destinationOfSymbolicLink(atPath: "/proc/self/exe")).map {
          URL(fileURLWithPath: $0).deletingLastPathComponent()
        }
      #else
        let directory = Optional(Bundle(for: Marker.self).bundleURL.deletingLastPathComponent())
      #endif
      let url = directory?.appending(path: "jetlink-server")
      return url.flatMap { FileManager.default.isExecutableFile(atPath: $0.path) ? $0 : nil }
    }()

    private final class Marker {}

    /// Runs it to the end: (status, stdout, stderr).
    static func run(_ arguments: [String], binary: URL? = url, environment: [String: String] = [:]) throws -> (Int32, String, String) {
      let process = Process()
      process.executableURL = try #require(binary)
      process.arguments = arguments
      process.environment = ProcessInfo.processInfo.environment.merging(environment) { _, new in new }
      let (out, err) = (Pipe(), Pipe())
      process.standardOutput = out
      process.standardError = err
      try process.run()
      // Read before waiting: a full pipe would stall the child.
      let output = out.fileHandleForReading.readDataToEndOfFile()
      let errors = err.fileHandleForReading.readDataToEndOfFile()
      process.waitUntilExit()
      return (process.terminationStatus, String(decoding: output, as: UTF8.self), String(decoding: errors, as: UTF8.self))
    }
  }

  @Suite(.enabled(if: Binary.url != nil, "the jetlink-server product is built"))
  struct ProcessTests {
    @Test func versionIsOneLine() throws {
      let (status, out, _) = try Binary.run(["--version"])
      #expect(status == 0 && out == Pinned.productVersion + "\n")
    }

    @Test func usageMistakesExitOne() throws {
      let (status, _, err) = try Binary.run(["--bogus"])
      #expect(status == 1 && err.contains("--bogus"))
      #expect(try Binary.run(["models", "rm", "nothex", "--model", "--cache", NSTemporaryDirectory()]).0 == 1)
    }

    @Test func specIsWhatPythonDerives() throws {
      let (status, out, err) = try Binary.run(["spec", TinyModel.queued.path])
      #expect(status == 0, "\(err)")
      let printed = try #require(try JSONSerialization.jsonObject(with: Data(out.utf8)) as? NSDictionary)
      #expect(printed == (try TinyModel.spec(TinyModel.queued)) as NSDictionary)
    }

    /// The installer's GPU check: TensorRT or a non-zero exit.
    @Test func backendsSaysWhyAndExitsOnTheOneAskedFor() throws {
      let (status, out, _) = try Binary.run(["backends"])
      #expect(out.contains("trt: ") && out.contains("ort: "))
      #expect(status == (out.contains("usable: ") ? 0 : 1))
      let (trt, said, _) = try Binary.run(["backends", "--backend", "trt"])
      #expect(trt == (said.contains("trt: usable") ? 0 : 1))
    }

    @Test func buildThenBench() throws {
      let tmp = try TemporaryDirectory()
      let cpu = ["--backend", "ort", "--device", "cpu", "--cache", tmp.path]
      let (built, _, buildLog) = try Binary.run(["build", TinyModel.stateful.path] + cpu)
      #expect(built == 0, "\(buildLog)")
      #expect(buildLog.contains("built: {"))
      let (again, _, againLog) = try Binary.run(["build", TinyModel.stateful.path] + cpu)
      #expect(again == 0 && againLog.contains("already built: {"))
      let (benched, report, benchLog) = try Binary.run(["bench", "--seconds", "1"] + cpu)
      #expect(benched == 0, "\(benchLog)")
      #expect(report.hasPrefix("Jetlink benchmark") && report.contains("over 35 ms: "), "\(report)")
      #expect(!benchLog.contains("Jetlink benchmark"), "the report is on stdout only")
    }

    /// SIGTERM stops it cleanly, in order: the server, then the web page.
    @Test func sigtermStopsItCleanly() throws {
      let tmp = try TemporaryDirectory()
      let page = try TCPListener(host: "127.0.0.1", port: 0).port
      let log = try tmp.file("server.log", "")
      let handle = try FileHandle(forWritingTo: log)
      let process = Process()
      process.executableURL = try #require(Binary.url)
      process.arguments = [
        "--backend", "ort", "--device", "cpu", "--listen", "--host", "127.0.0.1", "--port", "0", "--cache", tmp.path, "--no-preload",
        "--status-port", "\(page)",
      ]
      process.standardOutput = handle
      process.standardError = handle
      try process.run()
      defer { if process.isRunning { process.terminate() } }

      // Said once the server listens and the page is up.
      let serving = eventually(timeout: 30) {
        !process.isRunning || String(decoding: (try? Data(contentsOf: log)) ?? Data(), as: UTF8.self).contains("web page on port")
      }
      guard serving, process.isRunning else {
        Issue.record("never listened: \(String(decoding: (try? Data(contentsOf: log)) ?? Data(), as: UTF8.self))")
        return
      }
      kill(process.processIdentifier, SIGTERM)
      let stopped = Date().addingTimeInterval(10)
      while process.isRunning && Date() < stopped { Thread.sleep(forTimeInterval: 0.05) }
      #expect(!process.isRunning)
      #expect(process.terminationReason == .exit && process.terminationStatus == 0)
      let text = String(decoding: try Data(contentsOf: log), as: UTF8.self)
      // The line install.sh waits for, said once, with no comma anywhere.
      let ready = try #require(text.range(of: "jetlink-server is serving"), "\(text)")
      #expect(text.components(separatedBy: "jetlink-server is serving").count == 2, "\(text)")
      let stopping = try #require(text.range(of: "stopping on SIGTERM"), "\(text)")
      #expect(ready.upperBound <= stopping.lowerBound)
      let server = try #require(text.range(of: "stopped the server"), "\(text)")
      let statusPage = try #require(text.range(of: "stopped the web page"), "\(text)")
      #expect(stopping.upperBound <= server.lowerBound)
      #expect(server.upperBound <= statusPage.lowerBound)
    }

    #if os(Linux)
      /// A SIGTERM during start-up waits for the handler, rather than ending
      /// the process: Linux holds it until the signal sources are registered.
      @Test func sigtermDuringStartUpStillStopsCleanly() throws {
        let tmp = try TemporaryDirectory()
        let page = try TCPListener(host: "127.0.0.1", port: 0).port
        let log = try tmp.file("server.log", "")
        let process = Process()
        process.executableURL = try #require(Binary.url)
        process.arguments = [
          "--backend", "ort", "--device", "cpu", "--listen", "--host", "127.0.0.1", "--port", "0", "--cache", tmp.path, "--no-preload",
          "--status-port", "\(page)",
        ]
        let handle = try FileHandle(forWritingTo: log)
        process.standardOutput = handle
        process.standardError = handle
        try process.run()
        defer { if process.isRunning { process.terminate() } }

        // At the serving line, before the web page and the handler are up.
        let serving = eventually(timeout: 30) {
          !process.isRunning || String(decoding: (try? Data(contentsOf: log)) ?? Data(), as: UTF8.self).contains("jetlink-server is serving")
        }
        guard serving, process.isRunning else {
          Issue.record("never served: \(String(decoding: (try? Data(contentsOf: log)) ?? Data(), as: UTF8.self))")
          return
        }
        kill(process.processIdentifier, SIGTERM)
        process.waitUntilExit()
        let text = String(decoding: try Data(contentsOf: log), as: UTF8.self)
        #expect(process.terminationReason == .exit && process.terminationStatus == 0, "\(text)")
        #expect(text.contains("stopping on SIGTERM") && text.contains("stopped the web page"), "\(text)")
      }
    #endif
  }
#endif
