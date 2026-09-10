import Foundation
import Testing

@testable import Jetlink

struct ServerConfigurationTests {
  private func configuration(backend: BackendChoice, transport: TransportChoice = .usb, port: Int = 5599, logLevel: String = "INFO") -> ServerConfiguration {
    ServerConfiguration(
      python: URL(filePath: "/Applications/Jetlink.app/Contents/Resources/python/bin/python3"),
      backend: backend,
      transport: transport,
      tcpPort: port,
      cacheDirectory: URL(filePath: "/Users/me/Library/Application Support/Jetlink/cache"),
      controlSocket: URL(filePath: "/var/folders/ab/T/jetlink-control.sock"),
      logLevel: logLevel,
      logFile: URL(filePath: "/Users/me/Library/Logs/Jetlink/server.log"))
  }

  private func value(after flag: String, in arguments: [String]) -> String? {
    guard let index = arguments.firstIndex(of: flag), index + 1 < arguments.count else { return nil }
    return arguments[index + 1]
  }

  @Test func theModuleComesFirst() {
    let arguments = ServerProcess.arguments(for: configuration(backend: .auto))
    #expect(arguments.prefix(2) == ["-m", "jetlink.server.main"])
  }

  @Test(arguments: [
    (BackendChoice.auto, "auto", String?.none),
    (BackendChoice.coreml, "ort", String?.some("coreml")),
    (BackendChoice.ane, "ort", String?.some("ane")),
    (BackendChoice.tinygrad, "tinygrad", String?.some("METAL")),
  ])
  func backendMapping(choice: BackendChoice, backend: String, device: String?) {
    let arguments = ServerProcess.arguments(for: configuration(backend: choice))
    #expect(value(after: "--backend", in: arguments) == backend)
    #expect(value(after: "--device", in: arguments) == device)
    if device == nil {
      #expect(!arguments.contains("--device"))
    }
  }

  @Test func usbTransportHasNoHostOrPort() {
    let arguments = ServerProcess.arguments(for: configuration(backend: .auto, transport: .usb))
    #expect(value(after: "--transport", in: arguments) == "usb")
    #expect(!arguments.contains("--host"))
    #expect(!arguments.contains("--port"))
  }

  @Test func tcpTransportCarriesHostAndPort() {
    let arguments = ServerProcess.arguments(for: configuration(backend: .auto, transport: .tcp, port: 5601))
    #expect(value(after: "--transport", in: arguments) == "tcp")
    #expect(value(after: "--host", in: arguments) == "0.0.0.0")
    #expect(value(after: "--port", in: arguments) == "5601")
  }

  @Test func pathsAndLogLevelAreForwarded() {
    let arguments = ServerProcess.arguments(for: configuration(backend: .coreml, logLevel: "DEBUG"))
    #expect(value(after: "--cache", in: arguments) == "/Users/me/Library/Application Support/Jetlink/cache")
    #expect(value(after: "--control-socket", in: arguments) == "/var/folders/ab/T/jetlink-control.sock")
    #expect(value(after: "--log-level", in: arguments) == "DEBUG")
    #expect(value(after: "--parent-pid", in: arguments) == String(ProcessInfo.processInfo.processIdentifier))
  }

  @Test func theEnvironmentIsBuiltFromScratch() {
    let environment = ServerProcess.environment(for: configuration(backend: .auto))
    #expect(environment["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin")
    #expect(environment["LANG"] == "en_US.UTF-8")
    #expect(environment["PYTHONNOUSERSITE"] == "1")
    #expect(environment["PYTHONDONTWRITEBYTECODE"] == "1")
    #expect(environment["PYTHONUNBUFFERED"] == "1")
    #expect(environment["PYTHONIOENCODING"] == "utf-8")
    #expect(environment["TMPDIR"]?.isEmpty == false)
  }

  @Test func noInterpreterOrLoaderVariablesLeakThrough() {
    for choice in BackendChoice.allCases {
      for transport in TransportChoice.allCases {
        let environment = ServerProcess.environment(for: configuration(backend: choice, transport: transport))
        #expect(environment["PYTHONPATH"] == nil)
        #expect(environment["PYTHONHOME"] == nil)
        #expect(environment["JETLINK_CACHE"] == nil)
        #expect(environment["VIRTUAL_ENV"] == nil)
        #expect(environment.keys.contains { $0.hasPrefix("DYLD_") } == false)
        let allowed: Set<String> = [
          "PATH", "LANG", "HOME", "USER", "TMPDIR",
          "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "PYTHONIOENCODING",
        ]
        #expect(Set(environment.keys).isSubset(of: allowed))
      }
    }
  }
}
