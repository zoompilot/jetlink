import Foundation
import JetlinkKit
import JetlinkORT
import JetlinkServer
import JetlinkUI
import SwiftUI
import Testing

@testable import Jetlink

/// The Swift server the app runs: what a stored setting means, and what the
/// server is asked to be.
struct ServerStoreTests {
  private let cache = URL(filePath: "/Users/me/Library/Application Support/Jetlink/cache")

  @MainActor @Test func aRemovedOrOlderBackendReadsAsAutomatic() throws {
    let suite = "io.zoompilot.jetlink.tests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    for stored in ["tinygrad", "", "nonsense"] {
      defaults.set(stored, forKey: AppSettings.Key.backend)
      #expect(AppSettings(defaults: defaults).backend == .auto, "stored \(stored)")
    }
    defaults.set("coreml", forKey: AppSettings.Key.backend)
    #expect(AppSettings(defaults: defaults).backend == .coreml)
  }

  /// Keeping a closed Mac awake is opt-in.
  @MainActor @Test func lidClosedKeepAwakeIsOffUntilChosen() throws {
    let suite = "io.zoompilot.jetlink.tests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let settings = AppSettings(defaults: defaults)
    #expect(!settings.keepAwakeLidClosed)
    settings.keepAwakeLidClosed = true
    #expect(AppSettings(defaults: defaults).keepAwakeLidClosed)
  }

  /// The scheme's test action passes -startServerOnLaunch NO, so the app
  /// hosting these tests opens no USB link and loads no engine.
  @MainActor @Test func theTestHostStartsNoServer() {
    #expect(!AppSettings().startServerOnLaunch)
  }

  @Test func usbServesTheGadgetAndOpensNoPort() {
    let configuration = ServerStore.configuration(transport: .usb, tcpPort: 5599, cacheDirectory: cache)
    #expect(configuration.usb)
    #expect(!configuration.listen)
    #expect(configuration.cacheRoot == cache)
  }

  @Test func tcpListensOnThePortAndLeavesUSBAlone() {
    let configuration = ServerStore.configuration(transport: .tcp, tcpPort: 5601, cacheDirectory: cache)
    #expect(!configuration.usb)
    #expect(configuration.listen)
    #expect(configuration.port == 5601)
  }

  @MainActor @Test func tcpIsOnlyTheDeveloperSwitch() throws {
    let suite = "io.zoompilot.jetlink.tests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    // What the removed Connection picker stored stays behind and means USB.
    defaults.set("tcp", forKey: "transport")
    #expect(AppSettings(defaults: defaults).transport == .usb)
    defaults.set("tcp", forKey: AppSettings.Key.developerTransport)
    #expect(AppSettings(defaults: defaults).transport == .tcp)
  }

  @Test(arguments: [
    ("Apple M1", OrtProfile.ane), ("Apple M1 Pro", .ane), ("Apple M1 Max", .ane), ("Apple M1 Ultra", .ane),
    ("Apple M2", .aneBoth), ("Apple M2 Pro", .aneBoth), ("Apple M5 Max", .aneBoth), ("Apple M10", .aneBoth),
  ])
  func automaticByChip(chip: String, profile: OrtProfile) {
    #expect(BackendChoice.automatic(chip: chip) == profile)
  }

  @Test(arguments: [
    (BackendChoice.auto, BackendChoice.automatic(chip: HostChip.name())),
    (BackendChoice.ane, OrtProfile.ane),
    (BackendChoice.aneBoth, OrtProfile.aneBoth),
    (BackendChoice.coreml, OrtProfile.coreml),
  ])
  func backendMapping(choice: BackendChoice, profile: OrtProfile) {
    let backend = ServerStore.backend(for: choice)
    #expect(backend.profile == profile)
    #expect(backend.keepAlive && backend.keepCPUWarm)
    let quiet = ServerStore.backend(for: choice, keepGPUAwake: false, keepCPUWarm: false)
    #expect(!quiet.keepAlive && !quiet.keepCPUWarm)
  }

  @Test func logLinesReadLikePythons() throws {
    var parts = DateComponents()
    (parts.year, parts.month, parts.day, parts.hour, parts.minute, parts.second, parts.nanosecond) = (2026, 9, 27, 13, 4, 5, 123_000_000)
    let date = try #require(Calendar.current.date(from: parts))
    #expect(EmbeddedServer.logLine(.warning, "server", "hello", at: date) == "2026-09-27 13:04:05,123 WARNING jetlink.server: hello")
    #expect(EmbeddedServer.logLine(.info, "usb", "x", at: date) == "2026-09-27 13:04:05,123 INFO    jetlink.usb: x")
    #expect(LogTone.color(for: EmbeddedServer.logLine(.error, "session", "bad", at: date)) == .red)
  }

  @MainActor @Test func benchmarkEventsReachTheStore() {
    let store = ServerStore.preview(link: .waiting, engine: .none)
    let stats = BenchmarkStats(mean: 30, p50: 30, p90: 31, p99: 33, max: 36)
    let event = BenchmarkEvent(state: "running", elapsed: 10, total: 60, frames: 200, frame: stats, report: nil, detail: "")
    store.apply(.benchmark(event))
    #expect(store.state.benchmark == event)
  }

  @MainActor @Test func aStoreStartsServesAndStops() async throws {
    let suite = "io.zoompilot.jetlink.tests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let cache = FileManager.default.temporaryDirectory.appending(path: "jetlink-store-\(UUID().uuidString)", directoryHint: .isDirectory)
    defer { try? FileManager.default.removeItem(at: cache) }
    defaults.set("tcp", forKey: AppSettings.Key.developerTransport)
    defaults.set(Int.random(in: 50_000..<60_000), forKey: AppSettings.Key.tcpPort)
    let settings = AppSettings(defaults: defaults)
    settings.cacheDirectory = cache
    let store = ServerStore(settings: settings, logs: LogBuffer(), logFile: nil)
    try await store.startIfNeeded()
    #expect(store.runState == .serving)
    #expect(store.info?.port == settings.tcpPort)
    let transport = try TCPTransport.connect(host: "127.0.0.1", port: UInt16(settings.tcpPort), timeout: 2)
    transport.setReceiveTimeout(5)
    try transport.sendJSON(.helloReq, seq: 1, ["client": ["name": "test"]])
    #expect(try transport.recv().msgType == Wire.Msg.helloResp.rawValue)
    try transport.send(.ping, seq: 2)
    #expect(try transport.recv().msgType == Wire.Msg.pong.rawValue)
    transport.close()
    await store.stopAndWait()
    #expect(store.runState == .stopped)
  }
}
