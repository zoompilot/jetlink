// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.

import Foundation
import JetlinkKit
import Observation
import os
import UIKit

/// What the engine is doing, as the host's `engine` and `progress` events say.
struct EngineStatus: Equatable {
  var state = "none"  // none | building | loading | ready | failed
  var sha256: String?
  var detail = ""
  var stage: String?
  var frac = 0.0
  var msg = ""
  var units: ComputeUnits?

  var isBusy: Bool { state == "building" || state == "loading" }
}

enum LinkState: Equatable {
  case stopped
  case waiting
  case connected(peer: String)
  case disconnected(detail: String)
}

enum DownloadState: Equatable {
  case running(bytes: Int, total: Int)
  case failed(String)
}

struct ModelOnDisk: Identifiable, Equatable {
  let sha256: String
  let bytes: Int
  var name: String?
  var id: String { sha256 }
}

/// The composition root: the engine host, the server, the model registry,
/// and everything the views show about them. One exists for the app's life.
@MainActor
@Observable
final class AppModel {
  let settings: AppSettings
  let logs: LogStore

  // What the views show.
  private(set) var serverRunning = false
  private(set) var serverError: String?
  private(set) var link: LinkState = .stopped
  private(set) var engine = EngineStatus()
  private(set) var stats: FrameStats.Summary?
  private(set) var addresses: [InterfaceAddress] = []
  private(set) var thermal: ProcessInfo.ThermalState = ProcessInfo.processInfo.thermalState
  private(set) var batteryLevel: Float = -1
  private(set) var batteryState: UIDevice.BatteryState = .unknown
  private(set) var catalog: [CatalogModel] = []
  private(set) var catalogError: String?
  private(set) var catalogLoading = false
  private(set) var downloads: [String: DownloadState] = [:]  // by ref
  private(set) var onDisk: [ModelOnDisk] = []
  private(set) var prepared: [String: ComputeUnits] = [:]  // sha256 -> units with an engine on disk
  private(set) var importing: Double?
  private(set) var importError: String?
  private(set) var shutdownNotice: String?
  private(set) var inForeground = true
  /// How much more memory iOS will let the app have right now.
  private(set) var memoryAvailable = 0
  private(set) var benchProgress: BenchmarkProgress?
  private(set) var benchReport: BenchmarkReport?
  private(set) var benchError: String?
  let cacheRoot: URL

  @ObservationIgnored let cache: EngineCache
  @ObservationIgnored let host: EngineHost
  @ObservationIgnored let registry: Registry
  @ObservationIgnored private var server: JetlinkServer?
  @ObservationIgnored private var ticker: Timer?
  @ObservationIgnored private var downloadTasks: [String: Task<Void, Never>] = [:]
  @ObservationIgnored private var observers: [NSObjectProtocol] = []
  @ObservationIgnored private let log = JLogger("jetlink.app")
  @ObservationIgnored private var launched = false
  @ObservationIgnored private var benchRun: BenchmarkRun?

  init() {
    let settings = AppSettings()
    let logs = LogStore()
    self.settings = settings
    self.logs = logs
    Log.shared.setSink(minimum: .info) { line in
      Task { @MainActor in logs.append(line) }
    }
    OrtRuntime.gpuKeepAlive = settings.gpuKeepAlive
    OrtRuntime.cpuKeepWarm = settings.cpuKeepWarm ? .on : .off

    let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
    var root = support.appendingPathComponent("Jetlink/cache", isDirectory: true)
    try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    // Gigabytes of models the network can give back have no place in a backup.
    var values = URLResourceValues()
    values.isExcludedFromBackup = true
    try? root.setResourceValues(values)
    cacheRoot = root
    do {
      cache = try EngineCache(root: root)
      registry = try Registry(cacheRoot: root)
    } catch {
      fatalError("cannot create the model cache at \(root.path): \(error)")
    }
    host = EngineHost(cache: cache, preference: settings.compute)
    host.subscribe { [weak self] kind, payload in
      Task { @MainActor in self?.handle(kind, payload) }
    }
    observeDevice()
    refreshDisk()
  }

  // MARK: lifecycle

  /// Called when the first window appears; later calls do nothing.
  func launch() {
    guard !launched else { return }
    launched = true
    log.info("Jetlink \(Bundle.main.shortVersion), onnxruntime \(OrtRuntime.version), \(chipName())")
    if settings.startOnLaunch { startServer() }
    startTicker()
    Task { await refreshCatalog() }
  }

  func startServer() {
    guard server == nil else { return }
    let s = JetlinkServer(host: host, port: UInt16(clamping: settings.port))
    do {
      try s.start()
      server = s
      serverRunning = true
      serverError = nil
      link = .waiting
    } catch {
      serverError = String(describing: error)
      log.error("the server could not start: \(error)")
    }
    updateIdleTimer()
  }

  func stopServer() {
    guard let s = server else { return }
    server = nil
    // stop() waits for the serving thread; keep the UI responsive meanwhile.
    Task.detached { s.stop() }
    serverRunning = false
    link = .stopped
    updateIdleTimer()
  }

  func restartServer() {
    guard let s = server else { return startServer() }
    server = nil
    serverRunning = false
    Task.detached { [weak self] in
      s.stop()
      await self?.startServer()
    }
  }

  func scenePhaseChanged(active: Bool, background: Bool) {
    inForeground = !background
    if background && serverRunning {
      log.warning(
        "Jetlink went to the background. iOS stops apps it has suspended, and a background app may not use the GPU; "
          + "keep Jetlink on screen while driving.")
    }
    if active {
      addresses = InterfaceAddress.current()
      refreshDisk()
    }
    updateIdleTimer()
  }

  private func updateIdleTimer() {
    UIApplication.shared.isIdleTimerDisabled = settings.keepScreenAwake && serverRunning
  }

  private func startTicker() {
    ticker?.invalidate()
    ticker = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
      Task { @MainActor in self?.tick() }
    }
  }

  private func tick() {
    stats = host.frameStats.summary(seconds: 5)
    memoryAvailable = os_proc_available_memory()
    let now = InterfaceAddress.current()
    if now != addresses { addresses = now }
  }

  private func observeDevice() {
    UIDevice.current.isBatteryMonitoringEnabled = true
    batteryLevel = UIDevice.current.batteryLevel
    batteryState = UIDevice.current.batteryState
    let nc = NotificationCenter.default
    observers.append(
      nc.addObserver(forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: .main) { [weak self] _ in
        Task { @MainActor in self?.thermalChanged() }
      })
    for name in [UIDevice.batteryLevelDidChangeNotification, UIDevice.batteryStateDidChangeNotification] {
      observers.append(
        nc.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
          Task { @MainActor in
            self?.batteryLevel = UIDevice.current.batteryLevel
            self?.batteryState = UIDevice.current.batteryState
          }
        })
    }
    observers.append(
      nc.addObserver(forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: .main) { [weak self] _ in
        Task { @MainActor in self?.log.warning("iOS reports memory pressure") }
      })
  }

  private func thermalChanged() {
    thermal = ProcessInfo.processInfo.thermalState
    if thermal == .serious || thermal == .critical {
      log.warning("the phone is \(thermal.label.lowercased()); iOS slows the chip down to cool it, and frames may miss 50 ms")
    }
  }

  // MARK: host events

  private func handle(_ kind: String, _ payload: JSON) {
    switch kind {
    case "engine":
      var e = EngineStatus()
      e.state = payload["state"]?.string ?? "none"
      e.sha256 = payload["sha256"]?.string
      e.detail = payload["detail"]?.string ?? ""
      e.stage = payload["stage"]?.string
      e.frac = payload["frac"]?.double ?? 0
      e.msg = payload["msg"]?.string ?? ""
      e.units = payload["units"]?.string.flatMap(ComputeUnits.init(rawValue:))
      engine = e
      if !e.isBusy { refreshDisk() }
    case "progress":
      engine.stage = payload["stage"]?.string
      engine.frac = payload["frac"]?.double ?? 0
      engine.msg = payload["msg"]?.string ?? ""
    case "link":
      switch payload["state"]?.string {
      case "connected": link = .connected(peer: payload["peer"]?.string ?? "")
      case "disconnected": link = .disconnected(detail: payload["detail"]?.string ?? "")
      default: link = serverRunning ? .waiting : .stopped
      }
      addresses = InterfaceAddress.current()
    case "shutdown_requested":
      shutdownNotice = "The comma shut down (\(payload["reason"]?.string ?? "no reason given")). You can close Jetlink."
    default:
      break
    }
  }

  // MARK: settings that act

  func setCompute(_ p: ComputePreference) {
    settings.compute = p
    host.setPreference(p)
    refreshDisk()
  }

  func setGPUKeepAlive(_ on: Bool) {
    settings.gpuKeepAlive = on
    OrtRuntime.gpuKeepAlive = on
  }

  func setCPUKeepWarm(_ on: Bool) {
    settings.cpuKeepWarm = on
    OrtRuntime.cpuKeepWarm = on ? .on : .off
  }

  func setKeepScreenAwake(_ on: Bool) {
    settings.keepScreenAwake = on
    updateIdleTimer()
  }

  func setPort(_ port: Int) {
    guard (1024...65535).contains(port), port != settings.port else { return }
    settings.port = port
    if serverRunning { restartServer() }
  }

  func dismissShutdownNotice() { shutdownNotice = nil }

  // MARK: models

  func name(for sha256: String?) -> String? {
    guard let sha256 else { return nil }
    return catalog.first { $0.sha256 == sha256 }?.name ?? registry.name(for: sha256)
  }

  func refreshCatalog(force: Bool = false) async {
    catalogLoading = true
    let registry = registry
    let result = await registry.catalog(refresh: force)
    catalog = result.models
    catalogError = result.error
    await registry.resolveMissing(result.models.filter { $0.sha256 == nil }.map(\.ref))
    catalog = await registry.catalog().models
    catalogLoading = false
    refreshDisk()
  }

  func refreshDisk() {
    onDisk = registry.downloaded().map { ModelOnDisk(sha256: $0.sha256, bytes: $0.bytes, name: name(for: $0.sha256)) }
    var p: [String: ComputeUnits] = [:]
    for sha in Set(onDisk.map(\.sha256) + catalog.compactMap(\.sha256)) where isSHA256(sha) {
      for units in [ComputeUnits.ane, .coreml] {
        if let e = try? cache.entry(sha, tag: host.backend(for: units).tag), e.exists { p[sha] = units }
      }
    }
    prepared = p
    addresses = InterfaceAddress.current()
  }

  func isDownloaded(_ sha256: String?) -> Bool {
    guard let sha256 else { return false }
    return onDisk.contains { $0.sha256 == sha256 }
  }

  func download(_ model: CatalogModel) {
    guard downloadTasks[model.ref] == nil else { return }
    downloads[model.ref] = .running(bytes: 0, total: model.bytes ?? 0)
    let registry = registry
    let ref = model.ref
    downloadTasks[ref] = Task { [weak self] in
      do {
        _ = try await registry.fetch(ref) { bytes, total in
          Task { @MainActor in self?.downloads[ref] = .running(bytes: bytes, total: total) }
        }
        self?.downloads[ref] = nil
      } catch is CancellationError {
        self?.downloads[ref] = nil
      } catch {
        self?.downloads[ref] = .failed(String(describing: error))
      }
      self?.downloadTasks[ref] = nil
      self?.refreshDisk()
    }
  }

  func cancelDownload(_ ref: String) {
    downloadTasks[ref]?.cancel()
  }

  /// Builds if needed, then loads the model and keeps it ready for the comma.
  func prepare(_ sha256: String) {
    guard let bytes = onDisk.first(where: { $0.sha256 == sha256 })?.bytes ?? catalog.first(where: { $0.sha256 == sha256 })?.bytes,
      let req = try? EngineRequest(sha256: sha256, nbytes: bytes, frameSkip: defaultFrameSkip)
    else { return }
    let host = host
    Task.detached { host.request(req, session: nil) }
  }

  func unload() {
    let host = host
    Task.detached { host.unload() }
  }

  /// Measures the Neural Engine against the GPU again for a model.
  func recalibrate(_ sha256: String) {
    if host.loadedSha() == sha256 { host.unload() }
    host.forgetChoice(sha256)
    try? registry.remove(sha256, artifacts: true, model: false)
    refreshDisk()
    prepare(sha256)
  }

  func delete(_ sha256: String, engines: Bool, model: Bool) {
    if host.loadedSha() == sha256 { host.unload() }
    if engines { host.forgetChoice(sha256) }
    try? registry.remove(sha256, artifacts: engines, model: model)
    refreshDisk()
  }

  func choice(_ sha256: String) -> EngineHost.Choice? { host.choice(sha256) }

  func importModel(from url: URL) {
    importing = 0
    importError = nil
    let registry = registry
    Task.detached { [weak self] in
      let scoped = url.startAccessingSecurityScopedResource()
      defer { if scoped { url.stopAccessingSecurityScopedResource() } }
      do {
        _ = try registry.importModel(url, name: nil) { frac in
          Task { @MainActor in self?.importing = frac }
        }
        await MainActor.run {
          self?.importing = nil
          self?.refreshDisk()
        }
      } catch {
        await MainActor.run {
          self?.importing = nil
          self?.importError = String(describing: error)
        }
      }
    }
  }

  var diskFree: Int { freeBytes(cacheRoot) }

  // MARK: benchmarking

  var linkConnected: Bool {
    if case .connected = link { return true }
    return false
  }

  var benchRunning: Bool { benchRun != nil }

  /// Runs the loaded model at 20 Hz for `seconds`, on a thread of its own.
  func startBenchmark(seconds: Double) {
    guard benchRun == nil else { return }
    let run = BenchmarkRun()
    benchRun = run
    benchReport = nil
    benchError = nil
    benchProgress = BenchmarkProgress(elapsed: 0, total: seconds, frames: 0, lastWindow: nil)
    let host = host
    // The serving thread's priority: a default-priority thread can land on an
    // efficiency core, which is not what the comma's frames get.
    let thread = Thread { [weak self] in
      let result = Result {
        try host.benchmark(seconds: seconds, run: run) { p in
          Task { @MainActor in self?.benchProgress = p }
        }
      }
      Task { @MainActor in
        guard let self else { return }
        self.benchRun = nil
        self.benchProgress = nil
        switch result {
        case .success(let r): self.benchReport = r
        case .failure(let e): self.benchError = String(describing: e)
        }
      }
    }
    thread.name = "jetlink-benchmark"
    thread.qualityOfService = .userInteractive
    thread.start()
  }

  func cancelBenchmark() { benchRun?.cancel() }

  /// The address the comma should use: a wired one if there is one.
  var serverAddress: String? { (addresses.first(where: \.isWired) ?? addresses.first)?.address }

  /// The loaded model's size, which bench_link.py needs with its hash.
  var loadedModelBytes: Int? {
    guard let sha = engine.sha256 else { return nil }
    return onDisk.first { $0.sha256 == sha }?.bytes ?? catalog.first { $0.sha256 == sha }?.bytes
      ?? (try? host.cache.entry(sha, tag: host.backend(for: engine.units ?? .ane).tag).meta())?["spec"]?["nbytes"]?.int
  }
}

extension ProcessInfo.ThermalState {
  var label: String {
    switch self {
    case .nominal: "Nominal"
    case .fair: "Fair"
    case .serious: "Serious"
    case .critical: "Critical"
    @unknown default: "Unknown"
    }
  }
}

extension Bundle {
  var shortVersion: String { (infoDictionary?["CFBundleShortVersionString"] as? String) ?? "0" }
}
