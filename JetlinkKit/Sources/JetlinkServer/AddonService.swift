// The add-on server: a fork's own models beside the driving model, each on its
// own interface of the comma's gadget. docs/addons.md. From Peter Nguyen's
// V-ASM and sign-vision servers (trung791997/openpilot).
import Foundation
import JetlinkKit

/// Where an add-on's interface is: sysfs and usbfs on Linux, IOKit on a Mac,
/// a fake in the tests. Each claims only its own interface, by class.
public protocol AddonSource: Sendable {
  /// Is the comma's gadget on the bus? Cheap enough to poll every second.
  func present() -> Bool
  /// Claims the add-on's interface and opens its bulk pair.
  func open() throws -> AddonPipes
}

/// An add-on interface's bulk pair, a byte stream each way.
public final class AddonPipes: @unchecked Sendable {
  let pipes: any BulkPipes

  init(_ pipes: any BulkPipes) {
    self.pipes = pipes
  }

  func close() {
    pipes.close()
  }
}

/// Serves every add-on --addons names: builds and loads their models on the
/// server's backend, then answers each one's interface on a thread of its own,
/// running a pass only in a gap the driving model leaves (`AddonGate`).
public final class AddonService: @unchecked Sendable {
  static let readTimeout: TimeInterval = 1
  static let writeTimeout: TimeInterval = 1
  static let helloTimeout: TimeInterval = 0.1
  /// The comma waits 50 ms for an answer: a request may wait for a gap, the
  /// end of the next frame most often, as long as the pass still fits after.
  static let answerBudget: TimeInterval = 0.040
  static let reportEvery: TimeInterval = 60

  private let configs: [AddonConfig]
  private let host: EngineHost
  private let backend: any EngineBackend
  private let cacheRoot: URL
  private let source: @Sendable ([UInt8]) -> any AddonSource
  private let log = ServerLog(category: "addons")
  private let lock = NSLock()
  private var stopped = false
  private var open: [AddonPipes] = []

  /// `backend` builds the add-on models: the host's own, or one profile of it
  /// that takes a whole model where the host's splits the driving model.
  public init(
    configs: [AddonConfig], host: EngineHost, backend: (any EngineBackend)? = nil, cacheRoot: URL,
    source: @escaping @Sendable ([UInt8]) -> any AddonSource
  ) {
    self.configs = configs
    self.host = host
    self.backend = backend ?? host.backend
    self.cacheRoot = cacheRoot
    self.source = source
  }

  public var gate: AddonGate { host.addonGate }

  /// Prepares the models on a thread of its own, then serves each add-on.
  public func start() {
    Thread.detachNewThread { [self] in
      for config in configs {
        var models: [UInt32: AddonModel] = [:]
        for file in config.models {
          do {
            let model = try AddonModel.prepare(file, backend: backend, root: cacheRoot, log: log, host: host)
            models[model.tag] = model
          } catch {
            log.error("\(config.name): cannot serve \(file.lastPathComponent): \(error)")
          }
        }
        guard !models.isEmpty else { continue }
        let addon = Served(config: config, models: models)
        Thread.detachNewThread { [self] in loop(addon) }
      }
    }
  }

  public func stop() {
    let pipes = lock.withLock {
      stopped = true
      return open
    }
    for pipe in pipes { pipe.pipes.abort() }
  }

  private var isStopped: Bool { lock.withLock { stopped } }

  final class Served: @unchecked Sendable {
    let config: AddonConfig
    let models: [UInt32: AddonModel]
    var passes = 0
    var yields = 0
    /// Answers of any kind, ever: whether a link carried anything.
    var answered = 0
    var failures = 0
    var lastReport = Date()

    init(config: AddonConfig, models: [UInt32: AddonModel]) {
      self.config = config
      self.models = models
    }
  }

  private func loop(_ addon: Served) {
    let source = source(addon.config.interfaceClass)
    let name = addon.config.name
    let triple = addon.config.interfaceClass.map { String(format: "%02x", $0) }.joined(separator: "/")
    log.info("\(name): serving \(addon.models.count) model(s) on interface class \(triple)")
    var lastError = ""
    var quiet = false
    while !isStopped {
      guard source.present() else {
        Thread.sleep(forTimeInterval: 1)
        continue
      }
      let pipes: AddonPipes
      do {
        pipes = try source.open()
      } catch {
        // the comma has not linked this add-on's interface, most often
        if "\(error)" != lastError { log.info("\(name): \(error)") }
        lastError = "\(error)"
        Thread.sleep(forTimeInterval: 2)
        continue
      }
      lastError = ""
      lock.withLock { open.append(pipes) }
      let before = addon.answered
      do {
        try serve(addon, pipes: pipes)
      } catch {
        // before the comma opens its end, every read fails at once: said once
        if addon.answered > before {
          log.info("\(name): link ended after \(addon.answered - before) answers: \(error)")
          quiet = false
        } else if !quiet {
          log.info("\(name): claimed its interface; waiting for the comma to serve it (\(error))")
          quiet = true
        }
      }
      lock.withLock { open.removeAll { $0 === pipes } }
      pipes.close()
      Thread.sleep(forTimeInterval: quiet ? 2 : 1)
    }
  }

  /// Answers requests until the link fails, a hello whenever none comes for
  /// a second. A malformed header cannot be skipped in a byte stream, so it
  /// ends the link, which the loop opens again.
  func serve(_ addon: Served, pipes: AddonPipes) throws {
    let tags = Array(addon.models.keys)
    var header = [UInt8](repeating: 0, count: AddonWire.headerBytes)
    var body = [UInt8](repeating: 0, count: AddonWire.maxRequest + 1)
    while !isStopped {
      let first = try header.withUnsafeMutableBytes { try pipes.pipes.read(into: $0.baseAddress!, count: $0.count, timeout: Self.readTimeout) }
      if first == 0 {
        let hello = AddonWire.hello(tags: tags, yielding: gate.yielding)
        _ = try hello.withUnsafeBytes { try pipes.pipes.write(from: $0.baseAddress!, count: $0.count, timeout: Self.helloTimeout) }
        report(addon)
        continue
      }
      try readFully(pipes, &header, from: first, to: header.count)
      let request = try header.withUnsafeBytes { try AddonWire.request(header: $0) }
      try readFully(pipes, &body, from: 0, to: request.bodyBytes)
      let answer = body.withUnsafeBytes { handle(addon, request, image: UnsafeRawBufferPointer(rebasing: $0[0..<request.nbytes])) }
      try writeFully(pipes, answer)
      addon.answered += 1
    }
  }

  func handle(_ addon: Served, _ request: AddonWire.Request, image: UnsafeRawBufferPointer) -> [UInt8] {
    let started = DispatchTime.now().uptimeNanoseconds
    guard let model = addon.models[request.tag], model.shape == request.shape else {
      return AddonWire.answer(seq: request.seq, tag: request.tag, status: .unknownModel)
    }
    guard gate.gap(cost: model.costNs, wait: max(0, Self.answerBudget - Double(model.costNs) / 1e9)) else {
      addon.yields += 1
      return AddonWire.answer(seq: request.seq, tag: request.tag, status: .yield)
    }
    host.lock.lock()
    defer { host.lock.unlock() }
    do {
      let floats = try model.run(request, image: image)
      addon.passes += 1
      let us = UInt32(min(UInt64(UInt32.max), (DispatchTime.now().uptimeNanoseconds - started) / 1000))
      return AddonWire.answer(seq: request.seq, tag: request.tag, status: .ok, serverUs: us, floats: floats)
    } catch {
      addon.failures += 1
      if addon.failures == 1 || addon.failures % 100 == 0 { log.warning("\(addon.config.name): \(model.file) failed: \(error)") }
      return AddonWire.answer(seq: request.seq, tag: request.tag, status: .inferFailed)
    }
  }

  private func report(_ addon: Served) {
    guard Date().timeIntervalSince(addon.lastReport) >= Self.reportEvery, addon.passes + addon.yields + addon.failures > 0 else { return }
    let costs = addon.models.values.map { "\($0.file) \(String(format: "%.1f", Double($0.costNs) / 1e6)) ms" }.joined(separator: ", ")
    log.info("\(addon.config.name): \(addon.passes) passes, \(addon.yields) yielded, \(addon.failures) failed; \(costs)")
    addon.passes = 0
    addon.yields = 0
    addon.failures = 0
    addon.lastReport = Date()
  }

  private func readFully(_ pipes: AddonPipes, _ buffer: inout [UInt8], from start: Int, to end: Int) throws {
    var have = start
    let deadline = Date().addingTimeInterval(Self.readTimeout)
    while have < end {
      let got = try buffer.withUnsafeMutableBytes {
        try pipes.pipes.read(into: $0.baseAddress! + have, count: end - have, timeout: Self.readTimeout)
      }
      have += got
      if got == 0 && Date() >= deadline { throw LinkError.closed("a request stopped part way, \(have) of \(end) bytes") }
    }
  }

  private func writeFully(_ pipes: AddonPipes, _ bytes: [UInt8]) throws {
    var sent = 0
    let deadline = Date().addingTimeInterval(Self.writeTimeout)
    while sent < bytes.count {
      let wrote = try bytes.withUnsafeBytes {
        try pipes.pipes.write(from: $0.baseAddress! + sent, count: bytes.count - sent, timeout: Self.writeTimeout)
      }
      sent += wrote
      if wrote == 0 && Date() >= deadline { throw LinkError.closed("the comma stopped reading, \(sent) of \(bytes.count) bytes") }
    }
  }
}
