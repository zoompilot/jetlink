// The add-on server: a fork's own models beside the driving model, each on its
// own interface of the comma's gadget. docs/addons.md. From Peter Nguyen's
// V-ASM and sign-vision servers (trung791997/openpilot).
import Foundation
import JetlinkKit
import JetlinkRegistry

/// One add-on model, built and loaded on the server's backend: a single
/// [1, C, H, W] image input and any outputs, answered as f32 in name order.
final class AddonModel: @unchecked Sendable {
  let file: String
  let sha256: String
  let tag: UInt32
  let engine: any Engine
  let input: TensorSpec
  /// [C, H, W], as a request names it.
  let shape: [Int]
  let outputs: [TensorSpec]
  let outputCount: Int
  /// How long a pass holds the engine, staging its input and copying its
  /// outputs included, for fitting it between driving frames: measured once
  /// when it is loaded, then up at once with a slower pass, down slowly.
  private(set) var costNs: UInt64 = 5_000_000

  init(file: String, sha256: String, engine: any Engine) throws {
    guard let tag = AddonWire.tag(sha256: sha256) else { throw HostError.failed("\(file): no sha256") }
    guard engine.inputs.count == 1, let input = engine.inputs.values.first, input.shape.count == 4, input.shape[0] == 1,
      [.float, .float16, .uint8].contains(input.type)
    else {
      throw HostError.failed("\(file): an add-on model takes one [1, C, H, W] float or uint8 input, not \(Array(engine.inputs.values))")
    }
    let outputs = engine.outputs.values.sorted { $0.name < $1.name }
    guard !outputs.isEmpty, outputs.allSatisfy({ [.float, .float16].contains($0.type) }) else {
      throw HostError.failed("\(file): an add-on model's outputs are float, not \(outputs.map(\.type.name))")
    }
    let outputCount = outputs.reduce(0) { $0 + $1.count }
    try Self.fitsWire(file, input: input.shape.dropFirst().reduce(1, *), outputs: outputCount)
    self.file = file
    self.sha256 = sha256
    self.tag = tag
    self.engine = engine
    self.input = input
    shape = Array(input.shape.dropFirst())
    self.outputs = outputs
    self.outputCount = outputCount
  }

  /// Throws unless a request (its u8 image, the smallest the comma sends) and
  /// an answer (every output as f32) fit the wire's limits, saying which does not.
  static func fitsWire(_ file: String, input elements: Int, outputs floats: Int) throws {
    if AddonWire.headerBytes + elements >= AddonWire.maxRequest {
      throw HostError.failed("\(file): its input is \(elements) bytes even as u8, over the add-on wire's \(AddonWire.maxRequest >> 20) MiB request")
    }
    if AddonWire.headerBytes + 4 * floats > AddonWire.maxAnswer {
      throw HostError.failed("\(file): its outputs are \(floats) floats, \(4 * floats) bytes, over the add-on wire's \(AddonWire.maxAnswer >> 20) MiB answer")
    }
  }

  /// Stages `image` (the request's body, its pad byte not counted), runs,
  /// and returns every output. The caller holds the GPU.
  func run(_ request: AddonWire.Request, image: UnsafeRawBufferPointer) throws -> [Float] {
    let started = DispatchTime.now().uptimeNanoseconds
    guard let target = engine.hostInput(input.name) else { throw HostError.failed("\(file): no host input") }
    let count = input.count
    switch (request.encoding, input.type) {
    case (.u8, .uint8):
      target.copyMemory(from: image.baseAddress!, byteCount: count)
    case (.u8, .float):
      let out = target.assumingMemoryBound(to: Float.self)
      for i in 0..<count { out[i] = Float(image[i]) / 255 }
    case (.u8, .float16):
      let out = target.assumingMemoryBound(to: Float16.self)
      for i in 0..<count { out[i] = Float16(Float(image[i]) / 255) }
    case (.f32, .float):
      target.copyMemory(from: image.baseAddress!, byteCount: count * 4)
    case (.f32, .float16):
      let out = target.assumingMemoryBound(to: Float16.self)
      for i in 0..<count { out[i] = Float16(image.loadUnaligned(fromByteOffset: i * 4, as: Float.self)) }
    default:
      throw HostError.failed("\(file) takes \(input.type.name), not \(request.encoding)")
    }
    try engine.run()
    var floats: [Float] = []
    floats.reserveCapacity(outputCount)
    for spec in outputs {
      guard let source = engine.output(spec.name) else { throw HostError.failed("\(file): no output \(spec.name)") }
      if spec.type == .float {
        floats.append(contentsOf: UnsafeBufferPointer(start: source.assumingMemoryBound(to: Float.self), count: spec.count))
      } else {
        floats.append(contentsOf: UnsafeBufferPointer(start: source.assumingMemoryBound(to: Float16.self), count: spec.count).map(Float.init))
      }
    }
    let took = DispatchTime.now().uptimeNanoseconds - started
    costNs = took > costNs ? took : (costNs * 7 + took) / 8
    return floats
  }

  /// One pass of a blank u8 image, the dearer staging, to measure what a pass
  /// costs before the first request. The caller holds the GPU.
  func calibrate() throws {
    let count = shape.reduce(1, *)
    let blank = [UInt8](repeating: 0, count: count)
    let request = AddonWire.Request(seq: 0, tag: tag, shape: shape, encoding: .u8, nbytes: count)
    let started = DispatchTime.now().uptimeNanoseconds
    _ = try blank.withUnsafeBytes { try run(request, image: $0) }
    costNs = DispatchTime.now().uptimeNanoseconds - started
  }

  /// The add-on artifacts for `backend`: `root`/addons, a cache of its own, so
  /// add-ons and driving models never prune each other.
  static func store(root: URL, backend: any EngineBackend) -> EngineCache {
    EngineCache(
      layout: CacheLayout(root: root.appending(path: "addons", directoryHint: .isDirectory)), tag: backend.tag(), suffix: backend.suffix,
      backend: backend.name)
  }

  /// Builds `file`'s artifact for `backend` in the add-on store under `root`
  /// unless it is there, loads it, and warms it. With `host`, after the driving
  /// model's own build or load and never beside one, and warms under its lock.
  /// `interface` goes in the sidecar, so the next start serves it there.
  static func prepare(
    _ file: URL, backend: any EngineBackend, root: URL, log: ServerLog, host: EngineHost? = nil, sha256 known: String? = nil,
    name: String? = nil, interface: [UInt8]? = nil, port: UInt16? = nil, report progress: ProgressFn? = nil
  ) throws -> AddonModel {
    if let host {
      while host.drivingBusy { Thread.sleep(forTimeInterval: 1) }
      host.buildSlot.lock()
    }
    defer { host?.buildSlot.unlock() }
    let sha256 = try known ?? Files.sha256(of: file)
    let name = name ?? file.lastPathComponent
    let entry = try store(root: root, backend: backend).entry(sha256)
    let report: ProgressFn = { stage, frac, message in
      if !message.isEmpty { log.debug("\(name): \(stage) \(message)") }
      progress?(stage, frac, message)
    }
    var meta: [String: Any] = ["addon": name, "spec": ["sha256": sha256]]
    if let interface { meta["interface"] = AddonWire.triple(interface) }
    if let port { meta["port"] = Int(port) }
    func build() throws {
      log.info("building \(name) for \(backend.name)")
      try backend.build(model: file, artifact: entry.path, report: report, metaExtra: meta)
    }
    if !entry.exists { try build() }
    let engine: any Engine
    do {
      engine = try backend.load(artifact: entry.path, report: report)
    } catch let invalid as ArtifactInvalid {
      log.warning("rebuilding \(entry.path.lastPathComponent): \(invalid.description)")
      entry.remove()
      try build()
      engine = try backend.load(artifact: entry.path, report: report)
    }
    do {
      let model = try AddonModel(file: name, sha256: sha256, engine: engine)
      progress?("warm", 0, "warming up")
      let warmed = try underLock(host) { () throws -> String in
        let warmed = try engine.warm()
        try model.calibrate()
        return "\(warmed)"
      }
      log.info(
        "\(warmed); \(name) takes \(model.shape), gives \(model.outputs.map { "\($0.name)\($0.shape)" }), a pass \(String(format: "%.1f", Double(model.costNs) / 1e6)) ms"
      )
      return model
    } catch {
      engine.close()
      throw error
    }
  }

  private static func underLock<T>(_ host: EngineHost?, _ body: () throws -> T) rethrows -> T {
    guard let host else { return try body() }
    host.lock.lock()
    defer { host.lock.unlock() }
    return try body()
  }
}
