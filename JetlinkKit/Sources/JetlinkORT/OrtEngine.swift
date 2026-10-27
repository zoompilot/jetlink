import Foundation
import JetlinkServer

/// One session of a chain: its model file, the execution provider to run it
/// on, and that provider's options.
public struct SessionPlan: Sendable, Equatable {
  public let model: URL
  /// onnxruntime's name for the provider, "CoreML" or "QNN"; nil for the CPU
  /// provider alone.
  public let provider: String?
  public let options: [String: String]
  /// Per-session configuration, including an explicit fallback contract.
  public let config: [String: String]
  /// The CPU provider's intra-op pool: 1 where an accelerator does the work.
  public let threads: Int
  /// The session as the log and the hello name it: "CoreML(CPUAndGPU)",
  /// "QNN(htp)", "CPU".
  public let label: String
  /// Runs on the GPU, so the Metal keep-alive helps it.
  public let usesGPU: Bool
  /// Runs on the Neural Engine or an NPU, so the CPU keep-warm helps it.
  public let usesNeuralEngine: Bool

  public init(
    model: URL, provider: String?, options: [String: String] = [:], config: [String: String] = [:], threads: Int = 1, label: String,
    usesGPU: Bool = false, usesNeuralEngine: Bool = false
  ) {
    self.model = model
    self.provider = provider
    self.options = options
    self.config = config
    self.threads = threads
    self.label = label
    self.usesGPU = usesGPU
    self.usesNeuralEngine = usesNeuralEngine
  }

  /// A CoreML session on `computeUnits` ("CPUAndNeuralEngine", "CPUAndGPU"
  /// or "ALL"), with CoreML's compiled model kept in `cacheDirectory` so a
  /// load does not compile again. The provider options are the ones the
  /// Python backend passes (backends/ort/__init__.py).
  public init(model: URL, computeUnits: String, cacheDirectory: URL?) {
    let neuralEngine = computeUnits == "CPUAndNeuralEngine" || computeUnits == "ALL"
    var options = [
      "ModelFormat": "MLProgram",
      "MLComputeUnits": computeUnits,
    ]
    if neuralEngine {
      // Apple's hint for a model that is predicted many times. Only where
      // the Neural Engine is in play: on the GPU it changed nothing.
      options["SpecializationStrategy"] = "FastPrediction"
    }
    if let cacheDirectory {
      options["ModelCacheDirectory"] = cacheDirectory.path
    }
    self.init(
      model: model, provider: "CoreML", options: options, label: "CoreML(\(computeUnits))",
      usesGPU: computeUnits == "CPUAndGPU" || computeUnits == "ALL", usesNeuralEngine: neuralEngine)
  }
}

/// A loaded model: a chain of onnxruntime sessions run back to back over
/// buffers the engine owns, the Swift form of the Python ort worker.
///
/// The host writes a frame's inputs into `hostInput(_:)`, calls `run()`, and
/// reads `output(_:)`. The trunk's outputs feed the policy by name without
/// leaving the engine, and after `loopState` a stateful graph's next_state_
/// outputs feed its state_ inputs on the next run by swapping two buffers:
/// the 12 MB of queues never cross to the host.
public final class OrtEngine: EngineCore, @unchecked Sendable {
  public let device: String
  public let providers: [String]
  /// Whether the CPU is kept warm beside this engine: a busy thread, or
  /// Android's performance hints.
  public var keepsCPUWarm: Bool { keepWarm != nil || hint != nil }
  public override var notes: String {
    "CPU keep-warm \(keepsCPUWarm ? "on" : "off"); QNN shared I/O \(sharedAllocator == nil ? "off" : "on")"
  }
  public override var coolsWhenIdle: Bool { usesNeuralEngine }

  private let chain: [OrtSession]
  private let sharedAllocator: OrtSharedAllocator?
  private let sharedSessions: [Bool]
  private var bindings: [[OrtBinding]] = []  // [parity][session]
  private let keepAlive: MetalKeepAlive?
  private let keepWarm: CPUKeepWarm?
  private let hint: PerformanceHint?
  private let usesNeuralEngine: Bool

  /// `keepAlive` keeps the GPU clocked up between frames, `keepCPUWarm` the
  /// CPU; each only where a plan runs on that unit.
  public init(plans: [SessionPlan], device: String, keepAlive: Bool = true, keepCPUWarm: Bool = true) throws {
    self.device = device
    var chain: [OrtSession] = []
    for plan in plans {
      chain.append(try OrtSession(model: plan.model, provider: plan.provider, options: plan.options, config: plan.config, threads: plan.threads))
    }
    self.chain = chain
    self.providers = plans.map(\.label)
    let sharedSessions = plans.map { $0.options["enable_htp_shared_memory_allocator"] == "1" }
    for (index, enabled) in sharedSessions.enumerated() where enabled {
      guard plans[index].provider == "QNN", plans[index].usesNeuralEngine, !plans[index].usesGPU else {
        throw OrtError("only QNN HTP sessions may bind QNN shared memory")
      }
    }
    let shared = try sharedSessions.firstIndex(of: true).map { try OrtSharedAllocator(session: chain[$0]) }
    self.sharedAllocator = shared
    self.sharedSessions = sharedSessions

    // What the host stages: every session's inputs no earlier session
    // produces. What it can read: every session's outputs no later one reads.
    var produced = Set<String>()
    var inputs: [String: TensorSpec] = [:]
    for session in chain {
      for spec in session.inputs where !produced.contains(spec.name) && inputs[spec.name] == nil {
        inputs[spec.name] = spec
      }
      produced.formUnion(session.outputs.map(\.name))
    }
    var outputs: [String: TensorSpec] = [:]
    for (index, session) in chain.enumerated() {
      let readLater = Set(chain[(index + 1)...].flatMap { $0.inputs.map(\.name) })
      for spec in session.outputs where !readLater.contains(spec.name) {
        outputs[spec.name] = spec
      }
    }
    // One buffer per tensor name: inputs, hand-offs and outputs alike.
    var sizes: [String: Int] = [:]
    for session in chain {
      for spec in session.inputs + session.outputs {
        sizes[spec.name] = max(sizes[spec.name] ?? 0, spec.byteCount)
      }
    }
    self.keepAlive = keepAlive && plans.contains(where: \.usesGPU) ? MetalKeepAlive.make() : nil
    // Android holds the clocks up when told each frame's time; elsewhere a core spins.
    usesNeuralEngine = plans.contains(where: \.usesNeuralEngine)
    let warmCPU = keepCPUWarm && usesNeuralEngine
    let hint = warmCPU ? PerformanceHint.make() : nil
    self.hint = hint
    self.keepWarm = warmCPU && hint == nil ? CPUKeepWarm() : nil
    try super.init(inputs: inputs, outputs: outputs, sizes: sizes, allocator: shared?.hostAllocator ?? .heap)
    // A throw from here on closes through deinit.
    try rebind([])
  }

  deinit {
    close()
  }

  /// The state_ inputs double-buffered, the bindings made again to swap them.
  public override func bindLoop(_ pairs: [(input: String, output: String)]) throws {
    try doubleBuffer(pairs.map(\.input))
    try rebind(pairs)
  }

  /// One binding per session, or two with a loop: parity 0 reads state_ from
  /// the first buffer and writes next_state_ into the second, parity 1 the
  /// other way round. The next_state_ output has no buffer of its own.
  private func rebind(_ pairs: [(input: String, output: String)]) throws {
    let outputOf = Dictionary(uniqueKeysWithValues: pairs.map { ($0.output, $0.input) })
    var sets: [[OrtBinding]] = []
    for p in 0..<(pairs.isEmpty ? 1 : 2) {
      var set: [OrtBinding] = []
      for (index, session) in chain.enumerated() {
        let ins = session.inputs.map { ($0, buffer($0.name, parity: p)!) }
        let outs = session.outputs.map { spec -> (TensorSpec, UnsafeMutableRawPointer) in
          guard let input = outputOf[spec.name] else { return (spec, buffer(spec.name, parity: p)!) }
          // writes the buffer the state_ input reads next run
          return (spec, buffer(input, parity: p ^ 1)!)
        }
        // These allocations remain host-accessible. Only HTP sessions label
        // their bindings QnnHtpShared; CPU/GPU sessions use CPU memory metadata.
        set.append(try OrtBinding(
          session: session, inputs: ins, outputs: outs,
          sharedAllocator: sharedSessions[index] ? sharedAllocator : nil))
      }
      sets.append(set)
    }
    bindings = sets
  }

  public override func run() throws {
    keepAlive?.pulse()
    keepWarm?.pulse()
    do {
      try super.run()
    } catch {
      keepAlive?.pause()
      throw error
    }
    hint?.report(lastRunNanoseconds)
  }

  public override func execute() throws {
    for binding in bindings[parity] {
      try binding.run()
    }
  }

  /// CoreML and QNN allocate their working set on the first run and the
  /// second is the steady state.
  public override func warm() throws -> String {
    _ = try super.warm()
    let shared = sharedAllocator.map { "; QNN shared I/O: \($0.totalBytes) bytes allocated" } ?? ""
    return "onnxruntime \(OrtRuntime.version) on \(device) in process, sessions \(providers.joined(separator: " then "))\(shared)"
  }

  public override func close() {
    guard !isClosed else { return }
    keepAlive?.close()
    keepWarm?.close()
    hint?.close()
    // They bind the buffers the core frees.
    bindings = []
    super.close()
  }
}
