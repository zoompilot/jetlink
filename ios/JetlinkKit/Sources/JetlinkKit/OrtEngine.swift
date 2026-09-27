// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// One onnxruntime session, loaded and ready to run a frame at a time.
//
// The Python backend runs its session in a worker process because
// onnxruntime holds the GIL through a CoreML compile. Swift has no GIL, so
// the session lives here, created on the host's job thread and run on the
// session thread. Every input and output is a page-aligned buffer bound to
// the session once; a frame is the queues gathering into those buffers,
// `jl_ort_run`, and the output read back.

import COrtShim
import Foundation

public enum ComputeUnits: String, Sendable, CaseIterable {
  /// CoreML with the Neural Engine allowed: MLComputeUnits "ALL".
  case ane
  /// CoreML on the GPU: "CPUAndGPU". Called `coreml` as in the Python backend.
  case coreml
  /// onnxruntime's own CPU provider, for tests and benches only.
  case cpu

  var mlComputeUnits: String? {
    switch self {
    case .ane: "ALL"
    case .coreml: "CPUAndGPU"
    case .cpu: nil
    }
  }

  public var label: String {
    switch self {
    case .ane: "Neural Engine"
    case .coreml: "GPU"
    case .cpu: "CPU"
    }
  }
}

public struct EngineError: Error, CustomStringConvertible {
  public let description: String
  public init(_ d: String) { description = d }
}

public enum OrtRuntime {
  public static var version: String { String(cString: jl_ort_version()) }

  /// Whether the GPU path runs the Metal keep-alive between frames (see
  /// MetalKeepAlive). Read when an engine is created.
  public static var gpuKeepAlive: Bool {
    get { MetalKeepAlive.enabled }
    set { MetalKeepAlive.enabled = newValue }
  }

  /// How the Neural Engine path keeps the CPU ready between frames (see
  /// CPUKeepWarm). Read when an engine is created; JETLINK_CPU_KEEPWARM
  /// overrides it for measuring.
  nonisolated(unsafe) public static var cpuKeepWarm: KeepWarmMode =
    ProcessInfo.processInfo.environment["JETLINK_CPU_KEEPWARM"].flatMap(KeepWarmMode.init(rawValue:)) ?? .on

  static let initialized: Result<Void, EngineError> = {
    var err = [CChar](repeating: 0, count: 1024)
    // 3: errors only. The CoreML partitioner is chatty at warning.
    if jl_ort_init(3, &err, err.count) != 0 { return .failure(EngineError(cText(err))) }
    return .success(())
  }()

  static func ensure() throws { try initialized.get() }
}

public final class OrtEngine: @unchecked Sendable {
  public struct IO: Sendable {
    public let name: String
    public let shape: [Int]
    public let elemType: Int32
    public var count: Int { product(shape) }
  }

  public let inputs: [IO]
  public let outputs: [IO]
  public let providers: String
  public let units: ComputeUnits
  /// How long the last run took, in microseconds: the gpu_us on the wire.
  public private(set) var lastRunUs = 0

  private var session: OpaquePointer?
  private var inputBuffers: [String: TensorBuffer] = [:]
  private var outputBuffers: [TensorBuffer] = []
  private var allocations: [(UnsafeMutableRawPointer)] = []
  private let output32: UnsafeMutablePointer<Float>
  private let keepalive: MetalKeepAlive?
  private let keepWarm: CPUKeepWarm?

  /// Creates the session. With CoreML this is the conversion and compile, or
  /// the load of a compile already in `cacheDirectory`.
  public init(model: String, units: ComputeUnits, cacheDirectory: String?, keepAlive: Bool = true) throws {
    try OrtRuntime.ensure()
    self.units = units
    var err = [CChar](repeating: 0, count: 2048)
    var keys: [String] = [], values: [String] = []
    if let cu = units.mlComputeUnits {
      keys = ["ModelFormat", "MLComputeUnits"]
      values = ["MLProgram", cu]
      if let cacheDirectory {
        keys.append("ModelCacheDirectory")
        values.append(cacheDirectory)
      }
      if units == .ane {
        // Measured on an M1 Pro: p99 45.9 -> 36.0 ms back to back, parity
        // unchanged. A load-time hint; the compiled model is the same.
        keys.append("SpecializationStrategy")
        values.append("FastPrediction")
      }
    }
    let s: OpaquePointer? = withCStrings(keys) { k in
      withCStrings(values) { v in
        jl_ort_session_create(model, units == .cpu ? 0 : 1, k, v, keys.count, 1, &err, err.count)
      }
    }
    guard let s else { throw EngineError("onnxruntime: \(cText(err))") }
    session = s
    providers = String(cString: jl_ort_providers(s))

    func describe(_ io: UnsafePointer<jl_ort_io>) -> IO {
      let name = withUnsafeBytes(of: io.pointee.name) { String(cString: $0.bindMemory(to: CChar.self).baseAddress!) }
      let dims = withUnsafeBytes(of: io.pointee.dims) { Array($0.bindMemory(to: Int64.self).prefix(Int(io.pointee.rank))) }
      return IO(name: name, shape: dims.map(Int.init), elemType: io.pointee.elem_type)
    }
    inputs = (0..<jl_ort_input_count(s)).map { describe(jl_ort_input(s, $0)) }
    outputs = (0..<jl_ort_output_count(s)).map { describe(jl_ort_output(s, $0)) }
    let outCount = outputs.first?.count ?? 0
    output32 = .allocate(capacity: max(outCount, 1))
    keepalive = keepAlive && units == .coreml ? MetalKeepAlive.make() : nil
    keepWarm = keepAlive && units == .ane ? CPUKeepWarm.make(OrtRuntime.cpuKeepWarm) : nil

    do {
      guard !outputs.isEmpty else { throw EngineError("the model has no outputs") }
      for (i, io) in inputs.enumerated() {
        let b = try allocate(io, "input")
        inputBuffers[io.name] = b
        if jl_ort_bind_input(s, i, b.base, b.byteCount, &err, err.count) != 0 {
          throw EngineError("binding \(io.name): \(cText(err))")
        }
      }
      for (i, io) in outputs.enumerated() {
        let b = try allocate(io, "output")
        outputBuffers.append(b)
        if jl_ort_bind_output(s, i, b.base, b.byteCount, &err, err.count) != 0 {
          throw EngineError("binding \(io.name): \(cText(err))")
        }
      }
    } catch {
      close()
      throw error
    }
  }

  deinit {
    close()
    output32.deallocate()
  }

  private func allocate(_ io: IO, _ what: String) throws -> TensorBuffer {
    let kind: TensorBuffer.Kind
    switch io.elemType {
    case Int32(JL_ORT_FLOAT16): kind = .float16
    case Int32(JL_ORT_FLOAT): kind = .float32
    default: throw EngineError("\(what) \(io.name) is \(OnnxType.name(io.elemType)); jetlink stages float16 or float32")
    }
    if io.shape.contains(where: { $0 <= 0 }) {
      throw EngineError("\(io.name) has a dynamic shape \(io.shape); jetlink builds fixed-shape engines")
    }
    let bytes = io.count * (kind == .float16 ? 2 : 4)
    let p = UnsafeMutableRawPointer.allocate(byteCount: max(bytes, 1), alignment: Int(getpagesize()))
    p.initializeMemory(as: UInt8.self, repeating: 0, count: max(bytes, 1))
    allocations.append(p)
    return TensorBuffer(base: p, count: io.count, kind: kind)
  }

  /// The staging buffer for an input: write into it, then run.
  public func hostInput(_ name: String) -> TensorBuffer? { inputBuffers[name] }

  public var hostInputs: [String: TensorBuffer] { inputBuffers }

  /// Runs one frame over the staging buffers.
  public func run() throws {
    guard let session else { throw EngineError("engine is closed") }
    keepalive?.pulse()
    keepWarm?.pulse()
    var err = [CChar](repeating: 0, count: 1024)
    let t0 = monotonic()
    let rc = jl_ort_run(session, &err, err.count)
    lastRunUs = Int((monotonic() - t0) * 1e6)
    if rc != 0 {
      keepalive?.pause()
      throw EngineError("onnxruntime: \(cText(err))")
    }
  }

  /// The first output as float32, valid until the next run: the Python's
  /// np.asarray(out, dtype=np.float32).
  public func output() -> UnsafeBufferPointer<Float> {
    let b = outputBuffers[0]
    switch b.kind {
    case .float32:
      return UnsafeBufferPointer(start: b.base.assumingMemoryBound(to: Float.self), count: b.count)
    case .float16:
      Convert.f16ToF32(b.base.assumingMemoryBound(to: Float16.self), output32, count: b.count)
      return UnsafeBufferPointer(start: output32, count: b.count)
    }
  }

  /// Whether the first output is float16, so its float32 copy cannot overflow a sum.
  public var outputIsFloat16: Bool { outputBuffers.first?.kind == .float16 }

  /// CoreML allocates its working set on the first run; the second is the
  /// steady state.
  public func warm() throws -> String {
    try run()
    try run()
    return "onnxruntime \(OrtRuntime.version) on \(units.label), providers \(providers)"
  }

  public func close() {
    keepalive?.close()
    keepWarm?.close()
    if let s = session {
      jl_ort_session_release(s)
      session = nil
    }
    for p in allocations { p.deallocate() }
    allocations.removeAll()
    inputBuffers.removeAll()
    outputBuffers.removeAll()
  }
}

/// Calls `body` with a C array of C strings that live for the call.
private func withCStrings<R>(_ strings: [String], _ body: (UnsafePointer<UnsafePointer<CChar>?>?) -> R) -> R {
  let dup = strings.map { strdup($0) }
  defer { dup.forEach { free($0) } }
  let ptrs: [UnsafePointer<CChar>?] = dup.map { $0.map { UnsafePointer($0) } }
  return ptrs.withUnsafeBufferPointer { body($0.baseAddress) }
}
