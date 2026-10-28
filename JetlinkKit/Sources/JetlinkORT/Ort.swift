import COrt
import Foundation
import JetlinkServer

/// onnxruntime, through the C shim in COrt.
public enum OrtRuntime {
  public static var version: String { String(cString: jl_version()) }

  /// Opens onnxruntime and makes the process's environment, or throws why it
  /// cannot: on Linux and Android the library is opened at run time, and a
  /// machine may not have it.
  public static func load() throws {
    _ = try OrtEnvironment.shared()
  }
}

public struct OrtError: Error, CustomStringConvertible {
  public let description: String

  init(_ description: String) {
    self.description = description
  }

  /// Throws the shim's message when there is one, and frees it.
  static func check(_ message: UnsafeMutablePointer<CChar>?) throws {
    guard let message else { return }
    defer { jl_free(message) }
    throw OrtError(String(cString: message))
  }
}

/// The process's one onnxruntime environment.
final class OrtEnvironment: @unchecked Sendable {
  let pointer: OpaquePointer

  private init() throws {
    var env: OpaquePointer?
    // 3: errors only
    try OrtError.check(jl_env_create(3, &env))
    guard let env else { throw OrtError("onnxruntime returned no environment") }
    pointer = env
  }

  private static let lock = NSLock()
  nonisolated(unsafe) private static var cached: OrtEnvironment?

  static func shared() throws -> OrtEnvironment {
    lock.lock()
    defer { lock.unlock() }
    if let cached { return cached }
    let env = try OrtEnvironment()
    cached = env
    return env
  }
}

/// One onnxruntime session: a model file, an execution provider with its
/// options, and the inputs and outputs it declares.
final class OrtSession: @unchecked Sendable {
  let pointer: OpaquePointer
  let inputs: [TensorSpec]
  let outputs: [TensorSpec]

  /// `provider` nil runs the session on the CPU alone. `config` is session
  /// config entries on top of jetlink's own; `threads` the intra-op pool.
  init(model: URL, provider: String?, options: [String: String] = [:], config: [String: String] = [:], threads: Int = 1) throws {
    let env = try OrtEnvironment.shared()
    var session: OpaquePointer?
    var strings: [UnsafeMutablePointer<CChar>?] = []
    defer { strings.forEach { free($0) } }
    func entries(_ pairs: [String: String]) -> [jl_option] {
      pairs.sorted { $0.key < $1.key }.map { key, value in
        let k = strdup(key)
        let v = strdup(value)
        strings += [k, v]
        return jl_option(key: k, value: v)
      }
    }
    let providerOptions = entries(options)
    let configEntries = entries(config)
    let providerName = provider.flatMap { strdup($0) }
    strings.append(providerName)
    try providerOptions.withUnsafeBufferPointer { optionBuffer in
      try configEntries.withUnsafeBufferPointer { configBuffer in
        try model.path.withCString { path in
          // the shim ignores the options without a provider
          try OrtError.check(
            jl_session_create(
              env.pointer, path, providerName, optionBuffer.baseAddress, optionBuffer.count, configBuffer.baseAddress, configBuffer.count,
              Int32(threads), &session))
        }
      }
    }
    guard let session else { throw OrtError("onnxruntime returned no session for \(model.lastPathComponent)") }
    pointer = session
    do {
      inputs = try OrtSession.describe(session, output: false)
      outputs = try OrtSession.describe(session, output: true)
    } catch {
      jl_session_release(session)
      throw error
    }
  }

  deinit {
    jl_session_release(pointer)
  }

  private static func describe(_ session: OpaquePointer, output: Bool) throws -> [TensorSpec] {
    try TensorSpec.described(count: jl_session_io_count(session, output ? 1 : 0)) { index, name, type, dims, rank in
      try OrtError.check(jl_session_io_info(session, output ? 1 : 0, index, &name, name.count, &type, &dims, dims.count, &rank))
    }
  }
}

/// Host-accessible allocations tracked by QNN for RPC memory registration.
/// Retaining the session keeps its allocator valid until every buffer is freed.
final class OrtSharedAllocator: @unchecked Sendable {
  let pointer: OpaquePointer
  private let session: OrtSession

  init(session: OrtSession) throws {
    self.session = session
    var allocator: OpaquePointer?
    try OrtError.check(jl_shared_allocator_create(session.pointer, &allocator))
    guard let allocator else { throw OrtError("QNN returned no shared allocator") }
    pointer = allocator
  }

  deinit { jl_shared_allocator_release(pointer) }

  var totalBytes: Int { jl_shared_allocator_total_bytes(pointer) }

  var hostAllocator: HostAllocator {
    HostAllocator(
      allocate: { [self] bytes in
        var memory: UnsafeMutableRawPointer?
        try OrtError.check(jl_shared_allocator_alloc(pointer, bytes, &memory))
        guard let memory else { throw OrtError("QNN shared allocation returned no pointer") }
        return memory
      },
      free: { [self] memory in
        jl_shared_allocator_free(pointer, memory)
      })
  }
}

/// Caller-owned buffers bound to one session's inputs and outputs.
final class OrtBinding {
  private let pointer: OpaquePointer
  private let sharedAllocator: OrtSharedAllocator?

  init(session: OrtSession, inputs: [(TensorSpec, UnsafeMutableRawPointer)], outputs: [(TensorSpec, UnsafeMutableRawPointer)], sharedAllocator: OrtSharedAllocator? = nil) throws {
    self.sharedAllocator = sharedAllocator
    var names: [UnsafeMutablePointer<CChar>?] = []
    var dims: [UnsafeMutablePointer<Int64>] = []
    defer {
      names.forEach { free($0) }
      dims.forEach { $0.deallocate() }
    }
    func tensor(_ spec: TensorSpec, _ data: UnsafeMutableRawPointer) -> jl_tensor {
      let name = strdup(spec.name)
      names.append(name)
      let shape = UnsafeMutablePointer<Int64>.allocate(capacity: max(spec.shape.count, 1))
      for (index, dim) in spec.shape.enumerated() { shape[index] = Int64(dim) }
      dims.append(shape)
      return jl_tensor(name: name, elem_type: spec.type.rawValue, dims: shape, rank: spec.shape.count, data: data, nbytes: spec.byteCount)
    }
    let ins = inputs.map { tensor($0.0, $0.1) }
    let outs = outputs.map { tensor($0.0, $0.1) }
    var binding: OpaquePointer?
    try OrtError.check(jl_binding_create(session.pointer, ins, ins.count, outs, outs.count, sharedAllocator?.pointer, &binding))
    guard let binding else { throw OrtError("onnxruntime returned no binding") }
    pointer = binding
  }

  deinit {
    jl_binding_release(pointer)
  }

  func run() throws {
    try OrtError.check(jl_binding_run(pointer))
  }
}
