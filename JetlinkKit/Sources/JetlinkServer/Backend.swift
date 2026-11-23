import Foundation
import JetlinkRegistry

/// Progress of a build or a load: stage, fraction, message.
public typealias ProgressFn = @Sendable (String, Double, String) -> Void

/// The artifact on disk is not one this backend can load: another runtime's
/// compiled cache, an older preparation, a plan from another TensorRT build.
/// Any backend's `load` throws it for what is wrong with the file itself, and
/// the host deletes the artifact and rebuilds it once from the ONNX when that
/// is on disk at the size asked for; without it the client uploads again.
/// Anything else a load throws (memory, a device gone) leaves the artifact.
public struct ArtifactInvalid: Error, CustomStringConvertible {
  public let description: String

  public init(_ description: String) {
    self.description = description
  }
}

/// An error an engine cannot come back from, such as CUDA's sticky errors
/// (an illegal address, a failed launch): the context is broken, and every
/// later frame and every rejoin would fail with the engine still "loaded".
/// The session answers the frame INFER_FAILED, then calls
/// `ServerHooks.fatal`. A type whose codes are only sometimes fatal says
/// which through `isFatal`.
public protocol FatalEngineError: Error {
  var isFatal: Bool { get }
}

/// A loaded model, ready to run a frame at a time: a backend's engine
/// (JetlinkORT's `OrtEngine`), or the tests' one that needs no runtime.
public protocol Engine: AnyObject {
  var inputs: [String: TensorSpec] { get }
  var outputs: [String: TensorSpec] { get }
  var lastGpuUs: UInt32 { get }
  /// What ran beside the model, for a benchmark report's build line: "CPU
  /// keep-warm on". Empty when there is nothing to say.
  var notes: String { get }
  /// What runs the model when the backend's device does not say it all:
  /// LiteRT's NPU profile falls back to the GPU. Empty otherwise.
  var accelerator: String { get }
  /// Its first run after a long idle is slow, so the host runs it on zeros
  /// between sessions (`EngineHost.warmIfIdle`): Apple's Neural Engine.
  var coolsWhenIdle: Bool { get }
  /// Where the host writes an input. For every input but a looped pair's
  /// state_ one the pointer stays the same from load to close, as does
  /// `output`'s for every output the engine does not loop: staging looks
  /// them up once, at load.
  func hostInput(_ name: String) -> UnsafeMutableRawPointer?
  func output(_ name: String) -> UnsafeRawPointer?
  /// Feeds each next_state_ output back as its state_ input from the next
  /// run on, inside the engine. Throws for a pair it cannot loop.
  func loopState(_ pairs: [(input: String, output: String)]) throws
  func resetState()
  func run() throws
  /// One run between sessions (`EngineHost.warmIfIdle`): the model alone,
  /// without what keeps the clocks up beside a comma's frames.
  func runIdle() throws
  func warm() throws -> String
  /// Logs the engine's own timing so far: a benchmark's end.
  func flushTiming()
  func close()
}

/// Turns an ONNX file into an artifact it can load quickly, and loads one.
/// The Swift form of `server/backends/base.Backend`.
public protocol EngineBackend: AnyObject, Sendable {
  /// "ort", as the hello reports it.
  var name: String { get }
  /// The artifact's extension.
  var suffix: String { get }
  /// The runtime's release, "1.29.0".
  var runtimeVersion: String { get }
  /// The device part of the tag: "ane-Apple_M1_Pro", "htp-SM8650".
  func deviceTag() -> String
  /// What an artifact is valid for: runtime version and device, sanitized.
  func tag() -> String
  /// backend, runtime_version, device, and whatever else the backend adds
  /// (TensorRT's `trt_version`, which the comma logs): for the hello.
  func describe() -> [String: String]
  func deriveSpec(model: URL, sha256: String, nbytes: Int64, frameSkip: Int) throws -> ModelSpec
  func build(model: URL, artifact: URL, report: @escaping ProgressFn, metaExtra: [String: Any]) throws
  /// `build`, ending early with an error once `shouldStop` says so: an
  /// add-on's build gives the device up to the driving model's. A backend
  /// that cannot stop a build part way runs it to its end.
  func build(
    model: URL, artifact: URL, report: @escaping ProgressFn, metaExtra: [String: Any], shouldStop: @escaping @Sendable () -> Bool
  ) throws
  func load(artifact: URL, report: @escaping ProgressFn) throws -> any Engine
}

extension Engine {
  public var notes: String { "" }
  public var accelerator: String { "" }
  public var coolsWhenIdle: Bool { false }
  public func runIdle() throws { try run() }
  public func flushTiming() {}
}

extension EngineBackend {
  public func tag() -> String {
    "\(name)\(sanitize(runtimeVersion)).\(deviceTag())"
  }

  public func describe() -> [String: String] {
    ["backend": name, "runtime_version": runtimeVersion, "device": deviceTag()]
  }

  public func build(
    model: URL, artifact: URL, report: @escaping ProgressFn, metaExtra: [String: Any], shouldStop: @escaping @Sendable () -> Bool
  ) throws {
    try build(model: model, artifact: artifact, report: report, metaExtra: metaExtra)
  }

}

/// A version or device name as a filename component.
public func sanitize(_ s: String) -> String {
  String(s.map { $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "." || $0 == "_" || $0 == "-") ? $0 : "_" })
}
