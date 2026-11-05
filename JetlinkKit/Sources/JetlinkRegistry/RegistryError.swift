import Foundation

/// Anything the registry refuses to do, or could not do.
///
/// The Python registry has a class per kind (RegistryError, NetworkError,
/// VerifyError), and the control server writes a failure as
/// `"<class>: <message>"`. `kind.pythonName` keeps that text the same.
public struct RegistryError: Error, Sendable, Equatable, CustomStringConvertible, LocalizedError {
  public enum Kind: Sendable, Equatable {
    /// Refused: a bad ref, a model that is not there, too little disk.
    case registry
    /// A server could not be reached or did not answer sensibly.
    case network
    /// Bytes arrived, but not the bytes that were asked for.
    case verify
    /// A download or an import was stopped by its caller.
    case cancelled
    /// A model identity that is not a lowercase SHA-256 digest (Python's ValueError).
    case invalidIdentity

    public var pythonName: String {
      switch self {
      case .registry, .cancelled: return "RegistryError"
      case .network: return "NetworkError"
      case .verify: return "VerifyError"
      case .invalidIdentity: return "ValueError"
      }
    }
  }

  public let kind: Kind
  public let message: String

  public init(_ kind: Kind, _ message: String) {
    self.kind = kind
    self.message = message
  }

  /// A server could not be reached or did not answer sensibly: offline,
  /// rather than refused.
  public var isNetwork: Bool { kind == .network }

  public var description: String { message }
  public var errorDescription: String? { message }

  static func registry(_ message: String) -> RegistryError { RegistryError(.registry, message) }
  static func network(_ message: String) -> RegistryError { RegistryError(.network, message) }
  static func verify(_ message: String) -> RegistryError { RegistryError(.verify, message) }
  static func cancelled(_ message: String) -> RegistryError { RegistryError(.cancelled, message) }
  static let invalidIdentity = RegistryError(.invalidIdentity, "model identity must be a lowercase SHA-256 digest")
}
