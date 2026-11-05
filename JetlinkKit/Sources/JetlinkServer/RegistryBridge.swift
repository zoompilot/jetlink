import Foundation
import JetlinkKit
import JetlinkRegistry

/// The registry the apps use, as the control layer asks for it.
extension Registry: ModelRegistry {
  public func resolvePointer(ref: String) async throws -> (sha256: String, size: Int64) {
    let pointer = try await resolve(ref: ref)
    return (pointer.oid, pointer.size)
  }

  public func importModelFile(at url: URL, name: String?, progress: @escaping @Sendable (Double) -> Void) async throws -> String {
    try await importModel(at: url, name: name, progress: progress, shouldStop: { false }).sha256
  }

  public func fetch(
    sha256: String, size: Int64, progress: @escaping @Sendable (Double) -> Void, shouldStop: @escaping @Sendable () -> Bool
  ) async throws -> URL {
    try await fetch(pointer: Pointer(oid: sha256, size: size), progress: progress, shouldStop: shouldStop)
  }
}
