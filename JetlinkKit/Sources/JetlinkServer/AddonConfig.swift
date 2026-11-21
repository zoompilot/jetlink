// The add-on server: a fork's own models beside the driving model, each on its
// own interface of the comma's gadget. docs/addons.md. From Peter Nguyen's
// V-ASM and sign-vision servers (trung791997/openpilot).
import Foundation
import JetlinkKit

/// One add-on, as --addons names it: its interface's class triple on the
/// comma's gadget, and the ONNX files it serves.
///
///   {"addons": [{"name": "vasm", "class": "ff/56/41", "models": ["/opt/vasm/v_asm_model.onnx"]}]}
public struct AddonConfig: Sendable, Equatable {
  public let name: String
  public let interfaceClass: [UInt8]
  public let models: [URL]

  public init(name: String, interfaceClass: [UInt8], models: [URL]) throws {
    guard name.count <= 16, !name.isEmpty, name.allSatisfy({ $0.isASCII && ($0.isLowercase || $0.isNumber) }), name != "jetlink" else {
      throw AddonConfigError.invalid("add-on name \(name): 1 to 16 of a-z0-9, not jetlink")
    }
    guard interfaceClass.count == 3, interfaceClass != Pinned.usbVendorClass else {
      throw AddonConfigError.invalid("\(name): its class must be three bytes, and not Jetlink's ff/ff/ff")
    }
    guard !models.isEmpty else { throw AddonConfigError.invalid("\(name) names no models") }
    self.name = name
    self.interfaceClass = interfaceClass
    self.models = models
  }

  public static func load(_ url: URL) throws -> [AddonConfig] {
    let data = try Data(contentsOf: url)
    guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any], let list = root["addons"] as? [[String: Any]] else {
      throw AddonConfigError.invalid("\(url.path): expected {\"addons\": [...]}")
    }
    let base = url.deletingLastPathComponent()
    let configs = try list.map { entry in
      guard let name = entry["name"] as? String, let triple = entry["class"] as? String, let models = entry["models"] as? [String] else {
        throw AddonConfigError.invalid("\(url.path): each add-on needs name, class and models")
      }
      guard let bytes = interfaceClass(triple) else { throw AddonConfigError.invalid("\(name): class \(triple) is not like ff/56/41") }
      return try AddonConfig(
        name: name, interfaceClass: bytes, models: models.map { URL(fileURLWithPath: $0, relativeTo: base).standardizedFileURL })
    }
    guard Set(configs.map(\.name)).count == configs.count, Set(configs.map(\.interfaceClass)).count == configs.count else {
      throw AddonConfigError.invalid("\(url.path): two add-ons share a name or a class")
    }
    return configs
  }
}

extension AddonConfig {
  /// "ff/56/41" as bytes, or nil for anything else, Jetlink's own class included.
  public static func interfaceClass(_ triple: String) -> [UInt8]? {
    let parts = triple.split(separator: "/", omittingEmptySubsequences: false)
    let bytes = parts.compactMap { $0.count == 2 ? UInt8($0, radix: 16) : nil }
    guard parts.count == 3, bytes.count == 3, bytes != Pinned.usbVendorClass else { return nil }
    return bytes
  }

  /// The class as "ff/56/41".
  public static func triple(_ bytes: [UInt8]) -> String {
    bytes.map { String(format: "%02x", $0) }.joined(separator: "/")
  }
}

public enum AddonConfigError: Error, CustomStringConvertible {
  case invalid(String)

  public var description: String {
    switch self {
    case .invalid(let why): why
    }
  }
}
