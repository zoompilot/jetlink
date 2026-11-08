import Foundation
import JetlinkRegistry

/// A source-only, hash-bound split of one model's same-frame causal work.
/// Another architecture needs its own independently validated contract.
package enum CausalPrefillBundle {
  package static let identity = "causal-prefill-v1-5a159599"
  package static let sourceSHA = "404a18cfd86d29637d20c697dfde245bb47c666ae016730ab674c65f4d1e1aa4"
  private static let manifestSHA = "5a159599f57ce154026baf967825cc4890e2e93ebca00adda1819b1c5354b7a3"

  private struct Manifest: Decodable {
    let format: Int
    let strategy: String
    let sourceModelSHA256: String
    let parts: [Part]
    private enum CodingKeys: String, CodingKey {
      case format, strategy, parts
      case sourceModelSHA256 = "source_model_sha256"
    }
  }
  private struct Part: Decodable {
    let name: String
    let file: String
    let unit: String
    let weightBytes: Int64
    let sha256: String
    private enum CodingKeys: String, CodingKey {
      case name, file, unit, sha256
      case weightBytes = "weight_bytes"
    }
  }

  package static func prepare(model: URL, into directory: URL, bundle: URL?) throws -> PreparedModel {
    guard let bundle else { throw HostError.invalid("the causal-prefill model bundle is missing") }
    let manifestURL = bundle.appendingPathComponent("manifest.json")
    guard try Registry.hashFile(manifestURL).0 == manifestSHA else {
      throw HostError.invalid("causal-prefill manifest does not match this engine")
    }
    let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
    guard manifest.format == 1, manifest.strategy == "causal-prefill-v1",
      manifest.sourceModelSHA256 == sourceSHA,
      manifest.parts.map(\.name) == ["vision", "history", "decode", "join"],
      manifest.parts.map(\.unit) == ["htp", "gpu", "htp", "cpu"],
      try Registry.hashFile(model).0 == sourceSHA
    else { throw HostError.invalid("causal-prefill requires the exact Cinque Terre V3 export") }
    var parts: [PreparedModel.Part] = []
    for part in manifest.parts {
      guard part.file == "\(part.name).onnx", part.weightBytes >= 0, CacheLayout.isSHA256(part.sha256) else {
        throw HostError.invalid("invalid causal-prefill part identity")
      }
      let source = bundle.appendingPathComponent(part.file)
      guard try Registry.hashFile(source).0 == part.sha256 else {
        throw HostError.invalid("causal-prefill part \(part.name) failed its hash check")
      }
      try FileManager.default.copyItem(at: source, to: directory.appendingPathComponent(part.file))
      parts.append(.init(name: part.name, file: part.file, weightBytes: part.weightBytes))
    }
    try FileManager.default.copyItem(at: manifestURL, to: directory.appendingPathComponent("causal-prefill.json"))
    return PreparedModel(parts: parts, summary: "hash-verified causal split; original weights and same-frame feedback")
  }

  /// HTP parts become device-compiled contexts. The GPU graph and CPU assembly
  /// remain source ONNX and must still match the pinned contract at reload.
  package static func verifyRetainedParts(in directory: URL) throws {
    let file = directory.appendingPathComponent("causal-prefill.json")
    guard (try? Registry.hashFile(file).0) == manifestSHA else {
      throw ArtifactInvalid("causal-prefill retained manifest differs")
    }
    let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: file))
    for part in manifest.parts where part.unit != "htp" {
      guard (try? Registry.hashFile(directory.appendingPathComponent(part.file)).0) == part.sha256 else {
        throw ArtifactInvalid("causal-prefill retained part \(part.name) differs")
      }
    }
  }
}
