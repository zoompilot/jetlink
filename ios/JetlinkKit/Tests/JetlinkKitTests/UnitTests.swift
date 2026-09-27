// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// The pure pieces against what the Python produces for the same input. The
// expected values were printed by the Python package; see each test.

import XCTest

@testable import JetlinkKit

final class WireTests: XCTestCase {
  func hexBytes(_ s: String) -> [UInt8] {
    stride(from: 0, to: s.count, by: 2).map { UInt8(s.dropFirst($0).prefix(2), radix: 16)! }
  }

  /// protocol.pack_header(8, 7, 460000, 3, 0).hex()
  func testHeaderBytesMatchPython() throws {
    let h = Header(msgType: 8, seq: 7, flags: 3, length: 460000)
    XCTAssertEqual(h.bytes, hexBytes("4a4c4e4b020008000700000003000000e0040700000000000000000000000000"))
    let back = try h.bytes.withUnsafeBytes { try Header.decode($0.baseAddress!) }
    XCTAssertEqual(back, h)
  }

  /// protocol.pack_infer_resp(9, 4, 1, 2, 3).hex()
  func testInferResponseBytesMatchPython() {
    XCTAssertEqual(
      InferWire.resp(frameID: 9, status: .notFinite, gpuUs: 1, queueUs: 2, totalUs: 3),
      hexBytes("0900000004000000010000000200000003000000"))
  }

  func testABadMagicOrVersionIsRefused() {
    var b = Header(msgType: 1, seq: 1, flags: 0, length: 0).bytes
    b[0] ^= 1
    XCTAssertThrowsError(try b.withUnsafeBytes { try Header.decode($0.baseAddress!) })
    var v = Header(msgType: 1, seq: 1, flags: 0, length: 0).bytes
    v[4] = 1
    XCTAssertThrowsError(try v.withUnsafeBytes { try Header.decode($0.baseAddress!) })
  }

  func testIdentitiesAreExact() {
    XCTAssertTrue(isSHA256(String(repeating: "a", count: 64)))
    XCTAssertFalse(isSHA256(String(repeating: "a", count: 64) + "\n"))
    XCTAssertFalse(isSHA256(String(repeating: "A", count: 64)))
    XCTAssertFalse(isSHA256("../" + String(repeating: "a", count: 61)))
    XCTAssertTrue(isRef(RegistryConstants.defaultBigModelRef))
    XCTAssertFalse(isRef(String(RegistryConstants.defaultBigModelRef.dropLast())))
  }

  func testJSONKeepsIntegersIntegers() throws {
    let j = try JSON.parse(Data(#"{"a": 4, "b": 4.0, "c": 0.25, "d": null, "e": [true]}"#.utf8))
    XCTAssertEqual(j["a"], .int(4))
    XCTAssertEqual(j["b"], .double(4))
    XCTAssertEqual(JSON.object(["x": .int(4), "y": .double(1), "z": .double(0.25)]).encoded, #"{"x": 4, "y": 1.0, "z": 0.25}"#)
  }
}

final class PickleTests: XCTestCase {
  let expected: [String: SliceRange] = [
    "plan": SliceRange(start: 0, stop: 990), "lead_prob": SliceRange(start: 990, stop: 993),
    "hidden_state": SliceRange(start: 1000, stop: 70000), "a": SliceRange(start: 0, stop: 3),
    "pad": SliceRange(start: 70000, stop: 70002),
  ]

  /// codecs.encode(pickle.dumps(slices, protocol=p), 'base64') for p in 2...5,
  /// where slices = {'plan': slice(0, 990), 'lead_prob': slice(990, 993),
  /// 'hidden_state': slice(1000, 70000), 'a': slice(None, 3), 'pad': slice(70000, 70002)}
  let pickles = [
    "gAJ9cQAoWAQAAABwbGFucQFjX19idWlsdGluX18Kc2xpY2UKcQJLAE3eA06HcQNScQRYCQAAAGxl\nYWRfcHJvYnEFaAJN3gNN4QNOh3EGUnEHWAwAAABoaWRkZW5fc3RhdGVxCGgCTegDSnARAQBOh3EJ\nUnEKWAEAAABhcQtoAk5LA06HcQxScQ1YAwAAAHBhZHEOaAJKcBEBAEpyEQEATodxD1JxEHUu\n",
    "gAN9cQAoWAQAAABwbGFucQFjYnVpbHRpbnMKc2xpY2UKcQJLAE3eA06HcQNScQRYCQAAAGxlYWRf\ncHJvYnEFaAJN3gNN4QNOh3EGUnEHWAwAAABoaWRkZW5fc3RhdGVxCGgCTegDSnARAQBOh3EJUnEK\nWAEAAABhcQtoAk5LA06HcQxScQ1YAwAAAHBhZHEOaAJKcBEBAEpyEQEATodxD1JxEHUu\n",
    "gASVhwAAAAAAAAB9lCiMBHBsYW6UjAhidWlsdGluc5SMBXNsaWNllJOUSwBN3gNOh5RSlIwJbGVh\nZF9wcm9ilGgETd4DTeEDToeUUpSMDGhpZGRlbl9zdGF0ZZRoBE3oA0pwEQEAToeUUpSMAWGUaARO\nSwNOh5RSlIwDcGFklGgESnARAQBKchEBAE6HlFKUdS4=\n",
    "gAWVhwAAAAAAAAB9lCiMBHBsYW6UjAhidWlsdGluc5SMBXNsaWNllJOUSwBN3gNOh5RSlIwJbGVh\nZF9wcm9ilGgETd4DTeEDToeUUpSMDGhpZGRlbl9zdGF0ZZRoBE3oA0pwEQEAToeUUpSMAWGUaARO\nSwNOh5RSlIwDcGFklGgESnARAQBKchEBAE6HlFKUdS4=\n",
  ]

  func testEveryProtocolReadsTheSame() throws {
    for (i, p) in pickles.enumerated() {
      let meta = OnnxMeta(inputs: [], outputs: [.init(name: "outputs", shape: [1, 70002], type: "float16")], props: ["output_slices": p])
      XCTAssertEqual(try meta.outputSlices(), expected, "protocol \(i + 2)")
    }
  }

  /// pickle.dumps({'x': print}, protocol=4): anything but builtins.slice is refused.
  func testAnythingButASliceIsRefused() {
    let meta = OnnxMeta(
      inputs: [], outputs: [], props: ["output_slices": "gASVHQAAAAAAAAB9lIwBeJSMCGJ1aWx0aW5zlIwFcHJpbnSUk5RzLg==\n"])
    XCTAssertThrowsError(try meta.outputSlices())
  }
}

final class RegistryTests: XCTestCase {
  func fixture(_ name: String) throws -> Data {
    try Data(contentsOf: XCTUnwrap(Bundle.module.url(forResource: name, withExtension: nil, subdirectory: "Fixtures")))
  }

  /// jetlink.registry.catalog.parse_catalog on tests/fixtures/catalog_chestnut_v25.json:
  /// 13 models, newest (index 12, 37bfa1413e) first, fa0c6876d3 (index 0) last.
  func testCatalogParsesAsThePythonDoes() throws {
    let models = parseCatalog(try JSON.parse(fixture("catalog_chestnut_v25.json")))
    XCTAssertEqual(models.count, 13)
    XCTAssertEqual(models.first?.ref.prefix(10), "37bfa1413e")
    XCTAssertEqual(models.first?.index, 12)
    XCTAssertEqual(models.first?.name, "Cinque Terre Model V2 (September 08, 2026)")
    XCTAssertEqual(models[1].ref.prefix(10), "68b5f8e486")
    XCTAssertEqual(models[2].name, "BMRLNAP Model v6 (September 01, 2026)")
    XCTAssertEqual(models.last?.ref.prefix(10), "fa0c6876d3")
    XCTAssertEqual(models.last?.index, 0)
  }

  /// parse_pointer_text on tests/fixtures/pointer_f877d7a0.txt
  func testPointerParsesAsThePythonDoes() throws {
    let p = parsePointer(String(decoding: try fixture("pointer_f877d7a0.txt"), as: UTF8.self))
    XCTAssertEqual(p, Pointer(oid: "a086d5249fc308bb73993d1e64630c669d4c7df5bde85f42ad61902543648525", size: 765953504))
    XCTAssertNil(parsePointer("oid sha256:abc\nsize 12\n"))
    XCTAssertNil(parsePointer(String(repeating: "x", count: 5000)))
  }
}

final class CalibrationTests: XCTestCase {
  func r(_ p99: Double) -> BenchResult { BenchResult(mean: p99 - 2, p99: p99, max: p99 + 1, frames: 60) }

  func testTheNeuralEngineWinsWheneverItMakesTheBudget() {
    XCTAssertEqual(EngineHost.choose([.ane: r(30), .coreml: r(20)]), .ane)
    XCTAssertEqual(EngineHost.choose([.ane: r(35), .coreml: r(10)]), .ane)
  }

  func testOtherwiseTheGPUIfItDoes() {
    XCTAssertEqual(EngineHost.choose([.ane: r(40), .coreml: r(30)]), .coreml)
  }

  func testOtherwiseTheFaster() {
    XCTAssertEqual(EngineHost.choose([.ane: r(54), .coreml: r(48)]), .coreml)
    XCTAssertEqual(EngineHost.choose([.ane: r(45), .coreml: r(60)]), .ane)
    XCTAssertEqual(EngineHost.choose([.coreml: r(60)]), .coreml)
    XCTAssertNil(EngineHost.choose([:]))
  }
}

final class ConvertTests: XCTestCase {
  func testEveryByteConvertsExactly() {
    let src = (0..<256).map { UInt8($0) } + (0..<256).map { UInt8(255 - $0) }
    var dst = [Float16](repeating: .nan, count: src.count)
    src.withUnsafeBufferPointer { s in dst.withUnsafeMutableBufferPointer { Convert.u8ToF16(s.baseAddress!, $0.baseAddress!, count: s.count) } }
    XCTAssertEqual(dst, src.map { Float16(Float($0)) })
  }

  func testFloatConversionsRoundTrip() {
    let src: [Float] = [0, -0.0, 1, 1.0009765625 + 1.0 / 4096, 65504, 1e6, -7e4, 3e-8, .nan, .infinity, -.infinity]
    var half = [Float16](repeating: 0, count: src.count)
    src.withUnsafeBufferPointer { s in half.withUnsafeMutableBufferPointer { Convert.f32ToF16(s.baseAddress!, $0.baseAddress!, count: s.count) } }
    for (a, b) in zip(half, src.map { Float16($0) }) {
      XCTAssertEqual(a.bitPattern, b.bitPattern, "\(b)")
    }
    var back = [Float](repeating: 0, count: half.count)
    half.withUnsafeBufferPointer { s in back.withUnsafeMutableBufferPointer { Convert.f16ToF32(s.baseAddress!, $0.baseAddress!, count: s.count) } }
    for (a, b) in zip(back, half.map { Float($0) }) {
      XCTAssertEqual(a.bitPattern, b.bitPattern)
    }
  }

  func testFiniteCheck() {
    var v = [Float](repeating: 65504, count: 18452)
    XCTAssertTrue(Convert.allFinite(v, count: v.count, fromFloat16: true))
    v[9000] = .nan
    XCTAssertFalse(Convert.allFinite(v, count: v.count, fromFloat16: true))
    XCTAssertFalse(Convert.allFinite(v, count: v.count, fromFloat16: false))
    v[9000] = -.infinity
    XCTAssertFalse(Convert.allFinite(v, count: v.count, fromFloat16: true))
    v[9000] = 1
    v[100] = .infinity
    v[200] = -.infinity
    XCTAssertFalse(Convert.allFinite(v, count: v.count, fromFloat16: true))
  }
}
