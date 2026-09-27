// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.

import XCTest

@testable import JetlinkKit

/// The Swift queues against what jetlink.queues produced for the same frames
/// (ios/scripts/queue_fixtures.py), byte for byte.
final class QueueTests: XCTestCase {
  func fixture(_ name: String) throws -> Data {
    let url = try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: nil, subdirectory: "Fixtures"))
    return try Data(contentsOf: url)
  }

  func check(_ name: String) throws {
    let json = try JSON.parse(fixture("\(name).json"))
    let spec = try ModelSpec(json: json)
    let frames = try XCTUnwrap(json["frames"]?.int)
    let input = try fixture("\(name).in.bin")
    let expected = try fixture("\(name).out.bin")
    let q = try PolicyQueues(spec: spec)

    var buffers: [String: UnsafeMutablePointer<Float16>] = [:]
    var dest: [String: TensorBuffer] = [:]
    for n in PolicyQueues.inputNames {
      let count = product(spec.inputShapes[n]!)
      let p = UnsafeMutablePointer<Float16>.allocate(capacity: count)
      p.initialize(repeating: .nan, count: count)
      buffers[n] = p
      dest[n] = TensorBuffer(base: UnsafeMutableRawPointer(p), count: count, kind: .float16)
    }
    defer { buffers.values.forEach { $0.deallocate() } }

    let frameIn = 1 + spec.warpedNbytes + spec.packedNbytes
    var outOffset = 0
    for i in 0..<frames {
      let base = i * frameIn
      if input[base] == 1 { q.reset() }
      let warped = [UInt8](input[(base + 1)..<(base + 1 + spec.warpedNbytes)])
      let packedBytes = input[(base + 1 + spec.warpedNbytes)..<(base + frameIn)]
      let packed: [Float] = packedBytes.withUnsafeBytes { raw in
        (0..<spec.packedNelem).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: 4 * $0, as: UInt32.self))) }
      }
      warped.withUnsafeBufferPointer { w in
        packed.withUnsafeBufferPointer { p in q.step(warped: w.baseAddress!, packed: p.baseAddress!, into: dest) }
      }
      for n in PolicyQueues.inputNames {
        let bytes = dest[n]!.count * 2
        let got = Data(bytes: buffers[n]!, count: bytes)
        let want = expected[outOffset..<(outOffset + bytes)]
        XCTAssertEqual(got, Data(want), "\(name) frame \(i): \(n) differs from the Python queues")
        outOffset += bytes
      }
    }
    XCTAssertEqual(outOffset, expected.count)
  }

  func testMatchesPythonFrameSkip4() throws { try check("queues_fs4") }
  func testMatchesPythonFrameSkip2() throws { try check("queues_fs2") }
}
