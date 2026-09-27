// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// The frame path's bulk conversions, through Accelerate's vImage.
//
// A frame converts about 800,000 camera bytes to float16 and 35,000 floats
// each way. As element loops these cost 0.2 ms in an optimized build and
// over 40 ms in an unoptimized one: an iPhone 17 Pro running the app as
// Xcode's Run builds it, Debug, measured 65 ms a frame around a 21 ms model.
// vImage is vectorized whatever the build, and QueueTests holds these to the
// Python queues bit for bit, NaN, overflow and subnormals included.

import Accelerate

enum Convert {
  /// uint8 to float16 bit patterns: 0...255 are all exact in float16.
  private static let u8ToF16Bits: [UInt16] = (0..<256).map { Float16(Float($0)).bitPattern }

  private static func buffer(_ p: UnsafeRawPointer, _ n: Int, _ size: Int) -> vImage_Buffer {
    vImage_Buffer(data: UnsafeMutableRawPointer(mutating: p), height: 1, width: vImagePixelCount(n), rowBytes: n * size)
  }

  static func u8ToF16(_ src: UnsafePointer<UInt8>, _ dst: UnsafeMutablePointer<Float16>, count n: Int) {
    guard n > 0 else { return }
    var s = buffer(src, n, 1)
    var d = buffer(dst, n, 2)
    _ = u8ToF16Bits.withUnsafeBufferPointer { table in
      vImageLookupTable_Planar8toPlanar16(&s, &d, table.baseAddress!, vImage_Flags(kvImageDoNotTile))
    }
  }

  /// Round to nearest even, as numpy's float32 to float16 cast.
  static func f32ToF16(_ src: UnsafePointer<Float>, _ dst: UnsafeMutablePointer<Float16>, count n: Int) {
    guard n > 0 else { return }
    var s = buffer(src, n, 4)
    var d = buffer(dst, n, 2)
    vImageConvert_PlanarFtoPlanar16F(&s, &d, vImage_Flags(kvImageDoNotTile))
  }

  static func f16ToF32(_ src: UnsafePointer<Float16>, _ dst: UnsafeMutablePointer<Float>, count n: Int) {
    guard n > 0 else { return }
    var s = buffer(src, n, 2)
    var d = buffer(dst, n, 4)
    vImageConvert_Planar16FtoPlanarF(&s, &d, vImage_Flags(kvImageDoNotTile))
  }

  /// Whether every value is finite. The sum of finite float16-range values
  /// cannot overflow a float for any output this size, and a NaN or an
  /// infinity anywhere makes the sum non-finite.
  static func allFinite(_ p: UnsafePointer<Float>, count n: Int, fromFloat16: Bool) -> Bool {
    guard n > 0 else { return true }
    if fromFloat16 && n < 5_000_000 {
      var sum: Float = 0
      vDSP_sve(p, 1, &sum, vDSP_Length(n))
      return sum.isFinite
    }
    for i in 0..<n where !p[i].isFinite { return false }
    return true
  }
}
