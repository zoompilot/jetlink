// swift-tools-version: 6.0
//
// Jetlink's server, in Swift, for an iPhone and for testing it on a Mac.
//
// JetlinkKit is everything the Python server does that the phone needs: the
// wire protocol, the TCP transport, the history queues, the ONNX preparation
// onnx_patch.py does, and onnxruntime with CoreML through its C API. The iOS
// app (ios/Jetlink) is a SwiftUI shell around it. `jetlink-swift` is the same
// server as a macOS command, so scripts/bench_link.py and
// scripts/verify_parity.py can check it against the Python one on a Mac.
//
// onnxruntime is the release the Mac backend is measured with (1.29.0), as
// Microsoft publishes it for CocoaPods: a static xcframework with iOS,
// simulator and macOS slices. The checksum is that archive's SHA-256.

import PackageDescription

let package = Package(
  name: "JetlinkKit",
  platforms: [.iOS(.v17), .macOS(.v14)],
  products: [
    .library(name: "JetlinkKit", targets: ["JetlinkKit"]),
    .executable(name: "jetlink-swift", targets: ["jetlink-swift"]),
  ],
  targets: [
    .binaryTarget(
      name: "onnxruntime",
      url: "https://download.onnxruntime.ai/pod-archive-onnxruntime-c-1.29.0.zip",
      checksum: "ab89ea27b074201b83c12526d7f7206b916ecd5315174d372d6c27b659e49860"),
    .target(
      name: "COrtShim",
      dependencies: ["onnxruntime"],
      linkerSettings: [
        .linkedFramework("CoreML"),
        .linkedFramework("Foundation"),
        .linkedFramework("Network"),
        .linkedLibrary("c++"),
      ]),
    .target(
      name: "JetlinkKit",
      dependencies: ["COrtShim"]),
    .executableTarget(
      name: "jetlink-swift",
      dependencies: ["JetlinkKit"]),
    .testTarget(
      name: "JetlinkKitTests",
      dependencies: ["JetlinkKit"],
      resources: [.copy("Fixtures")]),
  ]
)
