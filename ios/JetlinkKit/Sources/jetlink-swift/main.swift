// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// The iPhone's server as a Mac command, for checking it against the Python
// one with the same tools:
//
//   jetlink-swift serve [--port 5599] [--device auto|ane|coreml|cpu] [--cache DIR]
//   jetlink-swift build MODEL.onnx [--device ...] [--cache DIR]
//   jetlink-swift prepare SRC.onnx DST.onnx --device ane|coreml|cpu [--cache-key KEY]
//   jetlink-swift spec MODEL.onnx

import Foundation
import JetlinkKit

func usage() -> Never {
  FileHandle.standardError.write(
    Data(
      """
      usage: jetlink-swift serve [--port N] [--host ADDR] [--device auto|ane|coreml|cpu] [--cache DIR]
             jetlink-swift build MODEL.onnx [--device ...] [--cache DIR] [--frame-skip N] [--bench SECONDS]
             jetlink-swift prepare SRC.onnx DST.onnx --device ane|coreml|cpu [--cache-key KEY]
             jetlink-swift spec MODEL.onnx

      """.utf8))
  exit(2)
}

var args = Array(CommandLine.arguments.dropFirst())
guard let command = args.first else { usage() }
args.removeFirst()

@MainActor func option(_ name: String) -> String? {
  guard let i = args.firstIndex(of: name) else { return nil }
  guard i + 1 < args.count else { usage() }
  let v = args[i + 1]
  args.removeSubrange(i...(i + 1))
  return v
}

Log.shared.setSink(minimum: ProcessInfo.processInfo.environment["JETLINK_DEBUG"] == nil ? .info : .debug) { line in
  let t = ISO8601DateFormatter.string(from: line.date, timeZone: .current, formatOptions: [.withTime, .withColonSeparatorInTime])
  FileHandle.standardError.write(Data("\(t) \(line.level) \(line.category): \(line.message)\n".utf8))
}

@MainActor func cacheRoot() -> URL {
  if let c = option("--cache") { return URL(fileURLWithPath: c, isDirectory: true) }
  if let c = ProcessInfo.processInfo.environment["JETLINK_CACHE"] { return URL(fileURLWithPath: c, isDirectory: true) }
  return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/jetlink-swift", isDirectory: true)
}

@MainActor func preference() -> ComputePreference {
  let d = option("--device") ?? "auto"
  guard let p = ComputePreference(rawValue: d) else { usage() }
  return p
}

do {
  switch command {
  case "serve":
    let port = UInt16(option("--port") ?? "5599") ?? 5599
    let bind = option("--host") ?? "0.0.0.0"
    let pref = preference()
    let root = cacheRoot()
    guard args.isEmpty else { usage() }
    let host = EngineHost(cache: try EngineCache(root: root), preference: pref)
    let server = JetlinkServer(host: host, bindAddress: bind, port: port)
    print("jetlink-swift: onnxruntime \(OrtRuntime.version), \(pref.label), cache \(root.path)")
    try server.start()
    signal(SIGINT, SIG_IGN)
    let sigint = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
    sigint.setEventHandler {
      server.stop()
      host.close()
      exit(0)
    }
    sigint.resume()
    dispatchMain()
  case "build":
    let skip = Int(option("--frame-skip") ?? "4") ?? 4
    let benchSeconds = option("--bench").flatMap(Double.init)
    let pref = preference()
    let root = cacheRoot()
    guard args.count == 1 else { usage() }
    let cache = try EngineCache(root: root)
    let src = URL(fileURLWithPath: args[0])
    let (sha, n) = try sha256File(src.path)
    let dst = try cache.modelPath(sha)
    if !FileManager.default.fileExists(atPath: dst.path) { try FileManager.default.copyItem(at: src, to: dst) }
    let host = EngineHost(cache: cache, preference: pref)
    let done = DispatchSemaphore(value: 0)
    host.subscribe { kind, payload in
      if kind == "progress" {
        print(String(format: "  %-8@ %5.1f%%  %@", payload["stage"]?.string ?? "", (payload["frac"]?.double ?? 0) * 100, payload["msg"]?.string ?? ""))
      }
      if kind == "engine", let s = payload["state"]?.string, s == "ready" || s == "failed" { done.signal() }
    }
    let t0 = Date()
    let first = host.request(try EngineRequest(sha256: sha, nbytes: n, frameSkip: skip), session: nil)
    if first["state"]?.string != "ready" { done.wait() }
    let st = host.snapshot()
    print("\(st["state"]?.string ?? "?") in \(String(format: "%.1f", Date().timeIntervalSince(t0))) s on \(host.loadedUnits?.label ?? "-") \(st["detail"]?.string ?? "")")
    if let benchSeconds, st["state"]?.string == "ready" {
      let report = try host.benchmark(seconds: benchSeconds, run: BenchmarkRun()) { _ in }
      print(report.text)
    }
    host.close()
    exit(st["state"]?.string == "ready" ? 0 : 1)
  case "prepare":
    let device = option("--device") ?? "coreml"
    let key = option("--cache-key")
    guard args.count == 2 else { usage() }
    let t0 = Date()
    let report = try OnnxPrepare.prepare(
      src: args[0], dst: args[1], forANE: device == "ane", forCoreML: device == "ane" || device == "coreml", cacheKey: key)
    print("prepared in \(String(format: "%.1f", Date().timeIntervalSince(t0))) s: \(report)")
  case "spec":
    guard args.count == 1 else { usage() }
    let meta = try OnnxMeta(path: args[0])
    let (sha, n) = try sha256File(args[0])
    print(try ModelSpec(meta: meta, sha256: sha, nbytes: n).json.encoded)
  default:
    usage()
  }
} catch {
  FileHandle.standardError.write(Data("error: \(error)\n".utf8))
  exit(1)
}
