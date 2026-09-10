import Foundation
import os

/// Tees the server's output to ~/Library/Logs/Jetlink/server.log, rotating at
/// 20 MB and keeping server.log.1 and server.log.2.
actor LogFileWriter {
  static let defaultRotateAtBytes = 20 * 1024 * 1024

  private let url: URL
  private let rotateAtBytes: Int
  private var handle: FileHandle?
  private var written: Int = 0
  private let log = Logger(subsystem: "io.zoompilot.jetlink", category: "logfile")

  init(url: URL = AppSettings.logFileURL, rotateAtBytes: Int = LogFileWriter.defaultRotateAtBytes) {
    self.url = url
    self.rotateAtBytes = rotateAtBytes
  }

  func append(_ line: String) {
    guard let handle = openIfNeeded() else { return }
    let data = Data((line + "\n").utf8)
    do {
      try handle.write(contentsOf: data)
      written += data.count
    } catch {
      log.error("could not write to the log file: \(error.localizedDescription, privacy: .public)")
      return
    }
    if written >= rotateAtBytes { rotate() }
  }

  func close() {
    try? handle?.close()
    handle = nil
  }

  /// The bytes written since the last rotation, for the tests.
  var bytesSinceRotation: Int { written }

  private func openIfNeeded() -> FileHandle? {
    if let handle { return handle }
    let directory = url.deletingLastPathComponent()
    do {
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
      if !FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) {
        FileManager.default.createFile(atPath: url.path(percentEncoded: false), contents: nil)
      }
      let opened = try FileHandle(forWritingTo: url)
      written = Int(try opened.seekToEnd())
      handle = opened
      return opened
    } catch {
      log.error("could not open the log file: \(error.localizedDescription, privacy: .public)")
      return nil
    }
  }

  private func rotate() {
    close()
    let manager = FileManager.default
    let path = url.path(percentEncoded: false)
    let first = path + ".1"
    let second = path + ".2"
    try? manager.removeItem(atPath: second)
    if manager.fileExists(atPath: first) {
      try? manager.moveItem(atPath: first, toPath: second)
    }
    if manager.fileExists(atPath: path) {
      try? manager.moveItem(atPath: path, toPath: first)
    }
    written = 0
    _ = openIfNeeded()
  }
}
