import Foundation
import Testing

@testable import Jetlink

struct LogFileWriterTests {
  private func temporaryDirectory() throws -> URL {
    let url = URL(filePath: NSTemporaryDirectory()).appending(path: "jetlink-logs-\(UUID().uuidString.prefix(8))")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }

  @Test func writesLinesWithNewlines() async throws {
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let url = directory.appending(path: "server.log")
    let writer = LogFileWriter(url: url, rotateAtBytes: 1024)
    await writer.append("first line")
    await writer.append("second line")
    await writer.close()
    let text = try String(contentsOf: url, encoding: .utf8)
    #expect(text == "first line\nsecond line\n")
  }

  @Test func rotationKeepsTwoOldFiles() async throws {
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let url = directory.appending(path: "server.log")
    let writer = LogFileWriter(url: url, rotateAtBytes: 1024)
    // Each line is 100 bytes, so every 11 lines forces one rotation.
    let line = String(repeating: "x", count: 99)
    for _ in 0..<40 { await writer.append(line) }
    await writer.close()

    let manager = FileManager.default
    #expect(manager.fileExists(atPath: url.path(percentEncoded: false)))
    #expect(manager.fileExists(atPath: url.path(percentEncoded: false) + ".1"))
    #expect(manager.fileExists(atPath: url.path(percentEncoded: false) + ".2"))
    #expect(!manager.fileExists(atPath: url.path(percentEncoded: false) + ".3"))

    let rotated = try Data(contentsOf: URL(filePath: url.path(percentEncoded: false) + ".1"))
    #expect(rotated.count >= 1024)
    let current = try Data(contentsOf: url)
    #expect(current.count < 1024)
  }

  @Test func anExistingFileIsAppendedTo() async throws {
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let url = directory.appending(path: "server.log")
    try "already here\n".write(to: url, atomically: true, encoding: .utf8)
    let writer = LogFileWriter(url: url, rotateAtBytes: 1024)
    await writer.append("and now this")
    await writer.close()
    let text = try String(contentsOf: url, encoding: .utf8)
    #expect(text == "already here\nand now this\n")
  }
}
