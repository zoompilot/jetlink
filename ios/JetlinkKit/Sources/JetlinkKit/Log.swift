// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// Logging: os.Logger for Console.app, plus one sink the app or the command
// line installs to show the same lines. Nothing on the frame path logs.

import Foundation
import os

public enum LogLevel: Int, Sendable, Comparable, CustomStringConvertible {
  case debug = 10, info = 20, warning = 30, error = 40

  public static func < (a: LogLevel, b: LogLevel) -> Bool { a.rawValue < b.rawValue }

  public var description: String {
    switch self {
    case .debug: "DEBUG"
    case .info: "INFO"
    case .warning: "WARNING"
    case .error: "ERROR"
    }
  }
}

public struct LogLine: Sendable, Identifiable {
  public let id: UInt64
  public let date: Date
  public let level: LogLevel
  public let category: String
  public let message: String
}

public final class Log: @unchecked Sendable {
  public static let shared = Log()

  private let lock = NSLock()
  private var sink: (@Sendable (LogLine) -> Void)?
  private var minimum: LogLevel = .info
  private var counter: UInt64 = 0
  private let os = Logger(subsystem: "io.zoompilot.jetlink", category: "jetlink")

  /// Where every line at or above `minimum` goes, besides os_log.
  public func setSink(minimum: LogLevel = .info, _ sink: (@Sendable (LogLine) -> Void)?) {
    lock.lock()
    self.sink = sink
    self.minimum = minimum
    lock.unlock()
  }

  public func log(_ level: LogLevel, _ category: String, _ message: String) {
    switch level {
    case .debug: os.debug("\(category, privacy: .public): \(message, privacy: .public)")
    case .info: os.info("\(category, privacy: .public): \(message, privacy: .public)")
    case .warning: os.warning("\(category, privacy: .public): \(message, privacy: .public)")
    case .error: os.error("\(category, privacy: .public): \(message, privacy: .public)")
    }
    lock.lock()
    guard level >= minimum, let sink else {
      lock.unlock()
      return
    }
    counter += 1
    let line = LogLine(id: counter, date: Date(), level: level, category: category, message: message)
    lock.unlock()
    sink(line)
  }
}

/// One logger per area, as the Python modules each have theirs.
public struct JLogger: Sendable {
  let category: String
  public init(_ category: String) { self.category = category }
  public func debug(_ m: @autoclosure () -> String) { Log.shared.log(.debug, category, m()) }
  public func info(_ m: @autoclosure () -> String) { Log.shared.log(.info, category, m()) }
  public func warning(_ m: @autoclosure () -> String) { Log.shared.log(.warning, category, m()) }
  public func error(_ m: @autoclosure () -> String) { Log.shared.log(.error, category, m()) }
}
