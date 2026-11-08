#if os(Linux)
  import Foundation
  import Glibc
  import JetlinkServer
  import JetlinkStatusPage

  /// Runs a command for the page and waits for it.
  public protocol HostCommands: Sendable {
    /// Its exit status and what it printed, within `timeout`.
    func run(_ argv: [String], timeout: TimeInterval) -> (status: Int32, output: String)
  }

  /// Through `PowerOff.spawn`, so the child gets back the signals the daemon
  /// blocks.
  public struct SpawnedCommands: HostCommands {
    public init() {}

    public func run(_ argv: [String], timeout: TimeInterval) -> (status: Int32, output: String) {
      var ends: [Int32] = [0, 0]
      guard !argv.isEmpty, pipe(&ends) == 0 else { return (127, "cannot run \(argv.first ?? "nothing")") }
      // neither end leaks into another child; the spawn dups the write end
      // onto the child's stdout and stderr
      for fd in ends { _ = fcntl(fd, F_SETFD, FD_CLOEXEC) }
      let pid: pid_t
      do {
        pid = try PowerOff.spawn(argv[0], Array(argv.dropFirst()), output: ends[1])
      } catch {
        close(ends[0])
        close(ends[1])
        return (127, "\(error)")
      }
      close(ends[1])
      var output = [UInt8]()
      var chunk = [UInt8](repeating: 0, count: 4096)
      let deadline = Date(timeIntervalSinceNow: timeout)
      while true {
        let left = deadline.timeIntervalSinceNow
        if left <= 0 {
          kill(pid, SIGKILL)
          break
        }
        var poller = pollfd(fd: ends[0], events: Int16(POLLIN), revents: 0)
        let ready = poll(&poller, 1, Int32(min(left, 1) * 1000) + 1)
        if ready < 0 && errno != EINTR { break }
        if ready <= 0 { continue }
        let n = read(ends[0], &chunk, chunk.count)
        if n < 0 && errno == EINTR { continue }
        if n <= 0 { break }
        output += chunk[..<n]
      }
      close(ends[0])
      return (PowerOff.exitStatus(of: pid), String(decoding: output, as: UTF8.self))
    }
  }

  /// An installed Jetlink, as install.sh left it: answers in
  /// /etc/jetlink/install.conf and server.env, the `jetlink` command, systemd.
  /// What restarts the server, a settings run or an update, runs as a
  /// transient unit, so the restart cannot take it down with it, at nice 10
  /// under the frame path; so does keeping the box awake, which outlives a
  /// restart the same way.
  public final class PageHost: PageSystem, @unchecked Sendable {
    static let taskUnit = "jetlink-web-task"
    static let awakeUnit = "jetlink-web-awake"
    static let longestAwake = 12 * 3600
    /// What install.sh's `--set` takes for each answer.
    static let choices: [String: Set<String>] = [
      "power": ["always", "switched"], "comma_poweroff": ["yes", "no"], "desktop": ["on", "off"], "swap": ["on", "off"],
      "clocks": ["on", "off"], "autostart": ["yes", "no"],
    ]
    static let colour = try! NSRegularExpression(pattern: "\u{1B}\\[[0-9;]*[A-Za-z]")

    let root: HostRoot
    let jetlink: String
    private let runner: any HostCommands
    private let log = ServerLog(category: "page")

    public init(root: HostRoot = .system, runner: any HostCommands = SpawnedCommands(), jetlink: String = "/usr/local/bin/jetlink") {
      self.root = root
      self.runner = runner
      self.jetlink = jetlink
    }

    /// The runs' scripts, output and exit status: tmpfs, gone at a reboot.
    private var run: String { root.path("/run/jetlink-web") }

    // MARK: what is set

    struct Answers {
      let conf: [String: String]
      let env: [String: String]

      var jetson: Bool { env["JETLINK_JETSON"] == "1" }
      var sleepAfter: Int { Int(env["JETLINK_SLEEP_AFTER"] ?? "") ?? 0 }
    }

    func answers() -> Answers? {
      guard let conf = ShellEnv.read(root.path("/etc/jetlink/install.conf")) else { return nil }
      return Answers(conf: conf, env: ShellEnv.read(root.path("/etc/jetlink/server.env")) ?? [:])
    }

    /// The answers that apply here, in the installer's words: the questions
    /// it would ask this computer.
    func settings(_ answers: Answers) -> [String: String] {
      let conf = answers.conf
      guard answers.jetson else { return ["autostart": conf["JETLINK_AUTOSTART"] == "0" ? "no" : "yes"] }
      var settings = ["power": conf["JETLINK_POWER"] ?? "", "comma_poweroff": conf["JETLINK_POWEROFF_WITH_COMMA"] == "1" ? "yes" : "no"]
      // the desktop question only for one that starts it, or that Jetlink turned off
      let off = conf["JETLINK_DESKTOP_OFF"] == "1"
      if off || bootsToDesktop() { settings["desktop"] = off ? "off" : "on" }
      // the swap for one whose largest-model build needs it, or that has Jetlink's file
      if (Int(conf["JETLINK_SWAP_NEED_GB"] ?? "") ?? 0) > 0 || !(conf["JETLINK_SWAP_FILE"] ?? "").isEmpty {
        settings["swap"] = conf["JETLINK_SWAP"] == "off" ? "off" : "on"
      }
      // an installer that lists the power modes also takes these two
      if !powerModes(answers).isEmpty {
        settings["power_mode"] = conf["JETLINK_POWER_MODE"] ?? "fastest"
        settings["clocks"] = conf["JETLINK_CLOCKS"] == "off" ? "off" : "on"
      }
      return settings
    }

    /// The Jetson's power modes, slowest first, as the installer found them:
    /// the last is the fastest.
    func powerModes(_ answers: Answers) -> [String] {
      (answers.conf["JETLINK_POWER_MODES"] ?? "").split(separator: " ").map(String.init)
    }

    /// The installer's recommended answers (`power=always swap=on`), as it
    /// recorded them.
    func defaults(_ answers: Answers) -> [String: String] {
      let pairs = (answers.conf["JETLINK_DEFAULTS"] ?? "").split(separator: " ").map { $0.split(separator: "=", maxSplits: 1).map(String.init) }
      return Dictionary(pairs.filter { $0.count == 2 }.map { ($0[0], $0[1]) }, uniquingKeysWith: { first, _ in first })
    }

    /// What the swap answer stands on, as the installer recorded it: what
    /// building the largest model needs, Jetlink's file, and what did not fit
    /// on the disk.
    func swapFacts(_ answers: Answers) -> [String: Int] {
      ["need_gb": "JETLINK_SWAP_NEED_GB", "file_gb": "JETLINK_SWAP_GB", "short_gb": "JETLINK_SWAP_SHORT_GB"].mapValues {
        Int(answers.conf[$0] ?? "") ?? 0
      }
    }

    /// `systemctl get-default`, without running it: the link systemd follows.
    private func bootsToDesktop() -> Bool {
      for link in ["/etc/systemd/system/default.target", "/usr/lib/systemd/system/default.target", "/lib/systemd/system/default.target"] {
        if let target = try? FileManager.default.destinationOfSymbolicLink(atPath: root.path(link)) {
          return (target as NSString).lastPathComponent == "graphical.target"
        }
      }
      return false
    }

    public func info() -> [String: Any] {
      guard let answers = answers() else { return ["installed": false] }
      let modes = Sysfs.read(root.path("/sys/power/mem_sleep")) ?? ""
      let current = settings(answers)
      let modeNames = powerModes(answers)
      return [
        "installed": true,
        "jetson": answers.jetson,
        "platform": answers.conf["JETLINK_PLATFORM_NAME"] ?? "",
        "version": answers.conf["JETLINK_VERSION"] ?? "",
        "ref": answers.conf["JETLINK_REF"] ?? "",
        "commit": answers.conf["JETLINK_COMMIT"] ?? "",
        "server_version": answers.env["JETLINK_SERVER_VERSION"] ?? "",
        "settings": current,
        "swap_facts": swapFacts(answers),
        // as the page offers them: the fastest's answer is `fastest`
        "power_modes": modeNames.map { ["value": $0 == modeNames.last ? "fastest" : $0, "name": $0] },
        "defaults": defaults(answers).filter { current[$0.key] != nil },
        "deep_sleep": modes.split(separator: " ").contains { $0 == "deep" || $0 == "[deep]" },
        "sleep_after": answers.sleepAfter,
        "page_port": Int(answers.env["JETLINK_STATUS_PORT"] ?? "") ?? 0,
        "cache_dir": answers.env["JETLINK_CACHE_DIR"] ?? "",
        "awake": awake(answers),
        "task": task(),
      ]
    }

    // MARK: settings

    /// One installer run with these answers. An answer that does not apply
    /// here, or a value the installer does not take, is refused before
    /// anything runs, so a stale page cannot believe it changed something.
    public func apply(_ settings: [String: String]) throws(PageSystemError) {
      guard let answers = answers() else { throw .refused("Jetlink is not installed here, so there is nothing to set.") }
      let here = self.settings(answers)
      var sets: [String] = []
      for (key, value) in settings.sorted(by: { $0.key < $1.key }) {
        guard here[key] != nil else { throw .refused("\(key) is not a setting on this computer.") }
        let allowed = key == "power_mode" ? Set(["fastest"] + powerModes(answers)) : PageHost.choices[key]
        guard allowed?.contains(value) == true else { throw .refused("\(key) cannot be \(value).") }
        sets.append("\(key)=\(value)")
      }
      guard !sets.isEmpty else { throw .refused("Nothing to change.") }
      try startTask(kind: "settings", [jetlink, "setup"] + sets.flatMap { ["--set", $0] })
      log.info("web page: applying \(sets.joined(separator: " "))")
    }

    // MARK: actions

    public func perform(_ action: PageAction, seconds: Int?) throws(PageSystemError) -> [String: Any] {
      switch action {
      case .restart, .reboot, .poweroff:
        guard answers() != nil else { throw .refused("Jetlink is not installed here; restart it where it runs.") }
        let command = action == .restart ? ["systemctl", "--no-block", "restart", "jetlink-server"] : ["systemctl", "--no-block", action.rawValue]
        log.info("web page: \(action.rawValue)")
        // once the reply has gone out
        let runner = self.runner
        Thread.detachNewThread {
          Thread.sleep(forTimeInterval: 0.5)
          _ = runner.run(command, timeout: 10)
        }
        return [:]
      case .update:
        guard answers() != nil else { throw .refused("Jetlink is not installed here, so there is nothing to update.") }
        try startTask(kind: "update", [jetlink, "update"])
        log.info("web page: updating")
        return [:]
      case .checkUpdate:
        return try checkUpdate()
      case .keepAwake:
        return try keepAwake(seconds ?? 0)
      }
    }

    /// The newest release, as `jetlink update` finds it, and whether it is
    /// newer than what runs.
    private func checkUpdate() throws(PageSystemError) -> [String: Any] {
      let current = answers()?.conf["JETLINK_VERSION"] ?? ""
      let result = runner.run([jetlink, "update", "--check"], timeout: 45)
      let latest = result.output.split(whereSeparator: \.isNewline).last.map(String.init) ?? ""
      guard result.status == 0, latest.hasPrefix("v") else {
        throw .failed("Could not find the newest release. Is this \(answers()?.jetson == false ? "computer" : "Jetson") online?")
      }
      return [
        "latest": latest, "current": current, "newer": PageHost.isNewer(latest, than: current),
        "url": "https://github.com/zoompilot/jetlink/releases/tag/\(latest)",
      ]
    }

    /// vX.Y.Z against vX.Y.Z; anything else (main, a checkout) is never older.
    static func isNewer(_ tag: String, than current: String) -> Bool {
      func numbers(_ text: String) -> [Int]? {
        guard text.hasPrefix("v") else { return nil }
        let parts = text.dropFirst().split(separator: ".").compactMap { Int($0) }
        return parts.count == 3 ? parts : nil
      }
      guard let new = numbers(tag), let old = numbers(current) else { return false }
      return old.lexicographicallyPrecedes(new)
    }

    // MARK: runs

    /// The one run there may be at a time, as its own unit; its output and
    /// exit status land beside `base` for whichever server process is up when
    /// the page asks.
    private func startTask(kind: String, _ argv: [String]) throws(PageSystemError) {
      if (task()["state"] as? String) == "running" { throw .refused("A run is going already. Wait for it to finish.") }
      let base = run + "/task"
      do {
        try FileManager.default.createDirectory(atPath: run, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        for suffix in [".log", ".exit"] { try? FileManager.default.removeItem(atPath: base + suffix) }
        let meta = try JSONSerialization.data(withJSONObject: ["kind": kind, "started_at": Date().timeIntervalSince1970])
        try meta.write(to: URL(fileURLWithPath: base + ".json"), options: .atomic)
      } catch {
        throw .failed("Could not prepare the run: \(error.localizedDescription)")
      }
      let script = #""$@" >"$0.log" 2>&1; echo $? >"$0.exit""#
      let result = runner.run(
        ["systemd-run", "--unit=\(PageHost.taskUnit)", "--collect", "--quiet", "--property=Nice=10", "--", "/bin/sh", "-c", script, base] + argv,
        timeout: 20)
      guard result.status == 0 else {
        try? FileManager.default.removeItem(atPath: base + ".json")
        let why = result.output.trimmingCharacters(in: .whitespacesAndNewlines)
        throw .failed("Could not start the run\(why.isEmpty ? "" : ": " + why)")
      }
    }

    /// Whether a unit of ours runs: its cgroup, rather than a systemctl.
    private func active(_ unit: String) -> Bool {
      root.exists("/sys/fs/cgroup/system.slice/\(unit).service")
    }

    public func task() -> [String: Any] {
      let base = run + "/task"
      guard let data = FileManager.default.contents(atPath: base + ".json"),
        var out = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
      else { return ["state": "none"] }
      let started = out["started_at"] as? Double ?? 0
      if let code = Sysfs.readInt(base + ".exit") {
        out["state"] = code == 0 ? "done" : "failed"
        out["exit"] = code
        var info = stat()
        if stat(base + ".exit", &info) == 0 { out["finished_at"] = Double(info.st_mtim.tv_sec) }
      } else if Date().timeIntervalSince1970 - started < 15 || active(PageHost.taskUnit) {
        out["state"] = "running"
      } else {
        out["state"] = "lost"
      }
      out["lines"] = PageHost.tail(base + ".log", lines: 40)
      return out
    }

    /// The last lines of a run's output, colour codes removed.
    static func tail(_ path: String, lines: Int) -> [String] {
      guard let handle = FileHandle(forReadingAtPath: path) else { return [] }
      defer { try? handle.close() }
      let size = (try? handle.seekToEnd()) ?? 0
      try? handle.seek(toOffset: size > 32768 ? size - 32768 : 0)
      let text = String(decoding: handle.readDataToEndOfFile(), as: UTF8.self)
      let plain = colour.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: "")
      // CRLF is one Character to Swift, so it never splits at "\n"; a step's
      // spinner redraws its line with carriage returns
      return plain.replacingOccurrences(of: "\r\n", with: "\n").split(separator: "\n").compactMap { line in
        let last = String(line.split(separator: "\r").last ?? "")
        return last.trimmingCharacters(in: .whitespaces).isEmpty ? nil : last
      }.suffix(lines).map { $0 }
    }

    // MARK: keeping awake

    /// `jetlink caffeinate -t` as a unit of its own until `seconds` from now;
    /// 0 lets go.
    private func keepAwake(_ seconds: Int) throws(PageSystemError) -> [String: Any] {
      guard (0...PageHost.longestAwake).contains(seconds) else { throw .refused("Keep awake for up to 12 hours.") }
      guard let answers = answers() else { throw .refused("Jetlink is not installed here.") }
      let until = run + "/awake-until"
      // one at a time: a new hold ends the last
      _ = runner.run(["systemctl", "stop", "\(PageHost.awakeUnit).service"], timeout: 10)
      try? FileManager.default.removeItem(atPath: until)
      if seconds > 0 {
        guard root.exists(Sleeper.awakeLock) else { throw .refused("This server does not sleep, so there is nothing to hold.") }
        let result = runner.run(
          ["systemd-run", "--unit=\(PageHost.awakeUnit)", "--collect", "--quiet", "--", jetlink, "caffeinate", "-t", "\(seconds)"], timeout: 20)
        guard result.status == 0 else { throw .failed("Could not keep it awake: \(result.output.trimmingCharacters(in: .whitespacesAndNewlines))") }
        try? FileManager.default.createDirectory(atPath: run, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        try? "\(Int(Date().timeIntervalSince1970) + seconds)\n".write(toFile: until, atomically: true, encoding: .utf8)
        log.info("web page: holding the Jetson awake for \(seconds / 60) min")
      } else {
        log.info("web page: no longer holding the Jetson awake")
      }
      return ["awake": awake(answers)]
    }

    private func awake(_ answers: Answers) -> [String: Any] {
      var out: [String: Any] = ["available": answers.sleepAfter > 0 && root.exists(Sleeper.awakeLock)]
      if let until = Sysfs.readInt(run + "/awake-until"), Double(until) > Date().timeIntervalSince1970, active(PageHost.awakeUnit) {
        out["until"] = until
      }
      return out
    }
  }

  /// `KEY=VALUE` lines as install.sh writes them with printf %q: bare words
  /// with backslash escapes, '' and "" quotes, $'' for control characters.
  enum ShellEnv {
    static func read(_ path: String) -> [String: String]? {
      guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { return nil }
      return parse(text)
    }

    static func parse(_ text: String) -> [String: String] {
      var out: [String: String] = [:]
      for raw in text.split(whereSeparator: \.isNewline) {
        let line = raw.trimmingCharacters(in: .whitespaces)
        guard !line.hasPrefix("#"), let equals = line.firstIndex(of: "=") else { continue }
        let key = String(line[..<equals])
        guard !key.isEmpty, key.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "_" }) else { continue }
        out[key] = value(line[line.index(after: equals)...])
      }
      return out
    }

    static func value(_ text: Substring) -> String {
      var out = ""
      var index = text.startIndex
      func next() -> Character? {
        guard index < text.endIndex else { return nil }
        defer { index = text.index(after: index) }
        return text[index]
      }
      while let c = next() {
        switch c {
        case "\\":
          if let escaped = next() { out.append(escaped) }
        case "'":
          while let q = next(), q != "'" { out.append(q) }
        case "\"":
          while let q = next(), q != "\"" {
            if q == "\\", let escaped = next() { out.append(escaped) } else { out.append(q) }
          }
        case "$" where index < text.endIndex && text[index] == "'":
          _ = next()
          while let q = next(), q != "'" {
            guard q == "\\", let escaped = next() else {
              out.append(q)
              continue
            }
            switch escaped {
            case "n": out.append("\n")
            case "t": out.append("\t")
            case "r": out.append("\r")
            case "e", "E": out.append("\u{1B}")
            default: out.append(escaped)
            }
          }
        case " ", "\t", "#":
          return out
        default:
          out.append(c)
        }
      }
      return out
    }
  }
#endif
