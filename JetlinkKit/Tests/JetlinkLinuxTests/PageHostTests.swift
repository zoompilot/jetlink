#if os(Linux)
  import Foundation
  import Glibc
  import JetlinkStatusPage
  import JetlinkTestSupport
  import Testing

  @testable import JetlinkLinux

  /// Commands the host would run, answered from a script.
  final class FakeCommands: HostCommands, @unchecked Sendable {
    let ran = Recorded<[String]>()
    let answer: @Sendable ([String]) -> (Int32, String)

    init(_ answer: @escaping @Sendable ([String]) -> (Int32, String) = { _ in (0, "") }) {
      self.answer = answer
    }

    func run(_ argv: [String], timeout: TimeInterval) -> (status: Int32, output: String) {
      ran.append(argv)
      return answer(argv)
    }
  }

  @Suite("The computer behind the web page")
  struct PageHostTests {
    /// An installed Jetson's answers, as install.sh writes them.
    final class Installed {
      let tree = Tree()
      let commands: FakeCommands
      let host: PageHost

      init(jetson: Bool = true, desktopOff: Bool = false, graphical: Bool = true, conf: String = "", commands: FakeCommands = FakeCommands()) {
        tree.write(
          "/etc/jetlink/install.conf",
          """
          JETLINK_REF=latest
          JETLINK_VERSION=v0.7.2
          JETLINK_COMMIT=835ca95
          JETLINK_PLATFORM_NAME=NVIDIA\\ Jetson\\ Orin\\ Nano
          JETLINK_POWER=\(jetson ? "always" : "")
          JETLINK_POWEROFF_WITH_COMMA=1
          JETLINK_DESKTOP_OFF=\(desktopOff ? 1 : 0)
          JETLINK_AUTOSTART=1
          \(conf)

          """)
        tree.write(
          "/etc/jetlink/server.env",
          """
          JETLINK_CACHE_DIR=/mnt/data/jetlink
          JETLINK_SLEEP_AFTER=120
          JETLINK_STATUS_PORT=5600
          JETLINK_POWEROFF="--poweroff"
          JETLINK_JETSON=\(jetson ? 1 : 0)

          """)
        tree.write("/sys/power/mem_sleep", "s2idle [deep]\n")
        tree.makeDirectory("/etc/systemd/system")
        tree.link(
          "/etc/systemd/system/default.target", to: graphical ? "/usr/lib/systemd/system/graphical.target" : "/usr/lib/systemd/system/multi-user.target")
        self.commands = commands
        host = PageHost(root: tree.root, runner: commands)
      }
    }

    @Test("install.sh's printf %q lines read back as written")
    func shellEnv() {
      let env = ShellEnv.parse(
        #"""
        # a comment
        PLAIN=word
        SPACED=NVIDIA\ Jetson\ Orin
        EMPTY=''
        QUOTED="--tensorrt-libs /opt/jetlink/tensorrt/11.3"
        SINGLE='a b'
        DOLLAR=$'line\nnext'
        MIXED=a'b c'"d e"
        TRAILING=value # a comment
        bad line
        =nokey
        """#)
      #expect(env["PLAIN"] == "word" && env["SPACED"] == "NVIDIA Jetson Orin" && env["EMPTY"] == "")
      #expect(env["QUOTED"] == "--tensorrt-libs /opt/jetlink/tensorrt/11.3" && env["SINGLE"] == "a b")
      #expect(env["DOLLAR"] == "line\nnext" && env["MIXED"] == "ab cd e" && env["TRAILING"] == "value")
      #expect(env.count == 8)
    }

    @Test("A Jetson's answers in the installer's words, and only those that apply")
    func info() throws {
      let installed = Installed()
      let info = installed.host.info()
      #expect(info["installed"] as? Bool == true && info["jetson"] as? Bool == true)
      #expect(info["platform"] as? String == "NVIDIA Jetson Orin Nano" && info["version"] as? String == "v0.7.2")
      #expect(info["page_port"] as? Int == 5600 && info["sleep_after"] as? Int == 120 && info["deep_sleep"] as? Bool == true)
      #expect(info["settings"] as? [String: String] == ["power": "always", "comma_poweroff": "yes", "desktop": "on"])
      #expect((info["task"] as? [String: Any])?["state"] as? String == "none")
      // no lock file: this server does not sleep
      #expect((info["awake"] as? [String: Any])?["available"] as? Bool == false)
      #expect(installed.commands.ran.all.isEmpty)
    }

    @Test("No desktop answer for a Jetson that never had one; a PC has its start-up instead")
    func applies() {
      // each kept while asked: its tree goes with it
      let bare = Installed(graphical: false)
      #expect(bare.host.info()["settings"] as? [String: String] == ["power": "always", "comma_poweroff": "yes"])
      let turnedOff = Installed(desktopOff: true, graphical: false)
      #expect(turnedOff.host.info()["settings"] as? [String: String] == ["power": "always", "comma_poweroff": "yes", "desktop": "off"])
      let computer = Installed(jetson: false)
      #expect(computer.host.info()["settings"] as? [String: String] == ["autostart": "yes"])
      let empty = Tree()
      #expect(PageHost(root: empty.root, runner: FakeCommands()).info()["installed"] as? Bool == false)
    }

    @Test("The swap answer where the largest model's build needs swap, with what the installer recorded")
    func swap() throws {
      let made = Installed(
        conf: #"""
          JETLINK_SWAP=on
          JETLINK_SWAP_FILE=/mnt/data/jetlink-swapfile
          JETLINK_SWAP_GB=1
          JETLINK_SWAP_NEED_GB=3
          JETLINK_SWAP_SHORT_GB=0
          """#)
      let info = made.host.info()
      #expect((info["settings"] as? [String: String])?["swap"] == "on")
      #expect(info["swap_facts"] as? [String: Int] == ["need_gb": 3, "file_gb": 1, "short_gb": 0])
      // turned off: the answer stays, and the page offers it back
      let off = Installed(conf: "JETLINK_SWAP=off\nJETLINK_SWAP_FILE=\nJETLINK_SWAP_NEED_GB=3")
      #expect((off.host.info()["settings"] as? [String: String])?["swap"] == "off")
      try off.host.apply(["swap": "on"])
      #expect(off.commands.ran.all.last?.suffix(2) == ["--set", "swap=on"])
      // a 16 GB Jetson needs none, and is not asked
      let big = Installed(conf: "JETLINK_SWAP=on\nJETLINK_SWAP_FILE=\nJETLINK_SWAP_NEED_GB=0")
      #expect((big.host.info()["settings"] as? [String: String])?["swap"] == nil)
      #expect(throws: PageSystemError.self) { try big.host.apply(["swap": "off"]) }
      #expect(throws: PageSystemError.self) { try made.host.apply(["swap": "maybe"]) }
    }

    @Test("The power mode and the clocks, and the installer's recommended answers")
    func performance() throws {
      let installed = Installed(
        conf: #"""
          JETLINK_POWER_MODE=25W
          JETLINK_POWER_MODES=15W\ 25W\ MAXN_SUPER
          JETLINK_CLOCKS=off
          JETLINK_DEFAULTS=power=always\ comma_poweroff=yes\ desktop=off\ swap=on\ power_mode=fastest\ clocks=on
          """#)
      let info = installed.host.info()
      let settings = try #require(info["settings"] as? [String: String])
      #expect(settings["power_mode"] == "25W" && settings["clocks"] == "off")
      // as the page offers them: the fastest's answer is fastest
      #expect(
        info["power_modes"] as? [[String: String]] == [
          ["value": "15W", "name": "15W"], ["value": "25W", "name": "25W"], ["value": "fastest", "name": "MAXN_SUPER"],
        ])
      // only those that apply here: no swap answer on this one
      #expect(
        info["defaults"] as? [String: String] == [
          "power": "always", "comma_poweroff": "yes", "desktop": "off", "power_mode": "fastest", "clocks": "on",
        ])
      try installed.host.apply(["power_mode": "fastest", "clocks": "on"])
      #expect(installed.commands.ran.all.last?.suffix(4) == ["--set", "clocks=on", "--set", "power_mode=fastest"])
      #expect(throws: PageSystemError.self) { try installed.host.apply(["power_mode": "60W"]) }
      #expect(throws: PageSystemError.self) { try installed.host.apply(["clocks": "auto"]) }
      // an installer that does not list the modes takes neither; kept in a
      // let, as its files go with it
      let plain = Installed()
      #expect(plain.host.info()["settings"] as? [String: String] == ["power": "always", "comma_poweroff": "yes", "desktop": "on"])
    }

    @Test("Settings become one installer run, as its own unit, its output beside it")
    func apply() throws {
      let installed = Installed()
      try installed.host.apply(["power": "switched", "comma_poweroff": "no", "desktop": "off"])
      let base = installed.tree.path("/run/jetlink-web/task")
      #expect(
        installed.commands.ran.all.last == [
          "systemd-run", "--unit=jetlink-web-task", "--collect", "--quiet", "--property=Nice=10", "--", "/bin/sh", "-c",
          #""$@" >"$0.log" 2>&1; echo $? >"$0.exit""#, base, "/usr/local/bin/jetlink", "setup",
          "--set", "comma_poweroff=no", "--set", "desktop=off", "--set", "power=switched",
        ])
      #expect(installed.host.task()["kind"] as? String == "settings" && installed.host.task()["state"] as? String == "running")
      // a second run waits for the first
      #expect(throws: PageSystemError.self) { try installed.host.apply(["power": "always"]) }
      // the run ends: its output and status are where the page looks
      installed.tree.write("/run/jetlink-web/task.log", "Applying\r\n\u{1B}[32m✓\u{1B}[0m Jetlink server 0.7.2\rspinning\rStarted\n")
      installed.tree.write("/run/jetlink-web/task.exit", "0\n")
      let done = installed.host.task()
      #expect(done["state"] as? String == "done" && done["exit"] as? Int == 0 && done["finished_at"] is Double)
      #expect(done["lines"] as? [String] == ["Applying", "Started"])
      installed.tree.write("/run/jetlink-web/task.exit", "1\n")
      #expect(installed.host.task()["state"] as? String == "failed")
    }

    @Test("A run with no status left is running while its unit's cgroup is there, and lost after")
    func lost() throws {
      let installed = Installed()
      installed.tree.write("/run/jetlink-web/task.json", #"{"kind":"update","started_at":1000}"#)
      #expect(installed.host.task()["state"] as? String == "lost")
      installed.tree.makeDirectory("/sys/fs/cgroup/system.slice/jetlink-web-task.service")
      #expect(installed.host.task()["state"] as? String == "running")
    }

    @Test("An answer that does not apply here, or a word the installer does not take, runs nothing")
    func refused() {
      let installed = Installed(graphical: false)
      for settings: [String: String] in [["desktop": "off"], ["autostart": "yes"], ["power": "sometimes"], ["comma_poweroff": "true"], [:]] {
        let error = #expect(throws: PageSystemError.self, "\(settings)") { try installed.host.apply(settings) }
        guard case .refused? = error else {
          Issue.record("\(settings) was not refused: \(String(describing: error))")
          continue
        }
      }
      #expect(installed.commands.ran.all.isEmpty)
    }

    @Test("A run that systemd will not start says why and leaves no run behind")
    func startFails() {
      let installed = Installed(
        commands: FakeCommands {
          $0.first == "systemd-run" ? (1, "Failed to start transient service unit: Unit jetlink-web-task.service already exists.\n") : (0, "")
        })
      let error = #expect(throws: PageSystemError.self) { try installed.host.perform(.update, seconds: nil) }
      #expect(error?.description.contains("already exists") == true)
      #expect(installed.host.task()["state"] as? String == "none")
    }

    @Test("An update is jetlink update; restarts go to systemd once the reply is out")
    func actions() throws {
      let installed = Installed()
      _ = try installed.host.perform(.update, seconds: nil)
      #expect(installed.commands.ran.all.last?.suffix(2) == ["/usr/local/bin/jetlink", "update"])
      #expect(installed.host.task()["kind"] as? String == "update")
      _ = try installed.host.perform(.restart, seconds: nil)
      #expect(installed.commands.ran.wait(timeout: 3) { $0.contains(["systemctl", "--no-block", "restart", "jetlink-server"]) })
      _ = try installed.host.perform(.reboot, seconds: nil)
      #expect(installed.commands.ran.wait(timeout: 3) { $0.contains(["systemctl", "--no-block", "reboot"]) })
    }

    @Test("The newest release is what jetlink update --check names")
    func checkUpdate() throws {
      let newer = Installed(commands: FakeCommands { $0.suffix(2) == ["update", "--check"] ? (0, "v0.7.4\n") : (0, "") })
      let reply = try newer.host.perform(.checkUpdate, seconds: nil)
      #expect(reply["latest"] as? String == "v0.7.4" && reply["current"] as? String == "v0.7.2" && reply["newer"] as? Bool == true)
      #expect(reply["url"] as? String == "https://github.com/zoompilot/jetlink/releases/tag/v0.7.4")
      let offline = Installed(commands: FakeCommands { _ in (1, "Could not look up the newest release.\n") })
      #expect(throws: PageSystemError.self) { try offline.host.perform(.checkUpdate, seconds: nil) }
    }

    @Test("Keeping awake is jetlink caffeinate in a unit of its own, one at a time; 0 lets go")
    func keepAwake() throws {
      let installed = Installed()
      #expect(throws: PageSystemError.self) { try installed.host.perform(.keepAwake, seconds: 60) }
      installed.tree.write(Sleeper.awakeLock, "")
      let reply = try installed.host.perform(.keepAwake, seconds: 900)
      let ran = installed.commands.ran.all
      #expect(
        ran.suffix(2) == [
          ["systemctl", "stop", "jetlink-web-awake.service"],
          ["systemd-run", "--unit=jetlink-web-awake", "--collect", "--quiet", "--", "/usr/local/bin/jetlink", "caffeinate", "-t", "900"],
        ])
      #expect((reply["awake"] as? [String: Any])?["available"] as? Bool == true)
      // its until shows while the unit runs
      installed.tree.makeDirectory("/sys/fs/cgroup/system.slice/jetlink-web-awake.service")
      let until = try #require((installed.host.info()["awake"] as? [String: Any])?["until"] as? Int)
      #expect(abs(Double(until) - Date().timeIntervalSince1970 - 900) < 5)
      _ = try installed.host.perform(.keepAwake, seconds: 0)
      #expect(installed.commands.ran.all.last == ["systemctl", "stop", "jetlink-web-awake.service"])
      #expect((installed.host.info()["awake"] as? [String: Any])?["until"] == nil)
      #expect(throws: PageSystemError.self) { try installed.host.perform(.keepAwake, seconds: 13 * 3600) }
    }

    @Test("Only a release newer than this one's is an update")
    func newer() {
      #expect(PageHost.isNewer("v0.7.3", than: "v0.7.2") && PageHost.isNewer("v0.10.0", than: "v0.9.9") && PageHost.isNewer("v1.0.0", than: "v0.99.99"))
      #expect(!PageHost.isNewer("v0.7.2", than: "v0.7.2") && !PageHost.isNewer("v0.7.1", than: "v0.7.2"))
      #expect(!PageHost.isNewer("v0.7.3", than: "main") && !PageHost.isNewer("v0.7.3", than: "local"))
    }

    @Test("A command's status and output, through the daemon's own spawn, cut off at its timeout")
    func spawned() {
      let commands = SpawnedCommands()
      let ok = commands.run(["/bin/sh", "-c", "echo out; echo err >&2; exit 3"], timeout: 5)
      #expect(ok.status == 3 && ok.output.contains("out\n") && ok.output.contains("err\n"))
      let started = Date()
      let slow = commands.run(["/bin/sh", "-c", "sleep 30"], timeout: 0.3)
      #expect(slow.status == -1 && Date().timeIntervalSince(started) < 3)
      #expect(commands.run(["/nonexistent/thing"], timeout: 1).status == 127)
    }
  }
#endif
