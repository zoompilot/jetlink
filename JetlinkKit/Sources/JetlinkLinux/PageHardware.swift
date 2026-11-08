#if os(Linux)
  import Foundation
  import Glibc
  import JetlinkKit
  import JetlinkServer
  import JetlinkStatusPage

  /// The status page's hardware panel on a Jetson or a Linux PC, read the way
  /// jtop reads it: procfs, sysfs, and the GPU's telemetry the server already
  /// reads. Only the page's sampler thread calls it, while a page is open.
  /// Paths are found at the first call; a sample is then a few dozen small
  /// reads.
  public final class PageHardware: PageHardwareSource, @unchecked Sendable {
    let root: HostRoot
    let cache: URL?
    private let gpu: GPUTelemetry?
    private var found: Found?
    private var previous: [Int: (total: UInt64, idle: UInt64)] = [:]

    /// Everything a sample reads, found once.
    struct Found {
      let zones: [(name: String, path: String)]
      let ina3221: String?
      let fanRPM: String?
      let fanPWM: String?
      let cpus: [Int]
      let modes: [Int: String]
      /// whose lowest and highest clocks meet while `JetsonClocks` holds them
      let gpuDevfreq: String?
    }

    /// `cache` is the models folder, whose disk the panel shows; `gpu` the
    /// telemetry the server reads (`LinuxHost.telemetry`).
    public convenience init(cache: URL, gpu: GPUTelemetry?) {
      self.init(root: .system, cache: cache, gpu: gpu)
    }

    init(root: HostRoot, cache: URL?, gpu: GPUTelemetry?) {
      self.root = root
      self.cache = cache
      self.gpu = gpu
    }

    private var tegra: Bool { gpu?.tegra ?? false }

    private func paths() -> Found {
      if let found { return found }
      var zones = ThermalZone.all(root).filter { $0.celsius() != nil }.map { ($0.type, $0.tempPath) }
      if zones.isEmpty {
        // A desktop without ACPI zones: its CPU package's sensor.
        for chip in ["coretemp", "k10temp", "zenpower", "cpu_thermal"] {
          if let directory = hwmon(named: chip, root), Sysfs.readInt("\(directory)/temp1_input") != nil {
            zones = [("cpu", "\(directory)/temp1_input")]
            break
          }
        }
      }
      let made = Found(
        zones: zones, ina3221: hwmon(named: "ina3221", root), fanRPM: hwmon(named: "pwm_tach", root).map { "\($0)/rpm" },
        fanPWM: hwmon(named: "pwmfan", root).map { "\($0)/pwm1" }, cpus: PageHardware.cpuList(Sysfs.read(root.path("/sys/devices/system/cpu/possible"))),
        modes: JetsonPower.modes(Sysfs.read(root.path("/etc/nvpmodel.conf"))),
        gpuDevfreq: gpuDevfreq(root))
      found = made
      return made
    }

    // MARK: once

    public func host() -> [String: Any] {
      var host: [String: Any] = ["jetson": tegra]
      host["hostname"] = Sysfs.read(root.path("/proc/sys/kernel/hostname"))
      host["kernel"] = Sysfs.read(root.path("/proc/sys/kernel/osrelease"))
      let model = Sysfs.read(root.path("/proc/device-tree/model"))?.trimmingCharacters(in: CharacterSet(charactersIn: "\0"))
        .replacingOccurrences(of: " Engineering Reference Developer Kit", with: "")
      host["board"] = model.flatMap { $0.isEmpty ? nil : $0 } ?? PageHardware.field("model name", in: Sysfs.read(root.path("/proc/cpuinfo")), separator: ":")
      let jetpack = tegra ? PageHardware.jetpack(Sysfs.read(root.path("/etc/nv_tegra_release"))) : nil
      let pretty = PageHardware.field("PRETTY_NAME", in: Sysfs.read(root.path("/etc/os-release")), separator: "=")
      host["os"] = jetpack ?? pretty?.trimmingCharacters(in: CharacterSet(charactersIn: "\""))
      host["gpu"] = gpu?.name ?? (tegra ? PageHardware.tegraGPU(Sysfs.read(root.path("/proc/device-tree/compatible"))) : nil)
      return host
    }

    /// "JetPack 7 (Jetson Linux 39.2.1)" from nv_tegra_release's first line,
    /// named as install.sh names it.
    static func jetpack(_ release: String?) -> String? {
      // "# R39 (release), REVISION: 2.1, GCID: ..."
      guard let line = release?.split(separator: "\n").first else { return nil }
      let parts = line.drop { $0 == "#" || $0 == " " }.split(separator: ",")
      guard let first = parts.first?.split(separator: " ").first, first.hasPrefix("R"), let major = Int(first.dropFirst()),
        let revision = parts.first(where: { $0.contains("REVISION") })?.split(separator: ":").last?.trimmingCharacters(in: .whitespaces),
        !revision.isEmpty
      else { return nil }
      let linux = "Jetson Linux \(major).\(revision)"
      let jetpack = [35: "JetPack 5", 36: "JetPack 6", 38: "JetPack 7", 39: "JetPack 7"][major]
      return jetpack.map { "\($0) (\(linux))" } ?? linux
    }

    /// The GPU's family from the SoC in the device tree: an Orin's is
    /// tegra234.
    static func tegraGPU(_ compatible: String?) -> String {
      let soc = compatible ?? ""
      if soc.contains("tegra234") { return "Orin" }
      if soc.contains("tegra264") { return "Thor" }
      return "Tegra"
    }

    /// "0-3,5" as [0, 1, 2, 3, 5].
    static func cpuList(_ spec: String?) -> [Int] {
      (spec ?? "").split(separator: ",").flatMap { part -> [Int] in
        let ends = part.split(separator: "-").map { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard let low = ends.first ?? nil else { return [] }
        let high = ends.count > 1 ? ends[1] ?? low : low
        return high >= low ? Array(low...high) : []
      }
    }

    /// The value after `key` and `separator` on the first line that starts
    /// with `key`.
    static func field(_ key: String, in text: String?, separator: Character) -> String? {
      for line in (text ?? "").split(separator: "\n") where line.hasPrefix(key) {
        if let at = line.firstIndex(of: separator) {
          return line[line.index(after: at)...].trimmingCharacters(in: .whitespaces)
        }
      }
      return nil
    }

    // MARK: every sample

    public func reset() {
      previous = [:]
    }

    public func sample() -> [String: Any] {
      let found = paths()
      var sample: [String: Any] = ["cpu": cpu(found), "mem": memory(), "temps": temperatures(found)]
      sample["gpu"] = gpuSample()
      if let directory = found.ina3221 {
        // Channel 1 is VDD_IN on an Orin: the whole board's input.
        let rails = (1...3).compactMap { channel -> [String: Any]? in
          guard let label = Sysfs.read("\(directory)/in\(channel)_label"), !label.isEmpty else { return nil }
          let mv = Sysfs.readInt("\(directory)/in\(channel)_input") ?? 0
          let ma = Sysfs.readInt("\(directory)/curr\(channel)_input") ?? 0
          return ["name": label, "w": pythonRound(Double(mv * ma) / 1e6, 2)]
        }
        sample["power"] = ["rails": rails]
      }
      let rpm = found.fanRPM.flatMap { Sysfs.readInt($0) }
      let pwm = found.fanPWM.flatMap { Sysfs.readInt($0) }
      if rpm != nil || pwm != nil {
        var fan: [String: Any] = [:]
        fan["rpm"] = rpm
        fan["pct"] = pwm.map { Int((Double($0) * 100 / 255).rounded()) }
        sample["fan"] = fan
      }
      sample["disk"] = disk()
      if let mode = JetsonPower.modeID(status: Sysfs.read(root.path("/var/lib/nvpmodel/status"))) {
        sample["mode"] = found.modes[mode] ?? "mode \(mode)"
      }
      if let gpu = found.gpuDevfreq, let low = Sysfs.readInt("\(gpu)/min_freq"), let high = Sysfs.readInt("\(gpu)/max_freq") {
        sample["clocks_held"] = low == high
      }
      sample["uptime"] = Sysfs.read(root.path("/proc/uptime"))?.split(separator: " ").first.flatMap { Double($0) }
      let load = Sysfs.read(root.path("/proc/loadavg"))?.split(separator: " ").prefix(3).compactMap { Double($0) }
      if let load, load.count == 3 { sample["load"] = load }
      return sample
    }

    /// Per core: online, load over the time since the last sample (none on
    /// the first), and clock. An offline core, as a Jetson's power modes
    /// leave some, is only that.
    private func cpu(_ found: Found) -> [[String: Any]] {
      var now: [Int: (total: UInt64, idle: UInt64)] = [:]
      for line in (Sysfs.read(root.path("/proc/stat")) ?? "").split(separator: "\n") where line.hasPrefix("cpu") {
        let fields = line.split(separator: " ")
        guard let core = Int(fields[0].dropFirst(3)) else { continue }
        // user nice system idle iowait irq softirq steal; guest is in user
        let values = fields.dropFirst().prefix(8).compactMap { UInt64($0) }
        guard values.count >= 4 else { continue }
        now[core] = (values.reduce(0, +), values[3] + (values.count > 4 ? values[4] : 0))
      }
      let online = Sysfs.read(root.path("/sys/devices/system/cpu/online")).map { Set(PageHardware.cpuList($0)) }
      let cpus = found.cpus.isEmpty ? now.keys.sorted() : found.cpus
      let out = cpus.map { core -> [String: Any] in
        guard online?.contains(core) ?? true, let counters = now[core] else { return ["online": false] }
        var entry: [String: Any] = ["online": true]
        if let before = previous[core], counters.total > before.total {
          let busy = 1 - Double(counters.idle &- before.idle) / Double(counters.total - before.total)
          entry["load"] = pythonRound(min(max(busy, 0), 1) * 100, 1)
        }
        entry["mhz"] = Sysfs.readInt(root.path("/sys/devices/system/cpu/cpu\(core)/cpufreq/scaling_cur_freq")).map { $0 / 1000 }
        return entry
      }
      previous = now
      return out
    }

    private func memory() -> [String: Any] {
      let bytes = Platform.meminfo(root)
      let total = bytes["MemTotal"] ?? 0
      let swap = bytes["SwapTotal"] ?? 0
      return ["total": total, "used": total - (bytes["MemAvailable"] ?? total), "swap_total": swap, "swap_used": swap - (bytes["SwapFree"] ?? swap)]
    }

    /// A Jetson's temperatures, rails and fan have panels of their own, and
    /// its GPU's memory is the system's.
    private func gpuSample() -> [String: Any]? {
      guard let gpu else { return nil }
      let read = gpu.read()
      var sample: [String: Any] = ["load": read["gpu_load_pct"] ?? 0, "mhz": read["gpu_clock_mhz"] ?? 0]
      if !gpu.tegra {
        sample["temp"] = read["temp_c"] ?? 0
        sample["power_w"] = read["power_w"] ?? 0
        sample["power_limit_w"] = read["power_limit_w"] ?? 0
        sample["fan_pct"] = read["fan_pct"] ?? 0
      }
      return sample
    }

    private func temperatures(_ found: Found) -> [[String: Any]] {
      found.zones.compactMap { zone in
        guard let milli = Sysfs.readInt(zone.path), milli > -40_000, milli < 150_000 else { return nil }
        return ["name": zone.name, "c": pythonRound(Double(milli) / 1000, 1)]
      }
    }

    private func disk() -> [String: Any]? {
      guard let cache else { return nil }
      var stat = statvfs()
      guard statvfs(cache.path, &stat) == 0 else { return nil }
      let total = Int(stat.f_blocks) * Int(stat.f_frsize)
      return ["path": cache.path, "total": total, "used": total - Int(stat.f_bavail) * Int(stat.f_frsize)]
    }
  }
#endif
