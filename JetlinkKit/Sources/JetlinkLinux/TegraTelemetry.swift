#if os(Linux)
  import Foundation
  import JetlinkKit

  /// One of the kernel's thermal zones: its type ("tj-thermal") and where
  /// its temperature is.
  public struct ThermalZone: Sendable, Equatable {
    public let zone: String
    public let type: String
    public let tempPath: String

    /// Degrees C, or nil when the zone has no reading: an Orin's cv* zones
    /// have no sensor wired up and answer ENODATA.
    public func celsius() -> Double? {
      Sysfs.readInt(tempPath).map { Double($0) / 1000 }
    }

    /// Every zone, in `sorted(glob)` order.
    public static func all(_ root: HostRoot = .system) -> [ThermalZone] {
      let base = "/sys/devices/virtual/thermal"
      return root.list(base).filter { $0.hasPrefix("thermal_zone") }.compactMap { zone in
        guard let type = Sysfs.read(root.path("\(base)/\(zone)/type")), !type.isEmpty else { return nil }
        return ThermalZone(zone: zone, type: type, tempPath: root.path("\(base)/\(zone)/temp"))
      }
    }
  }

  /// A hwmon device's directory by its driver's name, the first in
  /// `sorted(glob)` order: ina3221 (the power monitor), pwm_tach (the fan).
  public func hwmon(named name: String, _ root: HostRoot = .system) -> String? {
    root.list("/sys/class/hwmon").lazy.map { root.path("/sys/class/hwmon/\($0)") }.first { Sysfs.read("\($0)/name") == name }
  }

  /// A Jetson's GPU devfreq node (`/sys/class/devfreq/17000000.gpu` on an
  /// Orin): its clock now (`cur_freq`) and the lowest and highest it may run
  /// at (`min_freq`, `max_freq`).
  public func gpuDevfreq(_ root: HostRoot = .system) -> String? {
    root.list("/sys/class/devfreq").first { $0.hasSuffix(".gpu") }.map { root.path("/sys/class/devfreq/\($0)") }
  }

  /// A Jetson's health from Tegra sysfs, in the keys the comma logs as
  /// `jetlinkTelemetry` (Python's `telemetry.Telemetry`). Paths are found
  /// once; a sample is then a handful of small reads, each of which may fail
  /// and reads 0.
  public struct TegraTelemetry: Sendable {
    /// What the comma's UI measures power against: the Orin Nano Super's
    /// 25 W mode.
    public static let powerLimitW = 25.0

    let zones: [String: String]
    /// INA3221 channel 1, VDD_IN on an Orin: the whole board's input.
    let supplyMV: String?
    let supplyMA: String?
    let fanRPM: String?
    let gpuLoad: String
    let gpuFrequency: String

    public init(root: HostRoot = .system) {
      var zones: [String: String] = [:]
      for zone in ThermalZone.all(root) {
        zones[zone.type] = zone.tempPath
      }
      self.zones = zones
      let ina3221 = hwmon(named: "ina3221", root)
      supplyMV = ina3221.map { "\($0)/in1_input" }
      supplyMA = ina3221.map { "\($0)/curr1_input" }
      fanRPM = hwmon(named: "pwm_tach", root).map { "\($0)/rpm" }
      gpuLoad = root.path("/sys/devices/platform/bus@0/17000000.gpu/load")
      gpuFrequency = (gpuDevfreq(root) ?? root.path("/sys/class/devfreq/17000000.gpu")) + "/cur_freq"
    }

    public func read() -> [String: Any] {
      // tj-thermal is the junction temperature the throttle point is defined
      // on, the nearest thing to chestnut's GPU hotspot. As in Python, a zone
      // reading 0 counts as none.
      let temp = celsius(["tj-thermal", "gpu-thermal"])
      let memoryTemp = celsius(["soc0-thermal", "cpu-thermal"])
      let mv = supplyMV.flatMap(Sysfs.readInt) ?? 0
      let ma = supplyMA.flatMap(Sysfs.readInt) ?? 0
      let loadPermille = Sysfs.readInt(gpuLoad) ?? 0
      let frequency = Sysfs.readInt(gpuFrequency) ?? 0
      return [
        "temp_c": pythonRound(temp, 1),
        "memory_temp_c": pythonRound(memoryTemp, 1),
        "power_w": pythonRound(Double(mv * ma) / 1e6, 2),
        "power_limit_w": TegraTelemetry.powerLimitW,
        "gpu_load_pct": min(100, loadPermille / 10),
        "gpu_clock_mhz": frequency / 1_000_000,
        "fan_rpm": fanRPM.flatMap(Sysfs.readInt) ?? 0,
        "supply_mv": mv,
        "supply_ma": ma,
      ]
    }

    private func celsius(_ types: [String]) -> Double {
      for type in types {
        if let path = zones[type], let milli = Sysfs.readInt(path), milli != 0 {
          return Double(milli) / 1000
        }
      }
      return 0
    }
  }
#endif
