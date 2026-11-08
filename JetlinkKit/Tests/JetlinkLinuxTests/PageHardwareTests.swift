#if os(Linux)
  import Foundation
  import JetlinkServer
  import Testing

  @testable import JetlinkLinux

  @Suite("Status page hardware")
  struct PageHardwareTests {
    /// The fixture Jetson's telemetry, as LinuxHost makes it there.
    static let jetsonGPU = GPUTelemetry(read: { TegraTelemetry(root: jetson).read() }, name: nil, tegra: true)

    func valid(_ object: [String: Any]) -> Bool {
      JSONSerialization.isValidJSONObject(object) && (try? JSONSerialization.data(withJSONObject: object)) != nil
    }

    @Test("The bench Jetson as the host event names it")
    func jetsonHost() {
      let hardware = PageHardware(root: jetson, cache: nil, gpu: Self.jetsonGPU)
      let host = hardware.host()
      #expect(host["hostname"] as? String == "jetlink")
      #expect(host["board"] as? String == "NVIDIA Jetson Orin Nano Super")
      #expect(host["os"] as? String == "JetPack 7 (Jetson Linux 39.2.1)")
      #expect(host["kernel"] as? String == "6.8.12-1021-tegra")
      #expect(host["gpu"] as? String == "Orin")
      #expect(host["jetson"] as? Bool == true)
      #expect(valid(host))
    }

    @Test("A sample of the bench Jetson")
    func jetsonSample() throws {
      let cache = FileManager.default.temporaryDirectory
      let hardware = PageHardware(root: jetson, cache: cache, gpu: Self.jetsonGPU)
      let sample = hardware.sample()
      #expect(valid(sample))

      let cpu = try #require(sample["cpu"] as? [[String: Any]])
      #expect(cpu.count == 6)
      #expect(cpu.allSatisfy { $0["online"] as? Bool == true && $0["mhz"] as? Int == 1728 })
      // A load is a difference of two reads: none on the first.
      #expect(cpu.allSatisfy { $0["load"] == nil })

      let memory = try #require(sample["mem"] as? [String: Any])
      #expect(memory["total"] as? Int == 7_727_244 * 1024)
      #expect(memory["used"] as? Int == (7_727_244 - 5_362_572) * 1024)
      #expect(memory["swap_total"] as? Int == 10_485_752 * 1024)
      #expect(memory["swap_used"] as? Int == (10_485_752 - 10_481_400) * 1024)

      let gpu = try #require(sample["gpu"] as? [String: Any])
      #expect(gpu["load"] as? Int == 0 && gpu["mhz"] as? Int == 1020)

      // cv0-2 have no sensor: left out rather than shown as 0.
      let temps = try #require(sample["temps"] as? [[String: Any]])
      #expect(temps.map { $0["name"] as? String } == ["cpu-thermal", "gpu-thermal", "soc0-thermal", "soc1-thermal", "soc2-thermal", "tj-thermal"])
      #expect(temps.map { $0["c"] as? Double } == [58.3, 58.8, 57.3, 58.5, 56.9, 58.9])

      let rails = try #require((sample["power"] as? [String: Any])?["rails"] as? [[String: Any]])
      #expect(rails.map { $0["name"] as? String } == ["VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC"])
      #expect(rails.map { $0["w"] as? Double } == [8.64, 1.98, 3.72])

      let fan = try #require(sample["fan"] as? [String: Any])
      #expect(fan["rpm"] as? Int == 2367 && fan["pct"] as? Int == 42)
      #expect(sample["mode"] as? String == "MAXN_SUPER")
      #expect(sample["uptime"] as? Double == 13320.75)
      #expect(sample["load"] as? [Double] == [0.0, 0.01, 0.0])
      let disk = try #require(sample["disk"] as? [String: Any])
      #expect(disk["path"] as? String == cache.path)
      #expect((disk["total"] as? Int ?? 0) > 0 && (disk["used"] as? Int ?? -1) >= 0)
    }

    @Test("CPU load between two samples, and a core the power mode took offline")
    func cpuLoad() throws {
      let tree = Tree.jetsonCopy()
      let hardware = PageHardware(root: tree.root, cache: nil, gpu: Self.jetsonGPU)
      _ = hardware.sample()
      // cpu0: 100 jiffies, 50 idle. cpu1: 100, 70 idle (60 idle + 10 iowait).
      // cpu4 goes offline: out of /proc/stat and the online list.
      tree.write(
        "/proc/stat",
        """
        cpu  69911 52 55180 7738753 6113 13283 9366 0 0 0
        cpu0 12396 8 11368 1273479 1340 9643 7111 0 0 0
        cpu1 14789 6 11964 1285902 1130 786 344 0 0 0
        cpu2 13309 14 10989 1289252 323 735 324 0 0 0
        cpu3 14339 9 12386 1285426 1973 749 316 0 0 0
        cpu5 7984 5 4119 1302873 620 453 203 0 0 0
        """)
      tree.write("/sys/devices/system/cpu/online", "0-3,5\n")
      let cpu = try #require(hardware.sample()["cpu"] as? [[String: Any]])
      #expect(cpu.count == 6)
      #expect(cpu[0]["load"] as? Double == 50)
      #expect(cpu[1]["load"] as? Double == 30)
      #expect(cpu[2]["load"] as? Double == nil)
      #expect(cpu[4]["online"] as? Bool == false && cpu[4]["load"] == nil && cpu[4]["mhz"] == nil)
      #expect(cpu[5]["online"] as? Bool == true)

      // After a pause the page resets: no load averaged over the pause.
      hardware.reset()
      let fresh = try #require(hardware.sample()["cpu"] as? [[String: Any]])
      #expect(fresh.allSatisfy { $0["load"] == nil })
    }

    @Test("A PC: its OS and CPU by name, its CPU package sensor, no Jetson parts")
    func pc() throws {
      let tree = Tree()
      tree.write("/proc/sys/kernel/hostname", "desk\n")
      tree.write("/proc/sys/kernel/osrelease", "6.8.0-45-generic\n")
      tree.write("/etc/os-release", "NAME=\"Ubuntu\"\nPRETTY_NAME=\"Ubuntu 24.04.1 LTS\"\n")
      tree.write("/proc/cpuinfo", "processor\t: 0\nmodel name\t: AMD Ryzen 7 7700X 8-Core Processor\n")
      tree.write("/proc/meminfo", "MemTotal:       32000000 kB\nMemAvailable:   30000000 kB\nSwapTotal:             0 kB\nSwapFree:              0 kB\n")
      tree.write("/sys/devices/system/cpu/possible", "0-1\n")
      tree.write("/sys/devices/system/cpu/online", "0-1\n")
      tree.write("/proc/stat", "cpu  2 0 2 10 0 0 0 0\ncpu0 1 0 1 5 0 0 0 0\ncpu1 1 0 1 5 0 0 0 0\n")
      tree.write("/sys/class/hwmon/hwmon0/name", "k10temp\n")
      tree.write("/sys/class/hwmon/hwmon0/temp1_input", "45300\n")
      let hardware = PageHardware(root: tree.root, cache: nil, gpu: nil)
      let host = hardware.host()
      #expect(host["os"] as? String == "Ubuntu 24.04.1 LTS")
      #expect(host["board"] as? String == "AMD Ryzen 7 7700X 8-Core Processor")
      #expect(host["jetson"] as? Bool == false && host["gpu"] == nil)
      let sample = hardware.sample()
      #expect(valid(sample))
      #expect(sample["gpu"] == nil && sample["power"] == nil && sample["fan"] == nil && sample["mode"] == nil && sample["disk"] == nil)
      #expect((sample["temps"] as? [[String: Any]])?.map { $0["c"] as? Double } == [45.3])
      #expect((sample["cpu"] as? [[String: Any]])?.count == 2)
    }

    @Test("A PC's NVIDIA GPU, as NVML reads it, named in the host event")
    func nvmlGPU() throws {
      let tree = Tree()
      tree.write("/proc/stat", "cpu0 1 0 1 5 0 0 0 0\n")
      let nvml = GPUTelemetry(
        read: { ["temp_c": 63.0, "power_w": 87.65, "power_limit_w": 170.0, "gpu_load_pct": 37, "gpu_clock_mhz": 2505, "fan_pct": 44] }, name: "Test GPU",
        tegra: false)
      let hardware = PageHardware(root: tree.root, cache: nil, gpu: nvml)
      #expect(hardware.host()["gpu"] as? String == "Test GPU")
      let gpu = try #require(hardware.sample()["gpu"] as? [String: Any])
      #expect(gpu["load"] as? Int == 37 && gpu["mhz"] as? Int == 2505 && gpu["temp"] as? Double == 63 && gpu["fan_pct"] as? Int == 44)
      #expect(Set(gpu.keys) == ["load", "mhz", "temp", "power_w", "power_limit_w", "fan_pct"])
    }

    @Test("JetPack names, CPU lists and nvpmodel's modes as the files write them")
    func parsing() {
      #expect(PageHardware.jetpack("# R36 (release), REVISION: 4.4, GCID: 41062509, BOARD: generic") == "JetPack 6 (Jetson Linux 36.4.4)")
      #expect(PageHardware.jetpack("# R40 (release), REVISION: 1.0") == "Jetson Linux 40.1.0")
      #expect(PageHardware.jetpack("garbage") == nil)
      #expect(PageHardware.cpuList("0-3,5") == [0, 1, 2, 3, 5])
      #expect(PageHardware.cpuList("") == [])
      #expect(JetsonPower.modes("# < POWER_MODEL ID=id_num NAME=mode_name >\n< POWER_MODEL ID=0 NAME=15W >\n") == [0: "15W"])
      #expect(PageHardware.tegraGPU("nvidia,p3768-0000+p3767-0005-super\0nvidia,tegra234\0") == "Orin")
    }
  }
#endif
