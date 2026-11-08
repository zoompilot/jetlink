import Testing

@testable import JetlinkServer

/// nvpmodel's files and a clock's range, from an Orin Nano Super's.
@Suite("Jetson power")
struct JetsonPowerTests {
  static let gpuSteps = [306, 408, 510, 612, 714, 816, 918, 1020].map { $0 * 1_000_000 }

  static let conf = """
    # < POWER_MODEL ID=id_num NAME=mode_name >
    < PARAM TYPE=CLOCK NAME=GPU >
    MAX_FREQ /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/max_freq
    < POWER_MODEL ID=0 NAME=15W >
    CPU_A78_0 MAX_FREQ 1497600
    GPU MAX_FREQ 612000000
    EMC MAX_FREQ 2133000000
    < POWER_MODEL ID=2 NAME=MAXN_SUPER >
    CPU_A78_0 MAX_FREQ -1
    GPU MAX_FREQ -1
    EMC MAX_FREQ -1
    < PM_CONFIG DEFAULT=1 >
    """

  @Test("A clock's top is its fastest step under the mode's limit, a limit between steps taking the one below")
  func clock() {
    let free = JetsonPower.Clock(steps: Self.gpuSteps, cap: nil)
    #expect(free.top == 1_020_000_000 && free.bottom == 306_000_000)
    #expect(JetsonPower.Clock(steps: Self.gpuSteps, cap: 950_000_000).top == 918_000_000)
  }

  @Test("A mode's limits by name; -1 is no limit")
  func limits() {
    #expect(JetsonPower.limits(Self.conf, mode: "15W") == JetsonPower.ModeLimits(gpu: 612_000_000, memory: 2_133_000_000, cpu: 1_497_600))
    #expect(JetsonPower.limits(Self.conf, mode: "MAXN_SUPER") == JetsonPower.ModeLimits())
    #expect(JetsonPower.limits(Self.conf, mode: "60W") == nil)
  }

  @Test("The modes by ID, and the status file's mode, with or without its fan mode")
  func modes() {
    #expect(JetsonPower.modes(Self.conf) == [0: "15W", 2: "MAXN_SUPER"])
    #expect(JetsonPower.modeID(status: "pmode:0002") == 2)
    #expect(JetsonPower.modeID(status: "pmode:0000 fmode:quiet") == 0)
    #expect(JetsonPower.modeID(status: nil) == nil)
  }
}
