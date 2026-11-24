#if os(Linux)
  import Foundation
  import JetlinkTestSupport
  import Testing

  @testable import JetlinkLinux

  /// The clocks in a tree of their own: an Orin Nano Super's GPU steps and
  /// two of its modes, no memory or CPU nodes.
  @Suite("A Jetson's clocks")
  struct JetsonClocksTests {
    static func tree() -> Tree {
      let tree = Tree()
      let gpu = "/sys/class/devfreq/17000000.gpu"
      tree.write("\(gpu)/available_frequencies", "306000000 408000000 510000000 612000000 714000000 816000000 918000000 1020000000\n")
      tree.write("\(gpu)/min_freq", "306000000\n")
      tree.write("\(gpu)/max_freq", "1020000000\n")
      tree.write(
        "/etc/nvpmodel.conf",
        """
        < POWER_MODEL ID=0 NAME=15W >
        GPU MAX_FREQ 612000000
        < POWER_MODEL ID=2 NAME=MAXN_SUPER >
        GPU MAX_FREQ -1

        """)
      return tree
    }

    @Test("A GPU cap lowers the GPU's top inside the mode, and never lifts it")
    func gpuCap() throws {
      let tree = Self.tree()
      let top = { (mode: String, cap: Int?) in JetsonClocks(modeName: mode, gpuMax: cap, root: tree.root)?.nodes.first?.clock.top }
      #expect(top("MAXN_SUPER", nil) == 1_020_000_000)
      #expect(top("MAXN_SUPER", 510_000_000) == 510_000_000)
      // the mode's own limit stands when it is lower
      #expect(top("15W", 816_000_000) == 612_000_000)
      // under the slowest step: the slowest
      #expect(top("MAXN_SUPER", 100_000_000) == 306_000_000)
    }

    @Test("Released, the GPU's highest is the capped top and its lowest the bottom")
    func written() throws {
      let tree = Self.tree()
      let clocks = try #require(JetsonClocks(modeName: "MAXN_SUPER", gpuMax: 408_000_000, root: tree.root))
      clocks.release()
      #expect(eventually { tree.read("/sys/class/devfreq/17000000.gpu/max_freq") == "408000000" })
      #expect(tree.read("/sys/class/devfreq/17000000.gpu/min_freq") == "306000000")
    }
  }
#endif
