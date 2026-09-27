// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.

import Darwin
import Foundation

/// One IPv4 address the phone has right now, for telling the comma where to connect.
struct InterfaceAddress: Hashable, Identifiable {
  let interface: String
  let address: String
  var id: String { "\(interface) \(address)" }

  /// What the interface probably is. iOS names a USB network link or an
  /// Ethernet adapter en2 and up; en0 is Wi-Fi.
  var kind: String {
    if interface == "en0" { return "Wi-Fi" }
    if interface.hasPrefix("en") { return "USB or Ethernet" }
    if interface.hasPrefix("bridge") { return "Personal Hotspot" }
    if interface.hasPrefix("pdp_ip") { return "Cellular" }
    return interface
  }

  /// Whether the comma can reach the server this way within the frame budget.
  var isWired: Bool { interface.hasPrefix("en") && interface != "en0" }

  /// A self-assigned 169.254.x.x address: a link with no DHCP server on it.
  var isLinkLocal: Bool { address.hasPrefix("169.254.") }

  static func current() -> [InterfaceAddress] {
    var head: UnsafeMutablePointer<ifaddrs>?
    guard getifaddrs(&head) == 0, let first = head else { return [] }
    defer { freeifaddrs(head) }
    var out: [InterfaceAddress] = []
    for p in sequence(first: first, next: { $0.pointee.ifa_next }) {
      let flags = Int32(p.pointee.ifa_flags)
      guard let sa = p.pointee.ifa_addr, sa.pointee.sa_family == UInt8(AF_INET),
        flags & IFF_UP != 0, flags & IFF_LOOPBACK == 0
      else { continue }
      let name = String(cString: p.pointee.ifa_name)
      // VPN tunnels: never how the comma reaches the phone.
      if name.hasPrefix("utun") || name.hasPrefix("ipsec") { continue }
      var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
      guard getnameinfo(sa, socklen_t(sa.pointee.sa_len), &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST) == 0
      else { continue }
      let address = String(decoding: host.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
      out.append(InterfaceAddress(interface: name, address: address))
    }
    // Wired first: that is the one the comma should use.
    return out.sorted { ($0.isWired ? 0 : 1, $0.interface) < ($1.isWired ? 0 : 1, $1.interface) }
  }
}
