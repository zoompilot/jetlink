import JetlinkKit
import JetlinkUI
import SwiftUI

/// The Status tab's cards, most important first: the headroom, where the time
/// goes, the last two minutes, then the link and the device. On its side a
/// phone shows the ring on the left and the rest beside it; an iPad sets the
/// cards in two columns. No UIKit here, so it renders anywhere SwiftUI does.
struct StatusContent: View {
  /// How the cards are set out, from the screen's size classes.
  enum Arrangement {
    /// One column, as an iPhone held upright or a narrow iPad window shows them.
    case column
    /// The ring and beside it the numbers, as an iPhone on its side shows them.
    case sideways
    /// Two columns, for the width of an iPad.
    case columns
  }

  let state: StatusState
  var arrangement = Arrangement.column
  var actions = StatusActions()
  /// Off only for snapshots: ImageRenderer draws nothing inside a ScrollView.
  var scrolls = true

  static let margin: CGFloat = 16
  static let spacing: CGFloat = 16

  var body: some View {
    if scrolls {
      ScrollView {
        layout
          .padding(.bottom, 24)
      }
      .contentMargins(.horizontal, StatusContent.margin, for: .scrollContent)
      .background(Color.groupedBackground)
    } else {
      layout
        .padding(.horizontal, StatusContent.margin)
        .background(Color.groupedBackground)
    }
  }

  private var layout: some View {
    VStack(spacing: StatusContent.spacing) {
      if state.needsForeground {
        banner(
          "Keep Jetlink on Screen", "The big model stops while Jetlink is in the background.",
          symbol: "exclamationmark.triangle.fill")
      }
      if let warning = state.slowWarning {
        banner("Big Model Running Slowly", warning, symbol: "exclamationmark.triangle.fill")
      }
      if let warning = state.storageWarning {
        banner("Low Storage", warning, symbol: "externaldrive.badge.exclamationmark")
      }
      // in every arrangement: the Link tile's note sits below the fold upright
      // and is not drawn on a phone on its side, so a USB 2 cable, hub or
      // phone read as Connected and nothing more
      if let medium = state.linkMedium, let advice = medium.advice(cable: true) {
        banner(
          "\(medium.title) Link", "\(advice) \(USBSpeedGuide.summary(for: state.deviceName))",
          symbol: "tortoise.fill")
      }
      cards
    }
    // One column stays readable in a window wider than a phone.
    .frame(maxWidth: arrangement == .column ? readableContentWidth : .infinity)
    .frame(maxWidth: .infinity)
  }

  @ViewBuilder
  private var cards: some View {
    switch arrangement {
    case .column:
      VStack(spacing: StatusContent.spacing) {
        HeroCard(state: state, actions: actions)
        if let recent = state.servedStats {
          latency(recent)
          if state.history.count > 1 {
            history
          }
          SectionHeader("Link", detail: state.linkMedium?.phoneTitle)
          link(recent)
        }
        SectionHeader(state.deviceName)
        device
      }
    case .sideways:
      HStack(alignment: .top, spacing: StatusContent.spacing) {
        HeroCard(state: state, compact: true, actions: actions)
          .frame(width: 330)
        VStack(spacing: StatusContent.spacing) {
          if let recent = state.servedStats {
            latency(recent, compact: true)
            link(recent)
          } else {
            device
          }
        }
      }
    case .columns:
      // The headroom and where the time goes on the left; the history, the
      // link and the device on the right.
      HStack(alignment: .top, spacing: StatusContent.spacing) {
        VStack(spacing: StatusContent.spacing) {
          HeroCard(state: state, actions: actions)
          if let recent = state.servedStats {
            latency(recent)
          }
        }
        VStack(spacing: StatusContent.spacing) {
          if let recent = state.servedStats {
            if state.history.count > 1 {
              history
            }
            SectionHeader("Link", detail: state.linkMedium?.phoneTitle)
            link(recent)
          }
          SectionHeader(state.deviceName)
          device
        }
      }
    }
  }

  /// An orange banner over the cards: something the person should act on
  /// now, such as an app the system is about to suspend or a slow link.
  private func banner(_ title: String, _ detail: String, symbol: String) -> some View {
    Label {
      VStack(alignment: .leading, spacing: 2) {
        Text(title)
          .font(.subheadline.weight(.semibold))
        Text(detail)
          .font(.footnote)
      }
    } icon: {
      Image(systemName: symbol)
    }
    .foregroundStyle(.white)
    .padding(14)
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(.orange, in: .rect(cornerRadius: cardCornerRadius, style: .continuous))
    .padding(.top, 8)
  }

  private func latency(_ recent: StatsEvent, compact: Bool = false) -> some View {
    SummaryCard(title: "Latency", systemImage: "timer", tint: .blue, trailing: "Last 10 s") {
      LatencyBreakdown(stats: recent, compact: compact)
    }
  }

  private var history: some View {
    SummaryCard(title: "History", systemImage: "chart.xyaxis.line", tint: .purple, trailing: "2 min") {
      LatencyHistoryChart(history: state.history)
    }
  }

  private func link(_ recent: StatsEvent) -> some View {
    MetricGrid {
      GridRow {
        MetricTile(
          title: "Frame Rate", systemImage: "speedometer", tint: .teal,
          value: recent.fps.formatted(.number.precision(.fractionLength(1))), unit: "fps",
          note: recent.fps < 18 ? "Below 20" : nil, noteTone: .orange)
        MetricTile(
          title: "Slow Frames", systemImage: "tortoise.fill", tint: .pink,
          value: recent.slow.formatted(), note: recent.slow > 0 ? "Over 60 ms" : nil, noteTone: .red)
      }
    }
  }

  private var device: some View {
    let health = state.health
    return MetricGrid {
      GridRow {
        MetricTile(
          title: "Temperature", systemImage: health.thermal.symbol, tint: .orange,
          value: health.thermal.title, note: health.thermal.note, noteTone: health.thermal.tone)
        MetricTile(
          title: "Battery", systemImage: health.batterySymbol, tint: .green,
          value: health.batteryValue, unit: health.batteryLevel == nil ? nil : "%",
          note: health.powerText, noteTone: health.batteryTone)
      }
      GridRow {
        MetricTile(
          title: "Memory", systemImage: "memorychip", tint: .indigo,
          value: health.memoryValue, unit: health.availableMemory == nil ? nil : "GB",
          note: health.memoryNote, noteTone: health.memoryTone)
        MetricTile(
          title: "Link", systemImage: state.linkMedium == nil ? "cable.connector.slash" : "cable.connector", tint: .teal,
          value: state.linkMedium?.phoneTitle ?? "None",
          note: linkNote, noteTone: state.linkMedium?.isSlow == true ? .orange : nil)
      }
    }
  }

  /// Under the Link tile: why there is none, or that a slow one costs frames.
  private var linkNote: String {
    guard let medium = state.linkMedium else { return state.cableAddress == nil ? "Waiting" : "Connecting" }
    return medium.isSlow ? "Slow, use USB 3" : "Connected"
  }
}

/// A section title over a group of cards, as Health and Fitness set them,
/// with a word or two on the right when there is something to say.
struct SectionHeader: View {
  let title: String
  var detail: String?

  init(_ title: String, detail: String? = nil) {
    self.title = title
    self.detail = detail
  }

  var body: some View {
    HStack(alignment: .firstTextBaseline) {
      Text(title)
        .font(.title3.weight(.bold))
        .accessibilityAddTraits(.isHeader)
      Spacer()
      if let detail {
        Text(detail)
          .font(.subheadline)
          .foregroundStyle(.secondary)
      }
    }
    .frame(maxWidth: .infinity, alignment: .leading)
    .padding(.top, 8)
    .padding(.bottom, -4)
  }
}
