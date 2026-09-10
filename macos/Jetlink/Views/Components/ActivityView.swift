import SwiftUI

/// The centre of the toolbar, in the shape of Xcode's activity view: what is
/// being served on the left (the model, then the backend), what is happening
/// on the right, and a thin bar along the bottom while a model is prepared or
/// loaded. Clicking it opens Status.
///
/// Not a `Button`: a button in the principal placement is taken for a toolbar
/// button and the item is dropped from the toolbar altogether (bisected on
/// macOS 26). On macOS 26 the toolbar wraps the item in its own glass capsule,
/// so no background is drawn; on macOS 15 it is the inset rectangle Xcode 16
/// draws.
struct ToolbarActivityView: View {
  @Environment(ServerStore.self) private var server
  @Environment(ModelStore.self) private var models
  @Environment(AppSettings.self) private var settings
  @Environment(Navigation.self) private var navigation

  static let width: CGFloat = 480
  static let height: CGFloat = 30

  var body: some View {
    content
      .onTapGesture { navigation.selection = .status }
      .help("Open Status")
      .accessibilityElement(children: .ignore)
      .accessibilityLabel("\(modelName), \(backendName), \(summary.0)")
      .accessibilityAddTraits(.isButton)
      .accessibilityAction { navigation.selection = .status }
  }

  private var content: some View {
    HStack(spacing: 6) {
      crumb(modelName, symbol: "shippingbox")
      Image(systemName: "chevron.right")
        .font(.system(size: 9, weight: .semibold))
        .foregroundStyle(.tertiary)
      crumb(backendName, symbol: "cpu")
      Spacer(minLength: 16)
      Text(summary.0)
        .foregroundStyle(statusColor)
        .lineLimit(1)
        .fixedSize()
      if isBusy {
        ProgressView()
          .controlSize(.mini)
      }
    }
    .font(.callout)
    .padding(.horizontal, 12)
    // Centred on the window, not the detail column, so it has to give way to
    // the title on the left and the run button on the right: a fixed width
    // that does not fit is dropped from the toolbar altogether.
    .frame(
      minWidth: 240, idealWidth: ToolbarActivityView.width, maxWidth: ToolbarActivityView.width,
      minHeight: ToolbarActivityView.height, maxHeight: ToolbarActivityView.height
    )
    .overlay(alignment: .bottom) { progressBar }
    .contentShape(RoundedRectangle(cornerRadius: 9))
    .modifier(ActivityBackground())
  }

  private func crumb(_ text: String, symbol: String) -> some View {
    Label {
      Text(text)
        .lineLimit(1)
        .truncationMode(.middle)
    } icon: {
      Image(systemName: symbol)
        .foregroundStyle(.secondary)
    }
    .labelStyle(.titleAndIcon)
    .layoutPriority(1)
  }

  /// Xcode's build bar: the tint along the bottom edge, as far as the job got.
  @ViewBuilder
  private var progressBar: some View {
    if showsProgress {
      GeometryReader { proxy in
        ZStack(alignment: .leading) {
          Capsule().fill(.quaternary)
          Capsule().fill(.tint)
            .frame(width: max(6, proxy.size.width * min(max(server.engine.frac, 0), 1)))
        }
      }
      .frame(height: 3)
      .padding(.horizontal, 12)
      .padding(.bottom, 4)
      .accessibilityHidden(true)
    }
  }

  // MARK: What the pieces say

  private var summary: (String, StatusBadge.Tone) {
    StatusBadge.summary(runState: server.runState, link: server.link, engine: server.engine)
  }

  private var modelName: String {
    guard let sha = server.engine.sha256 else { return "No model" }
    return models.rows.first { $0.sha256 == sha }?.displayName ?? "Model \(sha.prefix(8))"
  }

  private var backendName: String {
    guard let info = server.info else { return settings.backend.title }
    return StatusView.backendDescription(backend: info.backend, device: info.device)
  }

  private var statusColor: Color {
    switch summary.1 {
    case .bad: .red
    case .warning: .orange
    case .neutral, .info, .good: .primary
    }
  }

  private var isBusy: Bool {
    server.runState == .starting || server.runState == .stopping
  }

  private var showsProgress: Bool {
    server.runState == .serving && (server.engine.state == .building || server.engine.state == .loading)
  }

  private struct ActivityBackground: ViewModifier {
    @ViewBuilder
    func body(content: Content) -> some View {
      if #available(macOS 26, *) {
        content
      } else {
        content.background(.quaternary, in: RoundedRectangle(cornerRadius: 9))
      }
    }
  }
}

#Preview("Loading") {
  ToolbarActivityView()
    .environment(ServerStore.preview(runState: .serving, info: PreviewData.serverInfo, link: PreviewData.linkWaiting, engine: PreviewData.engineLoading))
    .environment(ModelStore.preview(catalog: PreviewData.catalog, inventory: PreviewData.inventory, engine: PreviewData.engineLoading))
    .environment(AppSettings.preview())
    .environment(Navigation())
    .padding()
}

#Preview("Failed") {
  ToolbarActivityView()
    .environment(ServerStore.preview(runState: .failed("x"), info: nil, link: PreviewData.linkWaiting, engine: PreviewData.engineNone))
    .environment(ModelStore.preview(catalog: PreviewData.catalog, inventory: PreviewData.inventory, engine: PreviewData.engineNone))
    .environment(AppSettings.preview())
    .environment(Navigation())
    .padding()
}
