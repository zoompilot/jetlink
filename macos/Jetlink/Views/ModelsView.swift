import AppKit
import JetlinkKit
import JetlinkUI
import SwiftUI
import UniformTypeIdentifiers

struct ModelsView: View {
  @Environment(ServerStore.self) private var server
  @Environment(ModelStore.self) private var models
  @State private var selection: ModelRow.ID?
  @State private var inspectorPresented = false
  @State private var importing = false
  @State private var confirmation: Confirmation?

  enum Confirmation: Identifiable {
    case deleteDownload(ModelRow), deleteEngines(ModelRow), switchModel(ModelRow)

    var row: ModelRow {
      switch self {
      case let .deleteDownload(row), let .deleteEngines(row), let .switchModel(row): row
      }
    }

    var id: String {
      switch self {
      case .deleteDownload: "download-\(row.id)"
      case .deleteEngines: "engines-\(row.id)"
      case .switchModel: "use-\(row.id)"
      }
    }
  }

  var body: some View {
    content
      .toolbar {
        // Apart from the window's Start/Stop Server: these act on the list.
        if #available(macOS 26, *) {
          ToolbarSpacer(.fixed)
        }
        ToolbarItem {
          Button("Refresh", systemImage: "arrow.clockwise") { models.refreshCatalog() }
            .keyboardShortcut("r", modifiers: [.command, .shift])
            .help("Refresh the model list")
        }
        ToolbarItem {
          Button("Add Model File…", systemImage: "plus") { importing = true }
            .help("Add an ONNX model file from this Mac")
        }
        // Only opens and closes the inspector, which holds the model's actions.
        ToolbarItem {
          Button(inspectorPresented ? "Hide Inspector" : "Show Inspector", systemImage: "sidebar.trailing") {
            inspectorPresented.toggle()
          }
          .keyboardShortcut("i", modifiers: .command)
          .help(inspectorPresented ? "Hide the model's details and actions" : "Show the model's details and actions")
        }
      }
      .inspector(isPresented: $inspectorPresented) {
        Group {
          if let row = selectedRow {
            ModelDetailView(row: row) { inspectorActions(for: row) }
          } else {
            ContentUnavailableView("No Model Selected", systemImage: "shippingbox", description: Text("Select a model to see its details."))
          }
        }
        .inspectorColumnWidth(min: 280, ideal: 320, max: 440)
      }
      .fileImporter(isPresented: $importing, allowedContentTypes: [ModelsView.onnxType]) { result in
        switch result {
        case let .success(url): models.importModel(at: url)
        case let .failure(error): models.lastError = error.localizedDescription
        }
      }
      .alert(Text(confirmationTitle), isPresented: confirmationPresented, presenting: confirmation) { item in
        Button(confirmButtonTitle(item), role: isDestructive(item) ? .destructive : nil) { perform(item) }
        Button("Cancel", role: .cancel) {}
      } message: { item in
        Text(confirmationMessage(item))
      }
  }

  // MARK: Content

  @ViewBuilder
  private var content: some View {
    if models.catalog == nil, server.runState != .serving, server.runState != .starting {
      ContentUnavailableView {
        Label("Server Not Running", systemImage: "cable.connector.slash")
      } description: {
        Text("Start the server to load the model list.")
      } actions: {
        Button("Start Server") { server.start() }
      }
    } else if models.catalog == nil, models.rows.isEmpty {
      ProgressView("Loading model list…")
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    } else {
      listWithBar
    }
  }

  /// On macOS 26 the bar sits over the list's own scroll edge, the way the
  /// system's bars do, rather than on a slab of material of its own.
  @ViewBuilder
  private var listWithBar: some View {
    if #available(macOS 26, *) {
      // The hard edge Apple suggests on macOS for a bar that carries text.
      list
        .safeAreaBar(edge: .bottom, spacing: 0) { bottomBar }
        .scrollEdgeEffectStyle(.hard, for: .bottom)
    } else {
      list.safeAreaInset(edge: .bottom, spacing: 0) {
        bottomBar
          .background(.bar)
          .overlay(alignment: .top) { Divider() }
      }
    }
  }

  private var list: some View {
    List(selection: $selection) {
      Section {
        rows(models.rows.filter { !$0.isLocal && !$0.isOrphan })
      }
      let local = models.rows.filter { $0.isLocal && !$0.isOrphan }
      if !local.isEmpty {
        Section("Added From This Mac") { rows(local) }
      }
      let orphans = models.rows.filter(\.isOrphan)
      if !orphans.isEmpty {
        Section("Unrecognized Files") { rows(orphans) }
      }
    }
    .listStyle(.inset(alternatesRowBackgrounds: true))
    // Double-click or Return uses the model, as it opens a document in Finder.
    .contextMenu(forSelectionType: ModelRow.ID.self) { ids in
      if let row = models.rows.first(where: { ids.contains($0.id) }) {
        actionButtons(for: row)
      }
    } primaryAction: { ids in
      if let row = models.rows.first(where: { ids.contains($0.id) }), ModelStore.canUse(row) {
        startUse(row)
      }
    }
  }

  private func rows(_ rows: [ModelRow]) -> some View {
    ForEach(rows) { row in
      ModelListRow(
        row: row,
        isCheckingCatalog: models.catalog == nil,
        use: { startUse(row) },
        cancel: { models.cancelDownload(row) })
    }
  }

  private var bottomBar: some View {
    HStack(spacing: 12) {
      Text(diskSummary)
        .foregroundStyle(.secondary)
      Spacer()
      if let error = models.catalog?.error, !error.isEmpty {
        Label(catalogErrorTitle, systemImage: "exclamationmark.triangle.fill")
          .foregroundStyle(.orange)
          .lineLimit(1)
          .help(error)
      }
    }
    .font(.callout)
    // Lines up with the text of the rows above it.
    .padding(.horizontal, 20)
    .padding(.vertical, 8)
    .frame(maxWidth: .infinity)
  }

  /// The fetch failed. The cached list still drives the table, unless there is
  /// none. The reason itself is a whole paragraph, so it goes in the tooltip.
  private var catalogErrorTitle: String {
    let cached = models.catalog?.models.isEmpty ?? true
    return cached ? "Model list unavailable" : "Model list not refreshed"
  }

  private var diskSummary: String {
    guard let disk = models.inventory?.disk else { return "" }
    return ModelFormatting.diskSummary(models: disk.modelsBytes, engines: disk.enginesBytes, neuralEngine: disk.neuralEngineBytes, free: disk.freeBytes)
  }

  // MARK: Actions

  private var selectedRow: ModelRow? {
    guard let selection else { return nil }
    return models.rows.first { $0.id == selection }
  }

  /// Only what applies to this model, as a context menu should be.
  @ViewBuilder
  private func actionButtons(for row: ModelRow) -> some View {
    useActions(for: row)
    if revealURL(row) != nil {
      Divider()
      finderAction(for: row)
    }
    if hasDeletable(row) {
      Divider()
      deleteActions(for: row)
    }
  }

  /// The context menu's actions as two rows of buttons: what to do with the
  /// model, with the main one prominent, then the deletes in red.
  @ViewBuilder
  private func inspectorActions(for row: ModelRow) -> some View {
    let hasMain = ModelStore.canUse(row) || isDownloading(row) || row.status == .loaded || revealURL(row) != nil
    if hasMain || hasDeletable(row) {
      Section("Actions") {
        if hasMain {
          HStack(spacing: 8) {
            useActions(for: row)
              .buttonStyle(.borderedProminent)
            finderAction(for: row)
            Spacer(minLength: 0)
          }
        }
        if hasDeletable(row) {
          // Side by side when the inspector is wide enough, stacked when not.
          ViewThatFits(in: .horizontal) {
            HStack(spacing: 8) {
              deleteActions(for: row)
              Spacer(minLength: 0)
            }
            VStack(alignment: .leading, spacing: 6) { deleteActions(for: row) }
          }
          .foregroundStyle(.red)
        }
      }
    }
  }

  @ViewBuilder
  private func useActions(for row: ModelRow) -> some View {
    if ModelStore.canUse(row) {
      Button("Use Model") { startUse(row) }
    }
    if isDownloading(row) {
      Button("Cancel Download") { models.cancelDownload(row) }
    }
    if row.status == .loaded {
      Button("Stop Using Model") { models.unload() }
    }
  }

  @ViewBuilder
  private func finderAction(for row: ModelRow) -> some View {
    if let url = revealURL(row) {
      Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
    }
  }

  @ViewBuilder
  private func deleteActions(for row: ModelRow) -> some View {
    if hasModelFile(row) {
      Button("Delete Download…", role: .destructive) { confirmation = .deleteDownload(row) }
    }
    if !row.preparedFor.isEmpty {
      Button("Delete Prepared Engines…", role: .destructive) { confirmation = .deleteEngines(row) }
    }
  }

  private func isDownloading(_ row: ModelRow) -> Bool {
    if case .downloading = row.status { return true }
    return false
  }

  private func hasDeletable(_ row: ModelRow) -> Bool {
    hasModelFile(row) || !row.preparedFor.isEmpty
  }

  private func startUse(_ row: ModelRow) {
    if models.useNeedsConfirmation(row) {
      confirmation = .switchModel(row)
    } else {
      models.use(row)
    }
  }

  private func hasModelFile(_ row: ModelRow) -> Bool {
    guard let sha = row.sha256 else { return false }
    return models.inventory?.models.contains { $0.sha256 == sha } ?? false
  }

  private func revealURL(_ row: ModelRow) -> URL? {
    if let sha = row.sha256, let model = models.inventory?.models.first(where: { $0.sha256 == sha }) {
      return URL(fileURLWithPath: model.path)
    }
    if let artifact = row.preparedFor.first {
      return URL(fileURLWithPath: artifact.path)
    }
    return nil
  }

  private func perform(_ item: Confirmation) {
    switch item {
    case let .deleteDownload(row): models.forget(row, artifacts: false, model: true)
    case let .deleteEngines(row): models.forget(row, artifacts: true, model: false)
    case let .switchModel(row): models.use(row, confirmedInterruption: true)
    }
  }

  // MARK: Alert plumbing

  private var confirmationPresented: Binding<Bool> {
    Binding(get: { confirmation != nil }, set: { if !$0 { confirmation = nil } })
  }

  private var confirmationTitle: String {
    switch confirmation {
    case .deleteDownload: "Delete the download?"
    case .deleteEngines: "Delete the prepared engines?"
    case let .switchModel(row): "Use \(row.displayName)?"
    case nil: ""
    }
  }

  private func confirmButtonTitle(_ item: Confirmation) -> String {
    switch item {
    case .deleteDownload, .deleteEngines: "Delete"
    case .switchModel: "Use Model"
    }
  }

  private func isDestructive(_ item: Confirmation) -> Bool {
    switch item {
    case .deleteDownload, .deleteEngines: true
    case .switchModel: false
    }
  }

  private func confirmationMessage(_ item: Confirmation) -> String {
    switch item {
    case let .deleteDownload(row):
      let size = row.bytes.map { "\(ByteCount.string($0)) " } ?? ""
      return "Deletes the \(size)model file for \(row.displayName). The prepared engine stays, so the comma can still use this model."
    case let .deleteEngines(row):
      let bytes = row.preparedFor.reduce(Int64(0)) { $0 + $1.bytes }
      var text =
        "Deletes every prepared engine for \(row.displayName), \(ByteCount.string(bytes)) in all. Using it again prepares it again, which takes as long as the first time."
      if row.isLoaded {
        text += " Jetlink stops using it first."
      }
      return text
    case let .switchModel(row):
      let current = models.rows.first { $0.isLoaded }?.displayName ?? "another model"
      return "The comma is using \(current). Switching drops it to its small model until \(row.displayName) is ready and the comma reconnects."
    }
  }

  static let onnxType = UTType(filenameExtension: "onnx") ?? .data
}

/// One model in the list: its name and tags, one line of facts, and on the
/// trailing edge whatever applies now: Use Model, progress, or In Use.
struct ModelListRow: View {
  let row: ModelRow
  let isCheckingCatalog: Bool
  let use: () -> Void
  let cancel: () -> Void

  var body: some View {
    HStack(spacing: 12) {
      VStack(alignment: .leading, spacing: 2) {
        HStack(spacing: 6) {
          Text(row.displayName)
            .lineLimit(1)
            .truncationMode(.middle)
          if row.isDefault { ModelTag("Default", tone: .accentColor) }
          if row.isRequestedByComma { ModelTag("Comma", tone: .green) }
        }
        let detail = ModelFormatting.detailLine(row)
        if !detail.isEmpty {
          Text(detail)
            .font(.callout)
            .foregroundStyle(.secondary)
            .lineLimit(1)
        }
      }
      .layoutPriority(1)
      Spacer(minLength: 12)
      accessory
    }
    .padding(.vertical, 5)
    .padding(.horizontal, 4)
    .contentShape(Rectangle())
  }

  @ViewBuilder
  private var accessory: some View {
    switch row.status {
    case .loaded, .unresolved:
      ModelStatusLabel(row.status, isCheckingCatalog: isCheckingCatalog)
        .font(.callout)
    case let .downloading(frac, rateBps):
      HStack(spacing: 8) {
        progress(frac: frac, caption: ModelFormatting.downloadCaption(frac: frac, rateBps: rateBps))
        Button(action: cancel) {
          Image(systemName: "xmark.circle.fill")
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .help("Cancel the download")
      }
    case let .preparing(stage, frac, msg):
      HStack(spacing: 8) {
        progress(frac: frac, caption: ProgressRow.stageName(stage))
          .help(msg)
        // Where a download's cancel button sits, so the two bars line up.
        Image(systemName: "xmark.circle.fill")
          .hidden()
      }
    case .failed:
      HStack(spacing: 10) {
        ModelStatusLabel(row.status)
          .font(.callout)
        useButton
      }
    case .notDownloaded, .downloaded, .prepared:
      useButton
    }
  }

  private var useButton: some View {
    Button("Use Model", action: use)
      .buttonStyle(.bordered)
      .buttonBorderShape(.capsule)
      .controlSize(.small)
      .help(ModelFormatting.useHelp(row, device: "this Mac"))
  }

  private func progress(frac: Double, caption: String) -> some View {
    VStack(alignment: .trailing, spacing: 3) {
      ProgressView(value: min(max(frac, 0), 1))
        .controlSize(.small)
        .frame(width: 140)
      Text(caption)
        .font(.caption)
        .foregroundStyle(.secondary)
        .monospacedDigit()
        .lineLimit(1)
    }
  }
}

#Preview("Models") {
  ModelsView()
    .environment(
      ServerStore.preview(
        runState: .serving, info: PreviewData.serverInfo, link: PreviewData.linkConnected,
        engine: PreviewData.engineReady, stats: PreviewData.stats)
    )
    .environment(
      ModelStore.preview(
        catalog: PreviewData.catalog, inventory: PreviewData.inventory,
        downloads: [PreviewData.download.sha256: PreviewData.download], engine: PreviewData.engineReady)
    )
    .frame(width: 860, height: 480)
}

#Preview("Server stopped") {
  ModelsView()
    .environment(ServerStore.preview(runState: .stopped, info: nil, link: PreviewData.linkWaiting, engine: PreviewData.engineNone))
    .environment(ModelStore.preview(catalog: nil, inventory: nil, engine: PreviewData.engineNone))
    .frame(width: 860, height: 480)
}
