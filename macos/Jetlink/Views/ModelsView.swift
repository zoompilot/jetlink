import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// Build times as the catalog reports them, ISO-8601 in UTC.
enum BuildTime {
  static func date(_ text: String?) -> Date? {
    guard let text, !text.isEmpty else { return nil }
    let withFraction = ISO8601DateFormatter()
    withFraction.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    if let date = withFraction.date(from: text) { return date }
    let plain = ISO8601DateFormatter()
    plain.formatOptions = [.withInternetDateTime]
    return plain.date(from: text)
  }

  /// An abbreviated date, or nothing at all when the build time is unknown.
  static func text(_ text: String?) -> String {
    guard let date = date(text) else { return "" }
    return date.formatted(Date.FormatStyle(date: .abbreviated, time: .omitted))
  }
}

struct ModelsView: View {
  @Environment(ServerStore.self) private var server
  @Environment(ModelStore.self) private var models
  @State private var selection: ModelRow.ID?
  @State private var inspectorPresented = false
  @State private var importing = false
  @State private var confirmation: Confirmation?
  @State private var actionError: String?

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
        ToolbarItem {
          Menu {
            if let row = selectedRow {
              actionButtons(for: row)
            }
          } label: {
            Label("Actions", systemImage: "ellipsis.circle")
          }
          .disabled(selectedRow == nil)
          .help("Actions for the selected model")
        }
        ToolbarItem {
          Button("Inspector", systemImage: "info.circle") { inspectorPresented.toggle() }
            .keyboardShortcut("i", modifiers: .command)
            .help(inspectorPresented ? "Hide the model's details" : "Show the model's details")
        }
      }
      .inspector(isPresented: $inspectorPresented) {
        Group {
          if let row = selectedRow {
            ModelDetailView(row: row)
          } else {
            ContentUnavailableView("No Model Selected", systemImage: "shippingbox", description: Text("Select a model to see its details."))
          }
        }
        .inspectorColumnWidth(min: 280, ideal: 320, max: 440)
      }
      .fileImporter(isPresented: $importing, allowedContentTypes: [ModelsView.onnxType]) { result in
        switch result {
        case let .success(url): models.importModel(at: url)
        case let .failure(error): actionError = error.localizedDescription
        }
      }
      .alert(Text(confirmationTitle), isPresented: confirmationPresented, presenting: confirmation) { item in
        Button(confirmButtonTitle(item), role: isDestructive(item) ? .destructive : nil) { perform(item) }
        Button("Cancel", role: .cancel) {}
      } message: { item in
        Text(confirmationMessage(item))
      }
      .alert("Couldn't complete the action", isPresented: errorPresented) {
        Button("OK") { actionError = nil }
      } message: {
        Text(actionError ?? "")
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
    return ModelsView.diskSummary(models: disk.modelsBytes, engines: disk.enginesBytes, free: disk.freeBytes)
  }

  // MARK: Actions

  private var selectedRow: ModelRow? {
    guard let selection else { return nil }
    return models.rows.first { $0.id == selection }
  }

  /// Only what applies to this model, as a context menu should be.
  @ViewBuilder
  private func actionButtons(for row: ModelRow) -> some View {
    if ModelStore.canUse(row) {
      Button("Use Model") { startUse(row) }
    }
    if case .downloading = row.status {
      Button("Cancel Download") { models.cancelDownload(row) }
    }
    if row.status == .loaded {
      Button("Stop Using Model") { models.unload() }
    }
    if let url = revealURL(row) {
      Divider()
      Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
    }
    if hasModelFile(row) || !row.preparedFor.isEmpty {
      Divider()
    }
    if hasModelFile(row) {
      Button("Delete Download…", role: .destructive) { confirmation = .deleteDownload(row) }
    }
    if !row.preparedFor.isEmpty {
      Button("Delete Prepared Engines…", role: .destructive) { confirmation = .deleteEngines(row) }
    }
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

  private var errorPresented: Binding<Bool> {
    Binding(get: { actionError != nil }, set: { if !$0 { actionError = nil } })
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

  // MARK: Formatting

  /// What Use Model is about to do, for its tooltip.
  static func useHelp(_ row: ModelRow) -> String {
    switch row.status {
    case .notDownloaded:
      let size = row.bytes.map { " \(ByteCount.string($0))" } ?? ""
      return "Downloads\(size), prepares it for this Mac and starts using it"
    case .prepared:
      return "Starts using it. It is prepared already, so this takes seconds."
    case .failed:
      return "Tries again"
    default:
      return "Prepares it for this Mac and starts using it"
    }
  }

  /// "Sep 1, 2026 · 766 MB · Prepared for CoreML": everything but the name and
  /// what is happening right now, on one line under the name.
  static func detailLine(_ row: ModelRow) -> String {
    var parts: [String] = []
    let built = BuildTime.text(row.buildTime)
    if !built.isEmpty { parts.append(built) }
    if let bytes = row.bytes { parts.append(ByteCount.string(bytes)) }
    let prepared = preparedForText(row)
    if !prepared.isEmpty {
      parts.append("Prepared for \(prepared)")
    } else if row.status == .downloaded {
      parts.append("Downloaded")
    }
    if parts.isEmpty, row.isOrphan, let sha = row.sha256 {
      parts.append(String(sha.prefix(16)))
    }
    return parts.joined(separator: " · ")
  }

  /// "Downloads 2.3 GB · Prepared engines 6.9 GB · 13.6 GB available".
  static func diskSummary(models: Int64, engines: Int64, free: Int64) -> String {
    "Downloads \(ByteCount.string(models)) · Prepared engines \(ByteCount.string(engines)) · \(ByteCount.string(free)) available"
  }

  static let onnxType = UTType(filenameExtension: "onnx") ?? .data

  /// "CoreML, tinygrad": the backends a model already has an engine for.
  static func preparedForText(_ row: ModelRow) -> String {
    var seen: [String] = []
    for artifact in row.preparedFor {
      let name = backendName(artifact.backend)
      if !seen.contains(name) { seen.append(name) }
    }
    return seen.joined(separator: ", ")
  }

  static func backendName(_ backend: String) -> String {
    switch backend {
    case "ort": "CoreML"
    case "trt": "TensorRT"
    case "tinygrad": "tinygrad"
    default: backend
    }
  }
}

/// One model in the list: its name and tags, one line of facts, and on the
/// trailing edge whatever applies now: Use Model, progress, or In Use.
struct ModelListRow: View {
  let row: ModelRow
  let isCheckingCatalog: Bool
  let use: () -> Void
  let cancel: () -> Void

  @Environment(\.backgroundProminence) private var prominence

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
        let detail = ModelsView.detailLine(row)
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
    case .loaded:
      Label {
        Text("In Use")
      } icon: {
        Image(systemName: "checkmark.circle.fill")
          .foregroundStyle(onSelection(.green))
      }
      .font(.callout.weight(.medium))
      .help("Jetlink is using this model")
    case let .downloading(frac, rateBps):
      HStack(spacing: 8) {
        progress(frac: frac, caption: ModelListRow.downloadCaption(frac: frac, rateBps: rateBps))
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
    case .unresolved:
      Text(isCheckingCatalog ? "Checking…" : "Unavailable")
        .font(.callout)
        .foregroundStyle(.secondary)
    case let .failed(detail):
      HStack(spacing: 10) {
        Label {
          Text("Failed")
        } icon: {
          Image(systemName: "exclamationmark.triangle.fill")
            .foregroundStyle(onSelection(.orange))
        }
        .font(.callout)
        .foregroundStyle(.secondary)
        .help(detail)
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
      .help(ModelsView.useHelp(row))
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

  /// A status colour, or the selection's own text colour on a selected row,
  /// where the colour would sit on the accent and vanish.
  private func onSelection(_ color: Color) -> AnyShapeStyle {
    prominence == .increased ? AnyShapeStyle(.primary) : AnyShapeStyle(color)
  }

  /// "Downloading 42%, 41 MB/s".
  static func downloadCaption(frac: Double, rateBps: Double) -> String {
    let percent = "Downloading \(Int((min(max(frac, 0), 1) * 100).rounded()))%"
    return rateBps > 0 ? "\(percent), \(ByteCount.rate(rateBps))" : percent
  }
}

/// A small capsule label next to a model's name. On a selected row it turns
/// to the selection's text colour, so an accent tag never sits on the accent.
struct ModelTag: View {
  let text: String
  let tone: Color
  @Environment(\.backgroundProminence) private var prominence

  init(_ text: String, tone: Color) {
    self.text = text
    self.tone = tone
  }

  var body: some View {
    let selected = prominence == .increased
    Text(text)
      .font(.caption.weight(.medium))
      .padding(.horizontal, 6)
      .padding(.vertical, 1)
      .foregroundStyle(selected ? AnyShapeStyle(.primary) : AnyShapeStyle(tone))
      .background(Capsule().fill(selected ? AnyShapeStyle(.white.opacity(0.2)) : AnyShapeStyle(tone.opacity(0.14))))
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
