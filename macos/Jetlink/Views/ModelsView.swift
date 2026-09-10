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
    case deleteDownload(ModelRow), deleteEngines(ModelRow), prepareSwitch(ModelRow)

    var row: ModelRow {
      switch self {
      case let .deleteDownload(row), let .deleteEngines(row), let .prepareSwitch(row): row
      }
    }

    var id: String {
      switch self {
      case .deleteDownload: "download-\(row.id)"
      case .deleteEngines: "engines-\(row.id)"
      case .prepareSwitch: "prepare-\(row.id)"
      }
    }
  }

  var body: some View {
    content
      .toolbar {
        ToolbarItem {
          Button("Refresh", systemImage: "arrow.clockwise") { models.refreshCatalog() }
            .keyboardShortcut("r", modifiers: [.command, .shift])
            .help("Refresh the model list")
        }
        ToolbarItem {
          Button("Add ONNX…", systemImage: "plus") { importing = true }
            .help("Add a model file from this Mac")
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
          Button("Show details", systemImage: "sidebar.trailing") { inspectorPresented.toggle() }
            .help("Show the details of the selected model")
        }
      }
      .inspector(isPresented: $inspectorPresented) {
        if let row = selectedRow {
          ModelDetailView(row: row)
        } else {
          Text("Select a model to see its details.")
            .foregroundStyle(.secondary)
            .padding()
        }
      }
      .fileImporter(isPresented: $importing, allowedContentTypes: [ModelsView.onnxType]) { result in
        switch result {
        case let .success(url): models.importModel(at: url)
        case let .failure(error): actionError = error.localizedDescription
        }
      }
      .alert(Text(confirmationTitle), isPresented: confirmationPresented, presenting: confirmation) { item in
        Button(confirmButtonTitle(item), role: isDestructive(item) ? .destructive : nil) { perform(item) }
        Button("Cancel", role: .cancel) { }
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
        Label("Server not running", systemImage: "cable.connector.slash")
      } description: {
        Text("Start the server to load the model list.")
      } actions: {
        Button("Start server") { server.start() }
      }
    } else if models.catalog == nil, models.rows.isEmpty {
      ProgressView("Loading model list…")
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    } else {
      VStack(spacing: 0) {
        table
        Divider()
        bottomBar
      }
    }
  }

  private var table: some View {
    Table(models.rows, selection: $selection) {
      TableColumn("Model") { row in
        modelCell(row)
      }
      .width(min: 260)
      TableColumn("Built") { row in
        Text(BuildTime.text(row.buildTime))
      }
      TableColumn("Size") { row in
        ByteCount(row.bytes)
      }
      TableColumn("Status") { row in
        ModelStatusLabel(row.status, isCheckingCatalog: models.catalog == nil)
      }
      TableColumn("Prepared for") { row in
        Text(ModelsView.preparedForText(row))
      }
    }
    .contextMenu(forSelectionType: ModelRow.ID.self) { ids in
      if let row = models.rows.first(where: { ids.contains($0.id) }) {
        actionButtons(for: row)
      }
    }
  }

  private func modelCell(_ row: ModelRow) -> some View {
    VStack(alignment: .leading, spacing: 2) {
      HStack(spacing: 6) {
        Text(row.name)
        if row.isDefault { tag("Default", tone: .accentColor) }
        if row.isRequestedByComma { tag("Comma", tone: .green) }
        if row.isLocal { tag("Local", tone: .secondary) }
      }
      Text(row.ref.map { String($0.prefix(10)) } ?? row.sha256.map { String($0.prefix(16)) } ?? "")
        .font(.system(.caption, design: .monospaced))
        .foregroundStyle(.secondary)
    }
  }

  private func tag(_ text: String, tone: Color) -> some View {
    Text(text)
      .font(.caption2)
      .padding(.horizontal, 5)
      .padding(.vertical, 1)
      .background(Capsule().fill(tone.opacity(0.15)))
      .foregroundStyle(tone)
  }

  private var bottomBar: some View {
    HStack {
      Text(diskSummary)
        .font(.callout)
        .foregroundStyle(.secondary)
      Spacer()
      if let error = models.catalog?.error, !error.isEmpty {
        Label(error, systemImage: "exclamationmark.triangle")
          .font(.callout)
          .foregroundStyle(.orange)
      }
    }
    .padding(.horizontal, 12)
    .padding(.vertical, 6)
  }

  private var diskSummary: String {
    guard let disk = models.inventory?.disk else { return "" }
    return "Models \(ByteCount.string(disk.modelsBytes)), prepared engines \(ByteCount.string(disk.enginesBytes)), \(ByteCount.string(disk.freeBytes)) free on this disk"
  }

  // MARK: Actions

  private var selectedRow: ModelRow? {
    guard let selection else { return nil }
    return models.rows.first { $0.id == selection }
  }

  @ViewBuilder
  private func actionButtons(for row: ModelRow) -> some View {
    Button("Download") { models.download(row) }
      .disabled(!canDownload(row))
    Button("Cancel download") { models.cancelDownload(row) }
      .disabled(!canCancelDownload(row))
    Button("Prepare") { startPrepare(row) }
      .disabled(!canPrepare(row))
    Button("Load") { startPrepare(row) }
      .disabled(!canLoad(row))
    Button("Unload") { models.unload() }
      .disabled(!canUnload(row))
    Divider()
    Button("Reveal in Finder") {
      if let url = revealURL(row) {
        NSWorkspace.shared.activateFileViewerSelecting([url])
      }
    }
    .disabled(revealURL(row) == nil)
    Divider()
    Button("Delete download…", role: .destructive) { confirmation = .deleteDownload(row) }
      .disabled(!hasModelFile(row))
    Button("Delete prepared engines…", role: .destructive) { confirmation = .deleteEngines(row) }
      .disabled(row.preparedFor.isEmpty)
  }

  private func startPrepare(_ row: ModelRow) {
    if models.prepareNeedsConfirmation(row) {
      confirmation = .prepareSwitch(row)
    } else {
      models.prepare(row)
    }
  }

  private func canDownload(_ row: ModelRow) -> Bool {
    guard row.sha256 != nil else { return false }
    if case .failed = row.status { return true }
    return row.status == .notDownloaded
  }

  private func canCancelDownload(_ row: ModelRow) -> Bool {
    if case .downloading = row.status { return true }
    return false
  }

  private func canPrepare(_ row: ModelRow) -> Bool {
    row.status == .downloaded
  }

  private func canLoad(_ row: ModelRow) -> Bool {
    row.status == .prepared
  }

  private func canUnload(_ row: ModelRow) -> Bool {
    row.status == .loaded
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
    case let .prepareSwitch(row): models.prepare(row)
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
    case let .prepareSwitch(row): "Prepare \(row.name)?"
    case nil: ""
    }
  }

  private func confirmButtonTitle(_ item: Confirmation) -> String {
    switch item {
    case .deleteDownload, .deleteEngines: "Delete"
    case .prepareSwitch: "Prepare"
    }
  }

  private func isDestructive(_ item: Confirmation) -> Bool {
    switch item {
    case .deleteDownload, .deleteEngines: true
    case .prepareSwitch: false
    }
  }

  private func confirmationMessage(_ item: Confirmation) -> String {
    switch item {
    case let .deleteDownload(row):
      let size = row.bytes.map { " (\(ByteCount.string($0)))" } ?? ""
      return "The model file for \(row.name)\(size) is deleted. The prepared engine stays, so the comma can use this model without downloading it again."
    case let .deleteEngines(row):
      let bytes = row.preparedFor.reduce(Int64(0)) { $0 + $1.bytes }
      let count = row.preparedFor.count
      let engines = count == 1 ? "engine" : "engines"
      var text = "\(count) prepared \(engines) for \(row.name), \(ByteCount.string(bytes)) in all, are deleted. Preparing the model again takes as long as the first time."
      if row.isLoaded {
        text += " The model is loaded now, so it is unloaded first."
      }
      return text
    case let .prepareSwitch(row):
      let current = models.rows.first { $0.isLoaded }?.name ?? "another model"
      return "The comma is connected and using \(current). Preparing \(row.name) switches the server to it; the comma falls back to its small model until it reconnects and that model is loaded."
    }
  }

  // MARK: Formatting

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

#Preview("Models") {
  ModelsView()
    .environment(ServerStore.preview(runState: .serving, info: PreviewData.serverInfo, link: PreviewData.linkConnected,
                                     engine: PreviewData.engineReady, stats: PreviewData.stats))
    .environment(ModelStore.preview(catalog: PreviewData.catalog, inventory: PreviewData.inventory,
                                    downloads: [PreviewData.download.sha256: PreviewData.download], engine: PreviewData.engineReady))
    .frame(width: 860, height: 480)
}

#Preview("Server stopped") {
  ModelsView()
    .environment(ServerStore.preview(runState: .stopped, info: nil, link: PreviewData.linkWaiting, engine: PreviewData.engineNone))
    .environment(ModelStore.preview(catalog: nil, inventory: nil, engine: PreviewData.engineNone))
    .frame(width: 860, height: 480)
}
