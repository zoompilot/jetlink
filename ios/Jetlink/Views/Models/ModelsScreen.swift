import JetlinkKit
import JetlinkUI
import SwiftUI
import UniformTypeIdentifiers

/// The catalog and what is on this device. One button per model, as the App
/// Store has one: Get downloads, prepares and loads it; Use loads one that is
/// here; a ring shows a download, and a tap on it stops it.
struct ModelsScreen: View {
  @Environment(AppModel.self) private var app
  @State private var importing = false
  @State private var confirmation: Confirmation?
  @State private var importError: String?
  /// Copies of picked files, removed once their import has finished.
  @State private var stagedImports: Set<String> = []

  enum Confirmation: Identifiable {
    case delete(ModelRow), switchModel(ModelRow)
    /// Too little room to download or prepare it (refused when `blocking`), or to keep it on the Neural Engine.
    case notEnoughSpace(ModelRow, StorageShortfall)

    var row: ModelRow {
      switch self {
      case let .delete(row), let .switchModel(row), let .notEnoughSpace(row, _): row
      }
    }

    var id: String {
      switch self {
      case .delete: "delete-\(row.id)"
      case .switchModel: "use-\(row.id)"
      case .notEnoughSpace: "space-\(row.id)"
      }
    }
  }

  private var models: ModelStore { app.models }

  var body: some View {
    NavigationStack {
      content
        .navigationTitle("Models")
        .toolbar {
          ToolbarItem(placement: .topBarTrailing) {
            Button("Add Model File", systemImage: "plus") { importing = true }
          }
        }
    }
    .fileImporter(isPresented: $importing, allowedContentTypes: [ModelsScreen.onnxType]) { result in
      switch result {
      case let .success(url): stageImport(url)
      case let .failure(error): importError = error.localizedDescription
      }
    }
    .onChange(of: models.imports) { _, imports in
      for event in imports where ["done", "failed"].contains(event.state) && stagedImports.contains(event.path) {
        try? FileManager.default.removeItem(atPath: event.path)
        stagedImports.remove(event.path)
      }
    }
    .alert(confirmationTitle, isPresented: confirmationPresented, presenting: confirmation) { item in
      switch item {
      case let .delete(row):
        Button("Delete", role: .destructive) { models.forget(row, artifacts: true, model: true) }
        Button("Cancel", role: .cancel) {}
      case let .switchModel(row):
        Button("Use Model") { models.use(row, confirmedInterruption: true) }
        Button("Cancel", role: .cancel) {}
      case let .notEnoughSpace(row, short):
        // What cannot fit would stop part way and leave the disk full: the
        // server refuses it too. The rest fits, with too little left over.
        if short.blocking {
          Button("OK", role: .cancel) {}
        } else {
          Button(short.step == .download ? "Get Anyway" : "Use Anyway") { use(row) }
          Button("Cancel", role: .cancel) {}
        }
      }
    } message: { item in
      Text(confirmationMessage(item))
    }
    .alert("Couldn't Add File", isPresented: Binding(get: { importError != nil }, set: { if !$0 { importError = nil } })) {
      Button("OK") { importError = nil }
    } message: {
      Text(importError ?? "")
    }
    // The server's own words go to Logs; the alert says where to find them.
    .onChange(of: models.lastError) { _, error in
      if let error { app.server.note(.warning, error) }
    }
    .alert("Couldn't Complete", isPresented: Binding(get: { models.lastError != nil }, set: { if !$0 { models.clearError() } })) {
      Button("OK") { models.clearError() }
    } message: {
      Text("See Logs for details.")
    }
  }

  @ViewBuilder
  private var content: some View {
    if models.catalog == nil && models.rows.isEmpty {
      ProgressView()
        .controlSize(.large)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.groupedBackground)
    } else {
      List {
        Section {
          let available = models.rows.filter { !$0.isLocal && !$0.isOrphan }
          if available.isEmpty {
            catalogPlaceholder
          }
          rows(available)
        } header: {
          Text("Available")
        } footer: {
          if let error = models.catalog?.error, !error.isEmpty {
            Label("Couldn't refresh. Pull down to try again.", systemImage: "exclamationmark.triangle.fill")
          }
        }
        let local = models.rows.filter { $0.isLocal && !$0.isOrphan }
        if !local.isEmpty {
          Section("Added") { rows(local) }
        }
        let orphans = models.rows.filter(\.isOrphan)
        if !orphans.isEmpty {
          Section("Uploaded") { rows(orphans) }
        }
        if let disk = models.inventory?.disk {
          Section {
          } footer: {
            let sizes = [disk.modelsBytes, disk.enginesBytes, disk.freeBytes].map(ByteCount.string)
            Text("\(sizes[0]) Downloaded · \(sizes[1]) Prepared · \(sizes[2]) Free")
          }
        }
      }
      .listStyle(.insetGrouped)
      .readableWidth()
      .refreshable { models.refreshCatalog() }
    }
  }

  /// The Available section while the list has nothing in it: loading, or why not.
  @ViewBuilder
  private var catalogPlaceholder: some View {
    if let error = models.catalog?.error, !error.isEmpty {
      HStack {
        Label("Couldn't Load", systemImage: "exclamationmark.triangle.fill")
          .foregroundStyle(.secondary)
        Spacer()
        Button("Try Again") { models.refreshCatalog() }
          .buttonStyle(.bordered)
          .buttonBorderShape(.capsule)
      }
    } else {
      HStack(spacing: 10) {
        ProgressView()
        Text("Loading…")
          .foregroundStyle(.secondary)
      }
    }
  }

  private func rows(_ rows: [ModelRow]) -> some View {
    ForEach(rows) { row in
      ModelRowView(
        row: row,
        use: { startUse(row) },
        cancel: { models.cancelDownload(row) },
        stop: { models.unload() }
      )
      .swipeActions(edge: .trailing, allowsFullSwipe: false) {
        if row.hasFiles(in: models.inventory) {
          Button("Delete", systemImage: "trash", role: .destructive) { confirmation = .delete(row) }
        }
      }
      .contextMenu { menu(for: row) }
    }
  }

  @ViewBuilder
  private func menu(for row: ModelRow) -> some View {
    if ModelStore.canUse(row) {
      Button("Use Model", systemImage: "play.circle") { startUse(row) }
    }
    if case .downloading = row.status {
      Button("Stop Download", systemImage: "stop.circle") { models.cancelDownload(row) }
    }
    if row.status == .loaded {
      Button("Stop Using", systemImage: "stop.circle") { models.unload() }
    }
    if row.hasFiles(in: models.inventory) {
      Divider()
      Button("Delete", systemImage: "trash", role: .destructive) { confirmation = .delete(row) }
    }
  }

  // MARK: actions

  private func startUse(_ row: ModelRow) {
    if let short = models.storageShortfall(for: row) {
      confirmation = .notEnoughSpace(row, short)
    } else {
      use(row)
    }
  }

  private func use(_ row: ModelRow) {
    if models.useNeedsConfirmation(row) {
      // after the alert this may come from has gone
      Task { @MainActor in confirmation = .switchModel(row) }
    } else {
      models.use(row)
    }
  }

  /// A picked file is readable only while its security scope is open, and the
  /// import runs later on the server's own time, so it gets a copy of its own.
  private func stageImport(_ url: URL) {
    let scoped = url.startAccessingSecurityScopedResource()
    defer { if scoped { url.stopAccessingSecurityScopedResource() } }
    let staged = URL.temporaryDirectory.appending(path: "import-\(UUID().uuidString.prefix(8))-\(url.lastPathComponent)")
    do {
      try FileManager.default.copyItem(at: url, to: staged)
    } catch {
      importError = error.localizedDescription
      return
    }
    stagedImports.insert(staged.path)
    models.importModel(at: staged)
  }

  // MARK: alerts

  private var confirmationPresented: Binding<Bool> {
    Binding(get: { confirmation != nil }, set: { if !$0 { confirmation = nil } })
  }

  private var confirmationTitle: String {
    switch confirmation {
    case let .delete(row): "Delete \(row.displayName)?"
    case let .switchModel(row): "Use \(row.displayName)?"
    case let .notEnoughSpace(row, short): short.blocking ? "Not Enough Space for \(row.displayName)" : "Low Storage"
    case nil: ""
    }
  }

  private func confirmationMessage(_ item: Confirmation) -> String {
    switch item {
    case .delete:
      return "You can download it again later."
    case .switchModel:
      return "The comma uses its small model until this one is ready."
    case let .notEnoughSpace(_, short):
      return short.message
    }
  }

  static let onnxType = UTType(filenameExtension: "onnx") ?? .data
}

/// One model: its name, one line of facts or progress, and the button.
struct ModelRowView: View {
  let row: ModelRow
  let use: () -> Void
  let cancel: () -> Void
  let stop: () -> Void

  var body: some View {
    HStack(spacing: 12) {
      VStack(alignment: .leading, spacing: 2) {
        HStack(spacing: 6) {
          Text(row.isOrphan ? "Uploaded Model" : row.displayName)
            .lineLimit(1)
          if row.isDefault {
            ModelTag("Default", tone: .accentColor)
          }
        }
        Text(subtitle)
          .font(.subheadline)
          .foregroundStyle(subtitleTone)
          .lineLimit(1)
          .monospacedDigit()
      }
      Spacer(minLength: 8)
      accessory
    }
    .padding(.vertical, 2)
    .accessibilityElement(children: .combine)
  }

  private var subtitle: String {
    switch row.status {
    case let .downloading(frac, rateBps):
      let percent = frac.formatted(.percent.precision(.fractionLength(0)))
      return rateBps > 0 ? "Downloading · \(percent) · \(ByteCount.rate(rateBps))" : "Downloading · \(percent)"
    case let .preparing(stage, frac, msg):
      return ProgressRow.text(stage: stage, frac: frac, msg: msg)
    case .loaded:
      return row.isOrphan ? "In Use · \(row.sha256.map { String($0.prefix(12)) } ?? "")" : "In Use"
    case .failed:
      return "Failed"
    case .unresolved:
      return "Checking…"
    case .notDownloaded, .downloaded, .prepared:
      var parts: [String] = []
      if row.isOrphan, let sha = row.sha256 { parts.append(String(sha.prefix(12))) }
      if let bytes = row.bytes { parts.append(ByteCount.string(bytes)) }
      let built = BuildTime.text(row.buildTime)
      if !built.isEmpty { parts.append(built) }
      if row.status == .prepared { parts.append("Ready") }
      return parts.joined(separator: " · ")
    }
  }

  private var subtitleTone: Color {
    switch row.status {
    case .loaded: .green
    case .failed: .red
    default: .secondary
    }
  }

  @ViewBuilder
  private var accessory: some View {
    switch row.status {
    case .loaded:
      Menu {
        Button("Stop Using", systemImage: "stop.circle", action: stop)
      } label: {
        Image(systemName: "checkmark.circle.fill")
          .font(.title2)
          .foregroundStyle(.green)
      }
      .accessibilityLabel("In Use")
    case let .downloading(frac, _):
      Button(action: cancel) {
        ProgressRing(frac: frac, stoppable: true)
      }
      .buttonStyle(.plain)
      .accessibilityLabel("Stop Download")
    case let .preparing(_, frac, _):
      ProgressRing(frac: frac, stoppable: false)
    case .unresolved:
      EmptyView()
    case .notDownloaded:
      pill("Get")
    case .downloaded, .prepared:
      pill("Use")
    case .failed:
      pill("Retry")
    }
  }

  private func pill(_ title: String) -> some View {
    Button(title, action: use)
      .font(.subheadline.weight(.bold))
      .buttonStyle(.bordered)
      .buttonBorderShape(.capsule)
      .accessibilityHint(ModelFormatting.useHelp(row, device: "this \(ThisDevice.name)"))
  }
}

/// A small determinate ring, the App Store's download indicator: with a stop
/// square inside while it can be stopped.
struct ProgressRing: View {
  let frac: Double
  let stoppable: Bool

  var body: some View {
    // nothing to show yet, so it spins rather than sit at an empty ring
    if frac <= 0 && !stoppable {
      ProgressView()
        .frame(width: 28, height: 28)
    } else {
      ring
    }
  }

  private var ring: some View {
    ZStack {
      Circle()
        .stroke(Color.accentColor.opacity(0.2), lineWidth: 3)
      Circle()
        .trim(from: 0, to: min(max(frac, 0.02), 1))
        .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 3, lineCap: .round))
        .rotationEffect(.degrees(-90))
        .animation(.smooth, value: frac)
      if stoppable {
        RoundedRectangle(cornerRadius: 2)
          .fill(Color.accentColor)
          .frame(width: 9, height: 9)
      }
    }
    .frame(width: 28, height: 28)
  }
}
