// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.

import JetlinkKit
import SwiftUI
import UniformTypeIdentifiers

/// The large models: what is on the phone, what sunnypilot's catalog offers,
/// and getting either ready before a drive.
struct ModelsView: View {
  @Environment(AppModel.self) private var model
  @Environment(AppSettings.self) private var settings
  @State private var importing = false
  @State private var deleting: ModelOnDisk?

  var body: some View {
    Form {
      onPhoneSection
      catalogSection
      Section {
        Button("Import an ONNX file…", systemImage: "square.and.arrow.down") { importing = true }
          .disabled(model.importing != nil)
        if let frac = model.importing {
          ProgressView("Importing", value: frac)
        }
        if let error = model.importError {
          Text(error).font(.footnote).foregroundStyle(.red)
        }
      } footer: {
        Text(
          "The comma uploads the model it wants over the link, so none of this is required. Downloading and preparing at home, on Wi-Fi, saves the wait in the car."
        )
      }
    }
    .navigationTitle("Models")
    .refreshable { await model.refreshCatalog(force: true) }
    .fileImporter(isPresented: $importing, allowedContentTypes: [UTType(filenameExtension: "onnx") ?? .data]) { result in
      if case .success(let url) = result { model.importModel(from: url) }
    }
    .confirmationDialog(
      "Delete \(deleting?.name ?? "this model")?", isPresented: Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } }),
      titleVisibility: .visible
    ) {
      if let d = deleting {
        Button("Delete prepared engine", role: .destructive) { model.delete(d.sha256, engines: true, model: false) }
        Button("Delete model and engine", role: .destructive) { model.delete(d.sha256, engines: true, model: true) }
      }
      Button("Cancel", role: .cancel) {}
    } message: {
      Text("A deleted model is downloaded or uploaded again when it is needed.")
    }
  }

  // MARK: on the phone

  private var onPhoneSection: some View {
    Section {
      if model.onDisk.isEmpty {
        Text("No models yet").foregroundStyle(.secondary)
      }
      ForEach(model.onDisk) { m in
        VStack(alignment: .leading, spacing: 6) {
          HStack {
            VStack(alignment: .leading) {
              Text(m.name ?? "Unnamed model")
              Text("\(m.sha256.prefix(16))  \(ByteCountFormatter.string(fromByteCount: Int64(m.bytes), countStyle: .file))")
                .font(.caption).monospaced().foregroundStyle(.secondary)
            }
            Spacer()
            statusBadge(m.sha256)
          }
          if let c = model.choice(m.sha256), !c.measured.isEmpty {
            Text(measuredText(c)).font(.caption).foregroundStyle(.secondary)
          }
          HStack {
            if model.engine.sha256 == m.sha256 && model.engine.state == "ready" {
              Text("Loaded").font(.footnote).foregroundStyle(.green)
            } else {
              Button(model.prepared[m.sha256] == nil ? "Prepare" : "Load") { model.prepare(m.sha256) }
                .disabled(model.engine.isBusy)
            }
            if settings.compute == .auto && model.prepared[m.sha256] != nil {
              Button("Measure again") { model.recalibrate(m.sha256) }
                .disabled(model.engine.isBusy)
            }
            Spacer()
            Button("Delete", role: .destructive) { deleting = m }
              .disabled(model.engine.isBusy && model.engine.sha256 == m.sha256)
          }
          .buttonStyle(.borderless)
          .font(.footnote)
        }
        .padding(.vertical, 2)
      }
    } header: {
      Text("On this phone")
    } footer: {
      Text("\(ByteCountFormatter.string(fromByteCount: Int64(model.diskFree), countStyle: .file)) free. A prepared model takes about twice its size again.")
    }
  }

  @ViewBuilder
  private func statusBadge(_ sha: String) -> some View {
    if model.engine.sha256 == sha && model.engine.isBusy {
      ProgressView(value: min(max(model.engine.frac, 0), 1)).frame(width: 60)
    } else if let units = model.prepared[sha] {
      Text(units.label).font(.caption).padding(.horizontal, 6).padding(.vertical, 2)
        .background(.green.opacity(0.15), in: Capsule())
    } else {
      Text("Not prepared").font(.caption).foregroundStyle(.secondary)
    }
  }

  private func measuredText(_ c: EngineHost.Choice) -> String {
    let parts = [ComputeUnits.ane, .coreml].compactMap { u -> String? in
      guard let r = c.measured[u] else { return nil }
      return "\(u.label) p99 \(String(format: "%.1f", r.p99)) ms"
    }
    return "Measured at 20 Hz: " + parts.joined(separator: ", ") + ". Chose the \(c.units.label)."
  }

  // MARK: the catalog

  private var catalogSection: some View {
    Section {
      if model.catalog.isEmpty && model.catalogLoading {
        ProgressView()
      }
      ForEach(model.catalog) { m in
        HStack {
          VStack(alignment: .leading) {
            HStack(spacing: 4) {
              Text(m.name)
              if m.ref == RegistryConstants.defaultBigModelRef {
                Text("default").font(.caption2).foregroundStyle(.secondary)
              }
            }
            Text(
              [m.shortName, m.bytes.map { ByteCountFormatter.string(fromByteCount: Int64($0), countStyle: .file) }]
                .compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: " · ")
            )
            .font(.caption).foregroundStyle(.secondary)
          }
          Spacer()
          catalogAction(m)
        }
      }
      if let error = model.catalogError {
        Text(error).font(.footnote).foregroundStyle(.orange)
      }
    } header: {
      Text("sunnypilot catalog")
    } footer: {
      Text("The large (chestnut) driving models the comma can ask for. Pull down to refresh.")
    }
  }

  @ViewBuilder
  private func catalogAction(_ m: CatalogModel) -> some View {
    switch model.downloads[m.ref] {
    case .running(let bytes, let total)?:
      HStack {
        ProgressView(value: total > 0 ? Double(bytes) / Double(total) : 0).frame(width: 70)
        Button("Cancel", systemImage: "xmark.circle.fill") { model.cancelDownload(m.ref) }
          .labelStyle(.iconOnly).buttonStyle(.borderless)
      }
    case .failed(let error)?:
      Button("Retry") { model.download(m) }
        .buttonStyle(.borderless)
        .help(error)
    case nil:
      if model.isDownloaded(m.sha256) {
        Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
      } else {
        Button("Download", systemImage: "arrow.down.circle") { model.download(m) }
          .labelStyle(.iconOnly).buttonStyle(.borderless)
      }
    }
  }
}
