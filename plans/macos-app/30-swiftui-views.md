# 30. SwiftUI: windows, menu bar extra, settings

**Owner: agent D.** Files you create, all under `macos/Jetlink/`:
`App/JetlinkApp.swift`, `App/AppDelegate.swift`, `Views/MainWindow.swift`,
`Views/StatusView.swift`, `Views/ModelsView.swift`, `Views/ModelDetailView.swift`,
`Views/LogsView.swift`, `Views/SettingsView.swift`, `Views/MenuBarView.swift`,
`Views/Components/StatusBadge.swift`, `Views/Components/ProgressRow.swift`,
`Views/Components/ByteCount.swift`, `Views/Components/ModelStatusLabel.swift`,
`Views/Previews/PreviewData.swift`, and `macos/JetlinkTests/FormattingTests.swift`.
Read first: `01-contracts.md` section 7 (the store types you consume) and
this file. Build against the stores' declared API; until agent C lands, use
`PreviewData` stand-ins with the same shape (a `protocol`-free approach:
construct real store instances in previews and set state through an
`@testable` internal `preview(...)` factory that C is asked to provide; if it
is not there yet, write the previews with static data and leave a `// TODO(C)`).

## Design principles

Native, plain, quiet. This is a utility that sits in the background for hours.
Use system controls with their default styling, `Form` with `.formStyle(.grouped)`
for read-only information panels, `Table` for the model list, standard toolbar
placement, SF Symbols, `Text` with `.secondary` for detail. No custom colours
except the semantic ones on `StatusBadge`. No animation beyond what the system
does. Every control has an accessibility label where its icon is the only
content. Every long-running state shows progress and a way to cancel. Errors
are shown where they happen (inline under the row or in an alert), in one
sentence, with the server's own wording when it came from the server.

Sentence case for everything ("Start server", not "Start Server"), the macOS
convention for buttons and menu items in this style guide; window and sidebar
titles are title case ("Models"). No trailing periods on labels; full
sentences in explanatory text end with a period.

## `App/JetlinkApp.swift`

```swift
@main struct JetlinkApp: App {
  @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
  @State private var appState = AppState()
  var body: some Scene {
    Window("Jetlink", id: "main") { MainWindow().environment(appState) }
      .defaultSize(width: 860, height: 560)
      .commands { AppCommands(appState: appState) }
    MenuBarExtra("Jetlink", systemImage: menuBarSymbol) { MenuBarView().environment(appState) }
      .menuBarExtraStyle(.menu)
    Settings { SettingsView().environment(appState) }
  }
}
```

`menuBarSymbol` reflects `serverStore.runState` and `link`: stopped
`cable.connector.slash`, starting/stopping `cable.connector` (dimmed not
possible in a menu bar symbol; just the plain symbol), serving and waiting
`cable.connector`, connected `cable.connector.horizontal`... keep it to three:
`cable.connector.slash` (stopped or failed), `cable.connector` (serving, no
comma), `car.fill` (comma connected). Pass `appState` into the environment
with `.environment(appState)` and read stores through it (`@Environment(AppState.self)`).

`AppCommands`: replace `.newItem` with nothing (no new windows); add a
"Server" menu with "Start server" (⌘R when stopped), "Stop server" (⌘.),
"Restart server"; add "Refresh model list" (⇧⌘R) under "Models"? Put both in
one "Server" menu to keep the menu bar small. "Reveal cache in Finder" too.

## `App/AppDelegate.swift`

`NSApplicationDelegate`. `applicationShouldTerminate`: if the server is
running, call `appState.applicationWillTerminate()` in a `Task` and return
`.terminateLater`, then `NSApp.reply(toApplicationShouldTerminate: true)` when
it finishes (cap at 20 s). `applicationShouldHandleReopen` returns true and
opens the main window (use `@Environment(\.openWindow)` through a stored
closure set by the app, or `NSApp.windows.first?.makeKeyAndOrderFront`).
`applicationDidFinishLaunching`: nothing beyond logging; `AppState.init`
starts the server when the setting says so.

## `Views/MainWindow.swift`

`NavigationSplitView` with a `List(selection:)` sidebar of three items:
Status (`gauge.with.dots.needle.33percent`), Models (`shippingbox`), Logs
(`doc.text`). Column widths: sidebar `min: 160, ideal: 180`. Detail shows the
selected view. Toolbar (on the detail): a `Button` that is "Start server"
(`play.fill`) when stopped/failed, "Stop server" (`stop.fill`) when serving,
disabled with a `ProgressView().controlSize(.small)` when starting/stopping;
next to it a `Text` with the one-line status from `StatusBadge.summary`.
Toolbar items use `.help()` tooltips. Default selection: Status; if the
inventory has no artifacts and the catalog is loaded, still Status (its empty
state points to Models).

## `Views/StatusView.swift`

A `Form` (`.formStyle(.grouped)`) with three sections.

**Server**
- `LabeledContent("State") { StatusBadge(serverState) }`: Stopped (gray),
  Starting… (blue, with a small `ProgressView`), Serving (green), Stopping…
  (gray), Failed (red).
- "Backend": `"CoreML on the GPU"` for `ort`/`coreml-*`, `"CoreML with the
  Neural Engine"` for `ane`, `"tinygrad on Metal"` for `tinygrad`, else the
  raw backend name; secondary text: runtime version and device
  (`info.device` verbatim, e.g. `coreml-Apple_M1_Pro`).
- "Uptime": relative from `startedAt` (`Date.RelativeFormatStyle` is wrong for
  durations; use `Duration.UnitsFormatStyle` `hours, minutes`).
- When `.failed`, a red `Text(lastFailure)` in a monospaced 12 pt block with
  a "Show logs" button that selects the Logs item.

**Comma**
- "Link": Waiting for comma (gray) / Connected over USB (green) /
  Connected over TCP from host:port / Disconnected (orange) with `detail`
  as secondary text when non-empty.
- When connected and `stats` is non-nil: "Frames" (`frames` formatted with
  grouping), "Rate" (`fps` one decimal + " per second"), "Frame time" (mean /
  p99 / max in ms, one decimal, as `"31.2 ms mean, 38.0 ms p99, 41.5 ms max"`),
  "GPU time" (mean ms), "Slow frames" (count, red when > 0, with help text
  "Frames over 60 ms in the last second").
- Explanatory footer (section footer text): "The comma connects when it is
  plugged into a USB-A port with an A-to-C data cable. The small model keeps
  driving whenever the link is down."

**Engine**
- "Model": the name (via `modelStore.rows.first { $0.sha256 == engine.sha256 }`)
  or `"None"`; secondary: `sha256.prefix(16)`.
- "State": None (gray) / Preparing (blue) / Loading (blue) / Ready (green) / Failed (red).
- When building or loading: `ProgressRow(stage:frac:msg:)`, a `ProgressView(value:)`
  with the `msg` beneath in secondary text. For CoreML the server's message
  already says "compiling for CoreML, 3 min elapsed; the big model takes 11 min
  on an M1 Pro", show it verbatim.
- When failed: `detail` in red.
- Empty state when `engine.state == .none` and no artifacts exist: a
  `ContentUnavailableView("No model prepared", systemImage: "shippingbox",
  description: Text("Download and prepare the model your comma uses in Models. Keep Jetlink running afterwards; the model stays loaded and the comma connects to it immediately."))`
  with a "Open Models" button.

Buttons row at the bottom (section with no header): "Reveal cache in Finder"
(`NSWorkspace.shared.activateFileViewerSelecting([cacheURL])`), and when a
model is loaded, "Unload model" (confirmation: "Unload the model? The comma
will fall back to its small model until a model is loaded again.").

## `Views/ModelsView.swift`

Top: a `Table(rows, selection: $selection)` with columns:
- "Model": `Text(row.name)` with trailing small tags: "Default" (`.tint`),
  "Comma" (when `isRequestedByComma`, green), "Local" (when `isLocal`).
  Secondary line: `ref.prefix(10)` or `sha256.prefix(16)`. Column `width(min: 260)`.
- "Built": `buildTime` shown as a date (`Date.FormatStyle(date: .abbreviated, time: .omitted)`);
  parse ISO-8601 with `ISO8601DateFormatter`; empty when unknown.
- "Size": `ByteCount(bytes)`, or an empty cell when unknown (no dash placeholder).
- "Status": `ModelStatusLabel(row.status)`: Not downloaded (secondary),
  Downloading 42% (with a thin `ProgressView(value:)` and rate `41 MB/s`),
  Downloaded, Preparing (progress + msg), Prepared, Loaded (green bold),
  Failed (red, `detail` on hover via `.help`). "Unresolved" shows as "Checking…"
  while a catalog refresh runs, else "Unknown size".
- "Prepared for": a compact list of `preparedFor` as `backend` names joined
  by ", " (e.g. "CoreML, tinygrad"); empty when none.

Row context menu (`.contextMenu(forSelectionType: ModelRow.ID.self)`) and the
same actions in the toolbar's "Actions" `Menu` (ellipsis.circle) for the
selection: "Download", "Cancel download", "Prepare", "Load", "Unload",
"Reveal in Finder", "Delete download…", "Delete prepared engines…". Enablement:

| Action | Enabled when |
| --- | --- |
| Download | status is notDownloaded or failed, and sha known |
| Cancel download | status is downloading |
| Prepare | status is downloaded (no current artifact) |
| Load | status is prepared (artifact exists, not loaded) |
| Unload | status is loaded |
| Reveal in Finder | a model file or artifact exists |
| Delete download… | a model file exists; confirmation says the prepared engine stays |
| Delete prepared engines… | any artifact exists; confirmation lists sizes; if loaded, says it will be unloaded first |

Prepare/Load when `modelStore.prepareNeedsConfirmation(row)`: alert "The comma
is connected and using <current model name>. Preparing <row.name> switches the
server to it; the comma falls back to its small model until it reconnects and
that model is loaded." Buttons "Prepare" / "Cancel".

Toolbar: "Refresh" (`arrow.clockwise`, ⇧⌘R) → `refreshCatalog()`; "Add ONNX…"
(`plus`) → `.fileImporter` for `UTType(filenameExtension: "onnx")`, calls
`importModel(at:)`; the "Actions" menu. Bottom bar (a thin `HStack` under the
table): disk summary "Models 1.5 GB, prepared engines 5.5 GB, 120 GB free on
this disk" from `inventory.disk`; on the right, when a catalog `error` is set,
an orange `exclamationmark.triangle` with the error as text.

When `catalog == nil` and the server is stopped: `ContentUnavailableView("Server not running", systemImage: "cable.connector.slash", description: Text("Start the server to load the model list."))`
with a "Start server" button. While the first catalog fetch runs: a
`ProgressView("Loading model list…")`.

Selection shows `ModelDetailView` in an `.inspector(isPresented:)` toggled by
a toolbar `sidebar.trailing` button, default hidden. Detail: name, ref (full,
selectable, monospaced), SHA-256 (full, selectable, monospaced), size, build
time, checkpoint (from the current artifact), status, and a list of
`preparedFor` entries with backend, device, runtime version, built date, build
duration, size, each with a "Delete…" button.

## `Views/LogsView.swift`

A `ScrollViewReader` + `ScrollView` + `LazyVStack(alignment: .leading)` of
`Text(line).font(.system(.caption, design: .monospaced)).textSelection(.enabled)`.
Filter field in the toolbar (`.searchable` scoped to this view), "Auto-scroll"
`Toggle` (default on; turns off when the user scrolls up; use
`onScrollGeometryChange` to detect), "Copy all", "Clear", "Reveal log file".
Colour lines containing ` ERROR ` red and ` WARNING` orange (the server's
format is `%(asctime)s %(levelname)-7s %(name)s: %(message)s`). Cap rendering
to the buffer's 5000 lines; the file has the rest.

## `Views/SettingsView.swift`

`TabView` with two tabs.

**General** (`gear`)
- "Start server when Jetlink opens" `Toggle`.
- "Open Jetlink at login" `Toggle` bound to `LoginItem`; if
  `requiresApproval`, a caption "Approve Jetlink in System Settings > General >
  Login Items." with a button that opens
  `x-apple.systempreferences:com.apple.LoginItems-Settings.extension`.
- "Keep the Mac awake while serving" `Toggle`; caption "Only when connected to
  power. On battery, keep the lid open."
- "Cache folder": path in a `TextField` (read-only look: `.disabled(true)` is
  wrong for selection; use `Text` with `.textSelection`) plus "Choose…"
  (`NSOpenPanel`, directories only) and "Reveal". Caption: "Models and prepared
  engines. A CoreML engine is about 5.5 GB. Changing the folder takes effect
  when the server restarts." Changing it while serving offers "Restart now".

**Server** (`cpu`)
- "Backend" `Picker`: Automatic (recommended) / CoreML on the GPU / CoreML with
  the Neural Engine / tinygrad on Metal. Caption changes with the choice:
  auto and coreml: "About 43 ms a frame on an M1 Pro. Preparing a model takes
  about 9 minutes, and loading one again takes as long, so keep Jetlink
  running."; ane: "Faster back to back, slower at the comma's 20 Hz on an M1 Pro.
  Measure on your Mac before using it in the car."; tinygrad: "Loads in a
  second. About 66 ms a frame on an M1 Pro, which is over the 50 ms budget;
  a newer Mac may be under it."
- "Connection" `Picker`: USB (the comma) / TCP (bench client); "Port" field
  shown for TCP, default 5599.
- "Log level" `Picker`: Normal (INFO) / Verbose (DEBUG).
- Footer: "Changes apply when the server restarts." with a "Restart server"
  button enabled while serving.
- Advanced disclosure: "Python interpreter override" text field bound to
  `pythonOverride` with caption "For development. Leave empty to use the
  bundled runtime." and a line showing `EmbeddedPython.manifest()` versions
  ("Bundled: Python 3.14.7, onnxruntime 1.29.0, tinygrad e837e367").

## `Views/MenuBarView.swift`

Menu content (`.menuBarExtraStyle(.menu)` renders `Button`s as items):
- A disabled item with the status summary ("Serving, waiting for comma";
  "Comma connected, 19.9 frames per second"; "Stopped"; "Failed").
- A disabled item with the model line ("Loaded: BMRLNAP Model v4";
  "Preparing: 43%"; "No model").
- Divider. "Start server" / "Stop server". "Open Jetlink" (opens the main
  window via `openWindow(id: "main")` and `NSApp.activate`). Divider.
  "Quit Jetlink" (⌘Q) → `NSApp.terminate(nil)`.

## Components

- `StatusBadge(text:tone:)`: a capsule with a 6 pt dot and text; tones
  `.neutral (secondary)`, `.info (blue)`, `.good (green)`, `.warning (orange)`,
  `.bad (red)`. `static func summary(runState:link:engine:) -> (String, Tone)`
  used by the toolbar and the menu bar.
- `ProgressRow(stage:frac:msg:)`: determinate when `frac > 0`, indeterminate
  otherwise; stage names mapped: upload "Receiving model", patch "Preparing
  the model", parse "Reading the model", build "Building", save "Saving",
  load "Loading", failed "Failed".
- `ByteCount`: `ByteCountFormatStyle(style: .file)`; `static func rate(_ bps: Double) -> String` ("41.2 MB/s").
- `ModelStatusLabel(status:)` as described.

## Errors (exact strings)

- No embedded Python: "This build has no bundled Python runtime. Run `make
  python` in macos/, or set JETLINK_PYTHON to a Python 3.14 interpreter with
  the jetlink package installed." (shown in the Status view's failed block).
- Server startup failure: "The server could not start." followed by the last
  20 log lines.
- Action failed: the reply's `error` verbatim, in a `.alert` titled
  "Couldn't complete the action" (Apple's own contraction style in alerts is
  acceptable; use it consistently).

## Tests (`FormattingTests.swift`)

`StatusBadge.summary` for each combination; `ByteCount.rate`; the
build-time date parsing; `ProgressRow` stage mapping.

## Report back

Screenshots are not required (no display in CI). Report which views have
previews that compile, and any store API you needed that `01-contracts.md`
did not declare.

## Polish pass (decided 2026-09-09 from screenshots of the running app)

These override the sections above where they differ. All strings stay
sentence case, no em dashes.

1. **Toolbar.** `MainWindow` puts the summary in `ToolbarItem(placement: .principal)`
   as a `StatusBadge(text:tone:)` (dot plus text), and the Start/Stop button in
   `.primaryAction`. The summary must never truncate at the default window size.
2. **Summary strings** (`StatusBadge.summary`, also the menu bar's first line):
   stopped "Stopped"; starting "Starting…"; stopping "Stopping…"; failed "Failed";
   serving and waiting "Waiting for comma"; serving and connected "Comma connected";
   connected while building or loading "Comma connected, preparing"; connected and
   failed "Comma connected, model failed"; waiting while building or loading
   "Preparing a model"; waiting and failed "Model failed"; disconnected
   "Comma disconnected". The menu bar adds the rate when stats exist:
   "Comma connected, 19.9 fps".
3. **Short model names.** Add `ModelRow.displayName`: the name with a trailing
   parenthesised date removed (" (August 30, 2026)" and similar; regex
   ` \([A-Za-z]+ \d{1,2}, \d{4}\)$`). Use it in the table's Model column, the menu
   bar model line ("Loaded: BMRLNAP Model v4"), the Status view's Engine row, and
   every confirmation. The inspector shows the full name.
4. **Table widths.** Model `.width(min: 220, ideal: 320)`; Built `.width(min: 90, ideal: 100)`;
   Size `.width(min: 70, ideal: 80)`; Status `.width(min: 150, ideal: 170)`;
   Prepared for `.width(min: 110, ideal: 130)`. Inspector
   `.inspectorColumnWidth(min: 280, ideal: 320, max: 440)`. Window default size
   1000 by 640, minimum 860 by 540 (`.frame(minWidth:minHeight:)` on `MainWindow`).
5. **Bottom bar text.** "Models 766 MB, engines 777.2 MB, 23.2 GB free".
6. **Status view, Backend row.** Second line is "<runtime> <version>, <device>":
   runtime "onnxruntime" for ort and "tinygrad" for tinygrad; version with any
   "+local" suffix removed; device with the prefix up to and including the first
   "-" removed and underscores turned into spaces ("METAL-Apple_M1_Pro" becomes
   "Apple M1 Pro"). Example: "tinygrad 0.14.0, Apple M1 Pro".
7. **Status view, Link row.** Show the detail line only when the link is
   disconnected (the LinkError text) or when waiting over TCP (the listening
   address). Hide the USB "waiting for a jetlink gadget" detail; the badge says it.
8. **Status view, empty engine state.** No `ContentUnavailableView` inside the form.
   A compact block: `Label("No model prepared", systemImage: "shippingbox")` in
   `.headline`, the description in `.callout` secondary, and the "Open Models"
   button, left aligned, about three lines tall.
9. **Logs view.** The Auto-scroll control leaves the toolbar: a bottom bar with a
   checkbox "Follow new lines" on the left and "1,234 lines" (`.secondary`) on
   the right. The toolbar keeps Copy all, Clear, Reveal log file and the search field.
10. **Settings, Backend picker.** The first choice reads "Automatic (CoreML on the
    GPU)"; its caption starts "Recommended. " followed by the existing text.
11. **Inspector.** Long values (ref, SHA-256, checkpoint, artifact paths) sit under
    their label, left aligned, monospaced `.callout`, `.textSelection(.enabled)`,
    `.fixedSize(horizontal: false, vertical: true)`, no hyphenated wrapping. Short
    values (size, built, status) stay as `LabeledContent`.
12. **Sizes.** `ByteCount` uses at most one decimal and drops it when the value is
    a whole number of units ("766 MB", "1.8 GB", "23.2 GB").

## Text audit (decided 2026-09-09, after the polish pass)

The status indicator already shows the state, so text never repeats it and
every sentence is as short as it can be. These replace the earlier strings.

Summary (`StatusBadge.summary`, toolbar and menu bar first line): "Stopped",
"Starting…", "Stopping…", "Failed", "Waiting for comma", "Comma connected",
"Comma disconnected"; while a job runs, "Preparing model" (building) or
"Loading model" (loading) whether or not the comma is connected; a failed job
"Model failed". Menu bar model line: "Loaded: BMRLNAP Model v4",
"Preparing model, 43%", "Loading model, 43%", "No model", "Model failed".

| Where | Text |
| --- | --- |
| Status, Comma footer | "Plug the comma into a USB-A port with an A-to-C data cable. Until it connects, the comma drives on its small model." |
| Status, empty engine block | "Download and prepare the model your comma uses in Models. Leave Jetlink running afterwards so it stays loaded." |
| Status, unload confirmation | title "Unload the model?", message "The comma drives on its small model until one is loaded again." |
| Models, delete download | title "Delete the download?", message "Deletes the 766 MB model file for <name>. The prepared engine stays, so the comma can still use this model." (size omitted when unknown) |
| Models, delete engines | title "Delete the prepared engines?", message "Deletes every prepared engine for <name>, 10.3 GB in all. Preparing it again takes as long as the first time." plus, when loaded, " The model is unloaded first." |
| Models, prepare while the comma drives | title "Prepare <name>?", message "The comma is using <current>. Switching drops it to its small model until <name> is loaded and the comma reconnects." |
| Models, server stopped | "Server not running" with "Start the server to load the model list." |
| Logs, Clear help | "Clear the view. The log file keeps everything." |
| Logs, Copy all help | "Copy the shown lines" |
| Settings, cache caption | "Models and prepared engines. A CoreML engine is about 10 GB. Takes effect when the server restarts." |
| Settings, backend auto | "Recommended. About 43 ms a frame on an M1 Pro. Preparing takes about 18 minutes the first time and 9 minutes for each later load, so keep Jetlink running." |
| Settings, backend coreml | same as auto without "Recommended. " |
| Settings, backend ane | "Faster back to back, slower at the comma's 20 Hz on an M1 Pro. Measure before using it in the car." |
| Settings, backend tinygrad | "Loads in a second. About 66 ms a frame on an M1 Pro, over the 50 ms budget; a newer Mac may be under it." |
| Settings, override caption | "For development. Empty means the bundled runtime." |
| Inspector, no engines | "No prepared engine yet." |

## Toolbar activity view (decided 2026-09-09, replaces the plain badge)

The toolbar's centre is `ToolbarActivityView`
(`Views/Components/ActivityView.swift`), shaped like Xcode's activity view:

- 480 by 30 points at most (minimum 240, it gives way to the title and the
  run button because the toolbar centres it on the window), `.font(.callout)`,
  in the `.principal` placement. On macOS 26 the toolbar wraps the item in its
  own glass capsule, so it draws no background; on macOS 15 it is
  `.quaternary` in a 9 point rounded rectangle, the inset Xcode 16 used.
- Not a `Button`. Bisected on macOS 26: a `Button` in the principal placement
  is taken for a toolbar button and the whole item is dropped from the
  toolbar, whatever its label. The click is an `.onTapGesture`, with the
  button trait and action set for accessibility.
- Left, two crumbs with a small tertiary chevron between them: the model
  (`shippingbox`, the loaded or loading model's `displayName`, "No model"
  otherwise) and the backend (`cpu`, `StatusView.backendDescription` from the
  server info, or `BackendChoice.title` for the setting before the server has
  said). Long crumbs truncate in the middle.
- Right, `StatusBadge.summary` in `.primary`, red for `.bad`, orange for
  `.warning`. No dot. A `.mini` spinner while starting or stopping.
- Along the bottom edge, a 3 point capsule bar in the tint over `.quaternary`,
  `engine.frac` wide, while a model is being prepared or loaded. It is the
  only progress the toolbar shows.
- Clicking it selects Status. Its accessibility label is the three strings.

`StatusBadge` keeps only the capsule style; the `.plain` style is gone.

Beside this, the Status view's failure block no longer repeats itself: the
store's exit message is one sentence (`ServerStore.exitMessage`, "The server
was killed by signal 9 (SIGKILL). Console may have a crash report for
python3.14 under Crash Reports.") and the view shows the last 20 log lines
under it once.
