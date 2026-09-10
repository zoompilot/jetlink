# 20. Swift: the non-UI core

**Owner: agent C.** Files you create, all under `macos/Jetlink/`:
`App/AppState.swift`, `Server/EmbeddedPython.swift`, `Server/ServerProcess.swift`,
`Server/ControlProtocol.swift`, `Server/ControlClient.swift`,
`Server/ServerStore.swift`, `Server/SleepAssertion.swift`,
`Server/LogBuffer.swift`, `Server/LogFileWriter.swift`,
`Models/ModelStore.swift`, `Models/ModelRowBuilder.swift`,
`Settings/AppSettings.swift`, `Settings/LoginItem.swift`, and tests under
`macos/JetlinkTests/`: `ControlProtocolTests.swift`, `ControlClientTests.swift`,
`ModelRowBuilderTests.swift`, `ServerConfigurationTests.swift`,
`LogFileWriterTests.swift`, with fixtures in `macos/JetlinkTests/Fixtures/`.
Read first: `01-contracts.md` sections 4, 7, 8, 9, 10. You do not need the
Python code; the JSON fixtures are your oracle. The Xcode project comes from
`40-packaging-and-signing.md`; until it exists, develop in a scratch
`Package.swift` if you like, but the deliverable is files in the paths above.

## Concurrency recipe (follow it exactly; it keeps Swift 6 quiet)

- Stores are `@MainActor @Observable final class`. Views read them; only the
  store mutates its own `private(set)` properties.
- Blocking or callback-based APIs (`Process`, `Pipe.readabilityHandler`,
  `NWConnection` handlers, IOKit) deliver on arbitrary queues. Wrap them: the
  callback captures only `Sendable` values and does `Task { @MainActor in store.apply(...) }`,
  or feeds an `AsyncStream.Continuation` (which is `Sendable`).
- Long-running loops (`for await event in client.events`) run in a `Task`
  created on the main actor and stored so they can be cancelled.
- Types crossing actors are `struct`s or `enum`s marked `Sendable`. No class
  crosses an actor boundary except `ControlClient`, whose mutable state is
  protected by a private `NSLock` or an internal `actor`; mark it `final class
  ...: Sendable` only after making every stored property immutable or
  lock-protected, and comment why.
- Never `DispatchQueue.main.async` in new code; use the main actor.

## Files

### `Server/EmbeddedPython.swift`

```swift
struct PythonRuntime: Sendable {
  enum Source: Sendable, Equatable { case bundled, environment(String), settings(String) }
  let executable: URL; let source: Source
  static func locate(settings: AppSettings) -> Result<PythonRuntime, PythonRuntimeError>
}
enum PythonRuntimeError: Error, LocalizedError { case notBundled, overrideMissing(String) }
```
Order: `ProcessInfo.processInfo.environment["JETLINK_PYTHON"]` → `settings.pythonOverride`
→ `Bundle.main.resourceURL!/python/bin/python3`. An override that does not
exist or is not executable is `overrideMissing`. `notBundled` has an
`errorDescription` telling the developer to run `make python` or set
`JETLINK_PYTHON` (the exact text is in `30-swiftui-views.md` under "Errors").
Also expose `static func manifest() -> [String: String]?` reading
`python/MANIFEST.json` for the About/Settings display.

### `Server/ServerProcess.swift`

`Process` with `executableURL = configuration.python`, `arguments` per
`01-contracts.md` section 9, `environment` per section 9 (build a fresh
dictionary; copy only `HOME`, `TMPDIR`, `USER` from the current environment,
default `TMPDIR` to `NSTemporaryDirectory()`), `currentDirectoryURL = cacheDirectory`
(create it first with `FileManager.createDirectory(withIntermediateDirectories:)`).
`standardOutput` and `standardError` both go to one `Pipe` (stdout is empty in
practice; merging keeps ordering). The readability handler accumulates a
`Data` buffer, splits on `0x0A`, decodes UTF-8 lossily, and calls `onLine` per
line. On EOF (`availableData.isEmpty`) it flushes any partial line and removes
the handler. `terminationHandler` calls `onExit(status, reason)` once.

`stop()`: if not running return. Send `interrupt()` (SIGINT); wait up to 10 s
for termination (poll `isRunning` every 100 ms with `Task.sleep`; do not block
the main actor synchronously); then `terminate()` (SIGTERM) and wait 5 s; then
`kill(pid, SIGKILL)`. Log which level was needed.

Expose `static func arguments(for configuration: ServerConfiguration) -> [String]`
and `static func environment(for configuration: ServerConfiguration) -> [String: String]`
as pure functions so they can be tested.

Backend mapping (`BackendChoice` → arguments): `auto` → `--backend auto` and no
`--device`; `coreml` → `--backend ort --device coreml`; `ane` → `--backend ort --device ane`;
`tinygrad` → `--backend tinygrad --device METAL`.

### `Server/ControlProtocol.swift`

The types from `01-contracts.md` section 7. Decoding:

- `JSONDecoder` with `keyDecodingStrategy = .convertFromSnakeCase`. The
  `HelloEvent` maps `protocol` → `protocolVersion` with explicit `CodingKeys`.
- `ControlEvent.init(jsonLine:)`: first decode `struct Envelope: Decodable { let event: String }`,
  then switch on `event` to decode the payload type from the same data.
  Unknown names → `.unknown(name:)`. Malformed JSON throws.
- `ReplyEvent.extras`: everything in the object except `event`, `t`, `id`,
  `ok`, `error`. Implement a small `enum JSONValue: Codable, Sendable { case string, number, bool, null, array, object }`
  and decode the reply through a `[String: JSONValue]` then pick fields.
- `ControlCommand.jsonLine(id:)`: hand-built `[String: Any]` → `JSONSerialization`
  with `.sortedKeys`, then append `\n`. Command names and argument keys are
  exactly the contract's (`cancel_download`, `import`, `frame_skip`).

### `Server/ControlClient.swift`

Network.framework: `NWConnection(to: .unix(path: socketPath.path), using: .tcp)`.
(`NWParameters.tcp` is what Apple documents for Unix domain stream endpoints.)
`connect(retryingFor:)`: loop: create a connection, `start(queue:)` on a
private serial `DispatchQueue`, await `.ready` or `.failed`/`.waiting` via a
continuation with a 2 s per-attempt timeout; on failure sleep 250 ms and retry
until the deadline; then throw `ControlClientError.timedOut`. When ready, start
the receive loop: `receive(minimumIncompleteLength: 1, maximumLength: 1 << 16)`
recursively, appending to a `Data` buffer, splitting on newline, decoding each
line with `ControlEvent(jsonLine:)`. Replies go to the pending-command table
(`[Int: CheckedContinuation<ReplyEvent, Error>]` under a lock); other events go
to the `AsyncStream` continuation (buffering policy `.unbounded`; the consumer
is the main actor and keeps up). A decode failure of one line is logged and
skipped, never fatal. On connection failure or EOF: finish the stream, fail all
pending continuations with `ControlClientError.disconnected`.

`send(_:timeout:)`: assign `id` from an atomic counter (lock), register the
continuation, `connection.send(content:completion:)`; on send error fail the
continuation. Timeout via a racing `Task.sleep`; on timeout remove and throw
`ControlClientError.timedOut`.

If `NWEndpoint.unix` turns out not to connect (test it first with a Python
`socket.socket(AF_UNIX).bind` echo server in `ControlClientTests`), fall back
to POSIX: `socket(AF_UNIX, SOCK_STREAM, 0)`, `connect`, a `DispatchIO`
channel for reads and `write(2)` under the lock for sends. Keep the public API
identical either way.

### `Server/ServerStore.swift`

Owns a `ServerProcess`, a `ControlClient?`, the `LogBuffer`, the
`LogFileWriter`, a `SleepAssertion`, and the settings. State machine:

- `start()`: guard `runState == .stopped || .failed`. `runState = .starting`,
  `lastFailure = nil`. Locate Python (failure → `.failed(description)`).
  Build `ServerConfiguration` from settings; the control socket path from
  `01-contracts.md` section 2 (delete a stale file first). Set
  `process.onLine` to append to the log buffer and the log file (hop to main).
  Set `process.onExit` to `handleExit`. `process.start()`; on throw → `.failed`.
  Then `Task { await connectControl() }`: `client.connect(retryingFor: .seconds(120))`
  (CoreML backends can take a while to probe; the process exiting first will
  fire `handleExit`, which cancels this task). On success start the consume
  task: `for await event in client.events { apply(event) }`.
- `apply(_:)`: `hello` → `info` (partial), `runState = .serving`,
  `startedAt = Date()`, `restartCount` reset after 10 minutes of serving;
  `server` → fill backend fields of `info`, and if `state == "stopping"`
  mark `stopRequested = true`; `link`, `engine`, `stats` → assign (`stats` is
  set to nil when `link` leaves `connected`); everything else is forwarded to
  `modelEvents: AsyncStream<ControlEvent>`'s continuation so `ModelStore` can
  consume it (create the stream in `init`; `AppState` connects the two).
- `stop()`: `runState = .stopping`, `stopRequested = true`;
  `try? await client?.send(.shutdown, timeout: .seconds(3))`; `await process.stop()`;
  `client?.close()`; `runState = .stopped`; `link = waiting`, `engine = none`, `stats = nil`.
- `handleExit(status, reason)`: if `stopRequested` → `.stopped`. Else this is
  a crash: `lastFailure = "Server exited with status \(status)"` plus the last
  20 log lines joined; if `runState == .starting` → `.failed(lastFailure)` (a
  startup failure is not retried automatically: it is probably configuration);
  if `.serving` → schedule `start()` after the backoff from
  `01-contracts.md` section 10, and set `runState = .failed(...)` until it
  restarts. Give up (stay `.failed`) after five restarts within ten minutes.
- `startIfNeeded()`: if `.serving` return; if `.stopped`/`.failed` call
  `start()`; then await until `.serving` or `.failed` (poll with
  `withObservationTracking` or a simple `Task.sleep(100 ms)` loop, 180 s cap);
  throw `ServerStoreError.failed(String)` on failure.
- `restart()`: `stop()` then `start()`.
- Sleep: after every `runState` or power-source change,
  `sleepAssertion.setActive(runState == .serving && settings.keepAwakeWhileServing && onACPower)`.

### `Server/SleepAssertion.swift`

`IOPMAssertionCreateWithName(kIOPMAssertionTypePreventUserIdleSystemSleep as CFString, IOPMAssertionLevel(kIOPMAssertionLevelOn), "Jetlink is serving the comma" as CFString, &id)`
and `IOPMAssertionRelease`. `var isOnACPower: Bool` from
`IOPSCopyExternalPowerAdapterDetails() != nil` (nil on battery). Observe power
changes with `IOPSNotificationCreateRunLoopSource` added to the main run loop;
the callback posts to a closure `onPowerSourceChange` on the main actor.

### `Server/LogBuffer.swift` and `Server/LogFileWriter.swift`

`LogBuffer`: `lines` capped at 5000 (drop from the front in chunks of 500 to
avoid O(n) per append), `revision += 1` per append so views can observe one
integer. `LogFileWriter`: not main-actor bound; an `actor` owning a
`FileHandle` to `~/Library/Logs/Jetlink/server.log`; `append(line)` writes
`line + "\n"`; when the file exceeds 20 MB, close, rotate `.1` → `.2`, `.log`
→ `.1`, reopen. Test the rotation with a 1 KB threshold injected through the
initializer.

### `Models/ModelStore.swift` and `Models/ModelRowBuilder.swift`

`ModelStore` consumes `serverStore.modelEvents`: `inventory`, `catalog`,
`download` (keyed by sha256; remove on `done`/`failed`/`cancelled` after 3 s
so the row shows the terminal state briefly), `import`. After each, and after
every change of `serverStore.engine`, `rows = ModelRowBuilder.build(...)`
(observe `serverStore.engine` with `withObservationTracking` in a loop, or have
`AppState` call `modelStore.engineChanged()` from its own observation).

`ModelRowBuilder.build(catalog:inventory:downloads:engine:) -> [ModelRow]` is a
pure function:
1. One row per catalog model, in catalog order. `sha256`/`bytes` from the
   catalog entry when resolved, else `status = .unresolved`.
2. For rows with a sha: `downloads[sha]` → `.downloading(frac, rate)` (state
   `started`/`progress`), `.failed(detail)` (state `failed`); `engine.sha256 == sha`
   and engine state `building`/`loading` → `.preparing(stage, frac, msg)`,
   `ready` → `.loaded` with `isLoaded = true`, `failed` → `.failed(detail)`;
   else an artifact with `current: true` in inventory → `.prepared`; else a
   model file in inventory → `.downloaded`; else `.notDownloaded`.
3. `preparedFor` = all inventory artifacts with that sha (any backend).
4. `isDefault = ref == catalog.defaultRef`. `isRequestedByComma`: true when
   `engine.sha256 == sha` and the server's link is connected (pass `link` in too).
5. Then rows for inventory models and artifacts whose sha matches no catalog
   entry: name from the inventory `name` (a local import) → `isLocal = true`;
   with no name → `"Unknown model \(sha.prefix(16))"`, `isOrphan = true`. A
   16-character `sha256` from the inventory (identity unknown) is also an orphan.
6. Sort: catalog rows first in catalog order, then local rows by name, then
   orphans.

Actions call `serverStore.startIfNeeded()` first, then `send(...)`; errors
become `lastError: String?` on the store for the view to show. `prepare` while
`serverStore.link.state == .connected` and `engine.sha256 != row.sha256` must
not send until the view has confirmed (the view calls
`prepare(row, confirmedInterruption: true)`); expose
`func prepareNeedsConfirmation(_ row: ModelRow) -> Bool`.

### `Settings/AppSettings.swift`, `Settings/LoginItem.swift`

`AppSettings`: `@Observable` with stored properties whose `didSet` writes
`UserDefaults.standard` under the keys in `01-contracts.md` section 8;
`init()` reads them with the defaults there. `cacheDirectory` default computed
from `FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)`.
`LoginItem`: `SMAppService.mainApp`; `var isEnabled: Bool { status == .enabled }`,
`func setEnabled(_:) throws` calling `register()`/`unregister()`; surface
`requiresApproval` so the view can say to open System Settings.

### `App/AppState.swift`

The composition root: creates `AppSettings`, `LogBuffer`, `ServerStore`,
`ModelStore`; wires `modelStore` to `serverStore.modelEvents`; on launch, if
`settings.startServerOnLaunch`, calls `serverStore.start()` and then
`modelStore.refreshCatalog()` once `.serving` (catalog refresh needs the server).
Exposes `func applicationWillTerminate()` that awaits `serverStore.stop()`
(the app delegate in `30-swiftui-views.md` calls it and uses
`NSApplication.reply(toApplicationShouldTerminate:)` with `.terminateLater`).

## Tests

- `ControlProtocolTests`: decode each fixture line in
  `Fixtures/control_events.jsonl` (write it from the examples in
  `01-contracts.md` 4.2, one per event type, plus an unknown event and a
  malformed line); check field values; encode every `ControlCommand` and
  compare to expected JSON strings (sorted keys).
- `ControlClientTests`: spawn `/usr/bin/python3`? No: the test machine may
  lack it. Write a tiny in-test echo server with POSIX sockets (`socket`,
  `bind`, `listen`, `accept` on a background thread) that sends two fixture
  lines on connect and replies `{"event":"reply","id":N,"ok":true}` to any
  line; assert the events stream yields both events and `send(.status)`
  returns `ok`. Also assert `connect(retryingFor: .milliseconds(600))` throws
  when nothing listens.
- `ModelRowBuilderTests`: catalog fixture + hand-built inventory/downloads/engine
  covering every `ModelStatus`, the default flag, a local model, an orphan, and
  the sort order.
- `ServerConfigurationTests`: arguments and environment for each backend and
  transport; no `PYTHONPATH`/`DYLD_` keys ever present.
- `LogFileWriterTests`: rotation at a 1 KB threshold produces `.1` and `.2`.

## Report back

List the public API exactly as implemented where it differs from
`01-contracts.md`, and whether `NWEndpoint.unix` worked or the POSIX fallback
was needed.
