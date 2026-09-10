# 01. Contracts

Everything two workstreams must agree on. Treat this file as an interface
definition: implement exactly what is here, and if it has to change, change it
here first and tell the other side.

## 1. Pinned versions and toolchain

| Thing | Value | Why |
| --- | --- | --- |
| macOS deployment target | 15.0 | `@Observable`, `MenuBarExtra`, `SMAppService`, Swift Testing |
| Xcode | 26.6 (local and the `macos-26` GitHub runner default) | matches the machine this was planned on |
| Swift | 6 language mode | strict concurrency; patterns in `20-swift-core.md` |
| Python | 3.14.7, python-build-standalone release `20260901`, asset `cpython-3.14.7+20260901-aarch64-apple-darwin-install_only_stripped.tar.gz`, SHA-256 `4632cb1a6edad9e73d3c81b6d2e69131637d995173e3e85005df14102b0592ba` | same major as the bench venv |
| onnxruntime | 1.29.0 | measured on the bench |
| numpy | 2.5.3 | |
| onnx | 1.22.0 | |
| libusb1 (PyPI) | 3.4.0 | |
| protobuf | 7.36.1 | onnx and onnxruntime dependency |
| ml_dtypes | 0.6.0 | onnx dependency |
| typing_extensions | 4.16.0 | onnx dependency |
| flatbuffers | 25.12.19 | onnxruntime dependency |
| packaging | 26.3 | onnxruntime dependency |
| tinygrad | git `https://github.com/sunnypilot/tinygrad` at commit `e837e367aac9e1a66e689f4f32ce20ca9367df13` (the sunnypilot fork; the commit is not on any upstream branch, so pip cannot fetch it from tinygrad/tinygrad) | PyPI 0.14.0 lacks the `org.tinygrad` ONNX domain |
| libusb (native) | Homebrew `libusb` 1.0.30, file `libusb-1.0.0.dylib` | copied into the bundle |
| xcodegen | latest Homebrew | generates the Xcode project |

## 2. Paths

| What | Where |
| --- | --- |
| Embedded Python prefix | `Jetlink.app/Contents/Resources/python/` with `bin/python3`, `lib/python3.14/site-packages/` |
| Embedded runtime manifest | `Jetlink.app/Contents/Resources/python/MANIFEST.json` (versions, tarball sha, build time, git sha of the repo) |
| App support directory | `~/Library/Application Support/Jetlink/` |
| Default cache root (`--cache`) | `~/Library/Application Support/Jetlink/cache/` |
| Cache layout (existing) | `<cache>/engines/`, `<cache>/models/`, `<cache>/last-loaded.json` |
| Registry state (new, Python-owned) | `<cache>/registry/catalog.json`, `<cache>/registry/pointers.json`, `<cache>/registry/local-models.json` |
| In-flight download | `<cache>/models/<sha16>.onnx.part` (renamed to `<sha16>.onnx` only after size and hash verify) |
| Control socket | `NSTemporaryDirectory()/jetlink-control.sock` (the per-user `$TMPDIR`; AF_UNIX paths are limited to 104 bytes on macOS) |
| Server log file | `~/Library/Logs/Jetlink/server.log`, rotated at 20 MB, keep `server.log.1` and `.2` |
| App logs | `os.Logger`, subsystem `io.zoompilot.jetlink` |
| Bundle identifier | `io.zoompilot.jetlink` (change in one place: `macos/project.yml`) |

## 3. Python CLI surface (every platform)

### 3.1 `jetlink-server` gains

```
--control-socket ADDR   open a local control channel; ADDR is a filesystem path
                        (AF_UNIX) or tcp://127.0.0.1:PORT (loopback only, for Windows)
--parent-pid PID        exit cleanly when this process is no longer our parent
                        (checked once a second; a dead parent means getppid() changed)
```

SIGTERM is handled like SIGINT (clean shutdown, engine released, exit code 0).

### 3.2 `jetlink-models` (new; `python -m jetlink.registry` is the same)

```
jetlink-models list      [--refresh] [--json] [--cache DIR]
jetlink-models resolve   REF [--json] [--cache DIR]
jetlink-models fetch     REF_OR_SHA256 [--cache DIR]
jetlink-models import    PATH [--name NAME] [--cache DIR]
jetlink-models inventory [--json] [--cache DIR]
jetlink-models rm        SHA256 [--artifacts] [--model] [--cache DIR]
jetlink-models prepare   REF_OR_SHA256 [--backend auto] [--device auto] [--cache DIR]
```

- `--cache` defaults to `jetlink.server.platform.default_cache_dir()`, so on a
  Jetson it is `/mnt/data/jetlink` and `JETLINK_CACHE` wins everywhere.
- `REF_OR_SHA256`: a 40-hex string is a catalog ref, a 64-hex string is an ONNX
  SHA-256 (the LFS oid). Anything else is an error.
- `--json` prints exactly the `catalog` / `inventory` event payload from
  section 4, so a script and the app read the same shape.
- `prepare` fetches if needed, then builds with the chosen backend the same way
  `jetlink-server --build` does, and records `last-loaded.json` when none is
  recorded. It must warn (stderr) if a `jetlink-server` is likely running
  against the same cache: two processes building into one cache is unsupported.
- Progress on stderr, one line per whole percent, never on stdout.
- Exit codes: 0 ok, 1 usage or not found, 2 network failure, 3 verification
  failure (size or hash).

## 4. Control protocol, version 1

Transport: a stream socket. Encoding: UTF-8 JSON, one object per line, `\n`
terminated, no pretty printing. Numbers are JSON numbers. Absent optional
fields are `null`, never missing, so decoders can be strict.

Client to server: `{"id": <int>, "cmd": "<name>", ...arguments}`. `id` is
chosen by the client, positive, increasing.

Server to client: `{"event": "<name>", "t": <unix time, float seconds>, ...}`.
Exactly one `reply` event per command, carrying the command's `id`. Any other
event may arrive at any time, including between a command and its reply.

The server may have several clients; every event goes to all of them. A client
that does not read fast enough (more than 1000 queued lines) is disconnected.

### 4.1 On connect

The server sends, in this order: `hello`, `server`, `link`, `engine`,
`inventory`, `catalog`, then one `download` event per download in progress.
A client needs no command to render its first screen.

### 4.2 Events

```jsonc
{"event":"hello","t":0,"protocol":1,"pid":4242,"version":"0.2.0","python":"3.14.7",
 "platform":"darwin","cache":"/Users/me/Library/Application Support/Jetlink/cache",
 "transport":"usb","port":null}
// transport is "usb"|"tcp"|"ffs"; port is set for tcp.

{"event":"server","t":0,"state":"serving","detail":"","backend":"ort",
 "runtime_version":"1.29.0","device":"coreml-Apple_M1_Pro"}
// state: "serving" | "stopping". backend/runtime_version/device are what the
// hello to the comma carries (Backend.describe()).

{"event":"link","t":0,"state":"waiting","detail":"waiting for a jetlink gadget at 1209:0001","peer":null}
// state: "waiting" | "connected" | "disconnected". peer: "usb" or "host:port" when connected.
// Emitted on transitions only, never on every 2 s poll.

{"event":"engine","t":0,"state":"none","sha256":null,"detail":"","stage":null,"frac":0.0,"msg":"","load_only":false}
// state: "none" | "building" | "loading" | "ready" | "failed"
//   building: a Job with load_only false; loading: a Job with load_only true
//   ready: EngineHost.loaded is set; sha256 is the loaded model
//   failed: the last Job failed; detail says why
// stage: "upload"|"patch"|"parse"|"build"|"save"|"load"|"failed"|null, frac 0..1, msg free text,
// straight from EngineHost._progress. Emitted on every state change and on
// progress at most 4 times a second.

{"event":"stats","t":0,"frames":1234,"fps":19.9,"total_ms":{"mean":31.2,"p99":38.0,"max":41.5},
 "gpu_ms":{"mean":21.0},"slow":0,"window_s":1.0}
// Once a second while a link is connected and at least one frame was served in
// the window. frames is the session's running count. slow counts frames over
// 60 ms (SLOW_FRAME_US) in the window.

{"event":"inventory","t":0,"loaded":"<sha256>|null","last_loaded":"<sha256>|null",
 "models":[{"sha256":"…","bytes":765953504,"path":"…/models/a086d5249fc308bb.onnx","name":"BMRLNAP Model v4 (August 30, 2026)","ref":"f877d7a0…|null"}],
 "artifacts":[{"sha256":"…","key":"a086d5249fc308bb.ort1.29.0.coreml-Apple_M1_Pro","path":"…/engines/a086….ortcache",
               "bytes":5900000000,"backend":"ort","runtime_version":"1.29.0","device":"coreml-Apple_M1_Pro",
               "built_at":"2026-09-08T21:19:15Z","build_seconds":548.9,"checkpoint":"b9facbcc-…","current":true}],
 "disk":{"models_bytes":765953504,"engines_bytes":5900000000,"free_bytes":120000000000}}
// models: every complete <sha16>.onnx in models/ (a .part is not listed).
// artifacts: every engines/*.json sidecar with a spec.sha256, any backend.
//   current is true when the key equals this server's cache key for that sha.
//   runtime_version comes from the sidecar's onnxruntime | tinygrad | trt_version.
// name: from the catalog (via the pointer cache) or local-models.json, else null.
// Emitted on connect, after every build or load completes, after forget, after
// a download or import completes, and on the inventory command.

{"event":"catalog","t":0,"fetched_at":1757440000.0,"url":"https://…/driving_models_chestnut_v25.json",
 "default_ref":"f877d7a0ccc3cce943c76e285214c020cd65c899","error":null,
 "models":[{"name":"Cinque Terre Model V2 (September 08, 2026)","short_name":"CTMV2",
            "ref":"37bfa1413edcdc2e8844984b83727c33f81d8f46","build_time":"2026-09-08T…Z","index":12,
            "sha256":"…|null","bytes":765950064}]}
// Newest first (index descending). sha256/bytes are null until the pointer for
// that ref has been resolved. error is set when a refresh failed and the list
// is the previous cached one (possibly empty).

{"event":"download","t":0,"sha256":"…","ref":"…|null","state":"progress","frac":0.42,
 "bytes":321000000,"total":765953504,"rate_bps":41000000.0,"detail":"","source":"https://gitlab.com/…/info/lfs"}
// state: "started" | "progress" (at most 4 a second) | "done" | "failed" | "cancelled".

{"event":"import","t":0,"path":"/Users/me/Downloads/big.onnx","state":"hashing","frac":0.3,"sha256":null,"detail":""}
// state: "hashing" | "copying" | "done" | "failed". sha256 set from "copying" on.

{"event":"reply","t":0,"id":7,"ok":true}
{"event":"reply","t":0,"id":8,"ok":false,"error":"model a086d5249fc308bb is not downloaded"}
```

### 4.3 Commands

| cmd | arguments | reply extras | behaviour |
| --- | --- | --- | --- |
| `status` | | | re-sends `server`, `link`, `engine`, `inventory`, `catalog` |
| `catalog` | `refresh: bool` (default false) | `queued: true` | fetch the catalog when refresh is true, the cache is older than 3600 s, or missing; then resolve pointers for refs without one (parallel, 8 at a time); emit `catalog` when done (also when it fails, with `error`). Never blocks the reply. |
| `download` | `ref` or `sha256` (one of them) | `sha256` | resolve the pointer if needed; enqueue a download (one runs at a time, FIFO); `download` events follow. Error if already downloaded, already queued, or the ref is unknown. |
| `cancel_download` | `sha256` | | cancels a running or queued download; `.part` removed; a `download` event with `cancelled` |
| `import` | `path` | `queued: true` | hash the file (streaming), copy it to `models/<sha16>.onnx` via a `.part`, record `{sha256, bytes, name, added_at}` in `local-models.json`; `import` events, then `inventory` |
| `prepare` | `sha256`, `frame_skip` (default 4) | `state` | the ENGINE_REQ path without a comma: `EngineHost.request(Request(sha256, nbytes, frame_skip), session=None)`. `nbytes` is the model file's size if present, else the sidecar spec's `nbytes`, else the pointer's size, else 0. Error when neither an artifact nor the model file exists. `state` is the engine state afterwards. |
| `unload` | | | release the loaded engine; `engine` event with `none` |
| `forget` | `sha256`, `artifacts: bool`, `model: bool` | | unload first if that model is loaded; delete every `engines/<sha16>.*` when artifacts, `models/<sha16>.onnx` (and `.part`) when model; remove `last-loaded.json` if it names this sha and its artifact is gone; then `inventory` |
| `inventory` | | | emit `inventory` |
| `shutdown` | | | reply, emit `server` with `stopping`, then exit cleanly as SIGINT would |

Errors are plain English sentences in `error`. Unknown `cmd` replies
`ok:false`. A malformed line (not JSON, or no `id`) gets
`{"event":"reply","id":null,"ok":false,"error":"…"}`.

## 5. Registry Python API (agent A implements, agent B calls)

Module `jetlink.registry`. Stdlib only. Every network function takes an
optional `opener` (a callable like `urllib.request.urlopen`) so tests inject
fixtures.

```python
CATALOG_URL = "https://raw.githubusercontent.com/sunnypilot/sunnypilot-models/refs/heads/gh-pages/docs/driving_models_chestnut_v25.json"
REQUIRED_SELECTOR_VERSION = 19
DEFAULT_BIG_MODEL_REF = "f877d7a0ccc3cce943c76e285214c020cd65c899"
POINTER_URL = "https://raw.githubusercontent.com/commaai/openpilot/{ref}/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx"
LFS_ENDPOINTS = ("https://gitlab.com/commaai/openpilot-lfs.git/info/lfs",
                 "https://huggingface.co/commaai/openpilot-lfs.git/info/lfs")

@dataclass(frozen=True)
class CatalogModel:
  name: str; short_name: str; ref: str; build_time: str; index: int

@dataclass(frozen=True)
class Pointer:
  oid: str; size: int          # oid is the ONNX SHA-256

@dataclass(frozen=True)
class LocalModel:
  sha256: str; bytes: int; name: str; added_at: float

class RegistryError(Exception): ...
class NetworkError(RegistryError): ...          # exit code 2 in the CLI
class VerifyError(RegistryError): ...           # exit code 3

ProgressFn = Callable[[float], None]            # 0..1
StopFn = Callable[[], bool]

def parse_catalog(data: dict) -> list[CatalogModel]      # filter + sort, no IO
def parse_pointer_text(text: str) -> Pointer | None
def fetch_catalog(url: str = CATALOG_URL, timeout: float = 10.0, opener=None) -> dict
def fetch_pointer(ref: str, timeout: float = 10.0, opener=None) -> Pointer
def lfs_resolve(endpoint: str, pointer: Pointer, timeout: float = 30.0, opener=None) -> str | None
def lfs_download(href: str, pointer: Pointer, dest: Path, progress: ProgressFn | None = None,
                 should_stop: StopFn | None = None, opener=None) -> Path
def is_ref(s: str) -> bool; def is_sha256(s: str) -> bool

class Registry:
  def __init__(self, cache_root: Path): ...          # creates <root>/registry/ and <root>/models/
  # catalog
  def catalog(self, refresh: bool = False, max_age: float = 3600.0, opener=None) -> dict
      # the `catalog` event payload (section 4.2), from cache when fresh enough
  def resolve(self, ref: str, opener=None) -> Pointer          # pointers.json, fetched once, kept forever
  def resolve_missing(self, refs: list[str], workers: int = 8, opener=None) -> dict[str, Pointer | Exception]
  def name_for(self, sha256: str) -> tuple[str | None, str | None]   # (name, ref) via pointers or local models
  # models on disk
  def model_path(self, sha256: str) -> Path                    # <root>/models/<sha16>.onnx (same rule as EngineCache.model_path)
  def fetch(self, ref_or_sha256: str, progress: ProgressFn | None = None, should_stop: StopFn | None = None,
            opener=None) -> Path                                 # resolves, tries LFS_ENDPOINTS in order, downloads to .part, verifies, renames
  def import_model(self, path: Path, name: str | None = None, progress: ProgressFn | None = None,
                   should_stop: StopFn | None = None) -> LocalModel
      # progress covers the whole import: 0.0..0.5 is the hashing pass, 0.5..1.0 the copy.
      # The control server derives the `import` event's state from it: frac < 0.5 is "hashing",
      # frac >= 0.5 is "copying"; the sha256 is known only after the hashing pass (the
      # control server may hash the file itself first, or read it from the returned LocalModel).
  def local_models(self) -> list[LocalModel]
  # inventory and removal
  def inventory(self, cache: "EngineCache | None" = None) -> dict   # the `inventory` event payload
      # `current` is computed only when a cache with a backend is passed
  def remove(self, sha256: str, artifacts: bool, model: bool) -> None
```

Implemented notes (agent A, merged): `RegistryError`, `NetworkError`, `VerifyError`, `is_ref`
and `is_sha256` are defined in `jetlink/registry/catalog.py` and re-exported from
`jetlink.registry`. `fetch()` by a bare sha256 works only when a pointer with that oid is
already cached (the size is needed); the control server resolves the catalog first.
`inventory()['loaded']` is always `None` from the registry; the control server fills it.
A model file whose full identity is unknown is listed with its 16 character prefix as `sha256`.

`Registry` never imports a backend and never imports `jetlink.server.session`.
It may import `jetlink.server.cache` for `EngineCache`, `CacheEntry` and the
sha validation regex, and `jetlink.spec.sha256_file`.

## 6. EngineHost additions (agent B implements; recorded here because control.py and tests depend on them)

```python
class EngineHost:
  def subscribe(self, fn: Callable[[str, dict], None]) -> None
  def emit(self, kind: str, payload: dict) -> None      # never raises; a failing listener is logged and dropped
  def snapshot(self) -> dict                            # the `engine` event payload (without "event"/"t")
  def unload(self) -> None                              # public; emits engine
  frame_stats: FrameStats                               # .record(total_us, gpu_us) from Session._infer after the send

# kinds emitted: 'progress' {stage, frac, msg}; 'engine' snapshot(); 'link' {state, detail, peer} (from main.py)
```

## 7. Swift types (agent C implements, agent D consumes)

Module-internal, all in the `Jetlink` target. Signatures are the contract;
bodies are C's.

```swift
// ControlProtocol.swift
enum EngineState: String, Codable, Sendable { case none, building, loading, ready, failed }
enum LinkState: String, Codable, Sendable { case waiting, connected, disconnected }

struct HelloEvent: Codable, Sendable { let protocolVersion: Int; let pid: Int32; let version: String; let python: String; let platform: String; let cache: String; let transport: String; let port: Int? }
struct ServerEvent: Codable, Sendable { let state: String; let detail: String; let backend: String?; let runtimeVersion: String?; let device: String? }
struct LinkEvent: Codable, Sendable { let state: LinkState; let detail: String; let peer: String? }
struct EngineEvent: Codable, Sendable { let state: EngineState; let sha256: String?; let detail: String; let stage: String?; let frac: Double; let msg: String; let loadOnly: Bool }
struct StatsEvent: Codable, Sendable { struct Total: Codable, Sendable { let mean, p99, max: Double }; struct Gpu: Codable, Sendable { let mean: Double }
                                       let frames: Int; let fps: Double; let totalMs: Total; let gpuMs: Gpu; let slow: Int; let windowS: Double }
struct InventoryModel: Codable, Sendable, Identifiable { var id: String { sha256 }; let sha256: String; let bytes: Int64; let path: String; let name: String?; let ref: String? }
struct InventoryArtifact: Codable, Sendable, Identifiable { var id: String { key }; let sha256: String; let key: String; let path: String; let bytes: Int64; let backend: String; let runtimeVersion: String?; let device: String; let builtAt: String?; let buildSeconds: Double?; let checkpoint: String?; let current: Bool }
struct InventoryDisk: Codable, Sendable { let modelsBytes, enginesBytes, freeBytes: Int64 }
struct InventoryEvent: Codable, Sendable { let loaded: String?; let lastLoaded: String?; let models: [InventoryModel]; let artifacts: [InventoryArtifact]; let disk: InventoryDisk }
struct CatalogModel: Codable, Sendable, Identifiable { var id: String { ref }; let name: String; let shortName: String; let ref: String; let buildTime: String; let index: Int; let sha256: String?; let bytes: Int64? }
struct CatalogEvent: Codable, Sendable { let fetchedAt: Double?; let url: String; let defaultRef: String; let error: String?; let models: [CatalogModel] }
struct DownloadEvent: Codable, Sendable { let sha256: String; let ref: String?; let state: String; let frac: Double; let bytes: Int64; let total: Int64; let rateBps: Double; let detail: String; let source: String? }
struct ImportEvent: Codable, Sendable { let path: String; let state: String; let frac: Double; let sha256: String?; let detail: String }
struct ReplyEvent: Codable, Sendable { let id: Int?; let ok: Bool; let error: String?; let extras: [String: JSONValue] }

enum ControlEvent: Sendable {
  case hello(HelloEvent), server(ServerEvent), link(LinkEvent), engine(EngineEvent), stats(StatsEvent)
  case inventory(InventoryEvent), catalog(CatalogEvent), download(DownloadEvent), importEvent(ImportEvent)
  case reply(ReplyEvent), unknown(name: String)
  init(jsonLine: Data) throws          // switches on "event"; decodes with .convertFromSnakeCase; "protocol" maps to protocolVersion
}

enum ControlCommand: Sendable {
  case status, catalog(refresh: Bool), download(ref: String?, sha256: String?), cancelDownload(sha256: String)
  case importModel(path: String), prepare(sha256: String, frameSkip: Int), unload
  case forget(sha256: String, artifacts: Bool, model: Bool), inventory, shutdown
  func jsonLine(id: Int) -> Data
}

// ControlClient.swift
final class ControlClient: Sendable {
  init(socketPath: URL)
  func connect(retryingFor: Duration) async throws           // retries every 250 ms until connected or the deadline
  var events: AsyncStream<ControlEvent> { get }              // all non-reply events
  func send(_ command: ControlCommand, timeout: Duration = .seconds(30)) async throws -> ReplyEvent
  func close()
}

// ServerProcess.swift
enum BackendChoice: String, CaseIterable, Codable { case auto, coreml, ane, tinygrad }   // CLI mapping in 20-swift-core.md
enum TransportChoice: String, CaseIterable, Codable { case usb, tcp }
struct ServerConfiguration: Sendable { let python: URL; let backend: BackendChoice; let transport: TransportChoice; let tcpPort: Int; let cacheDirectory: URL; let controlSocket: URL; let logLevel: String; let logFile: URL }
@MainActor final class ServerProcess {
  var isRunning: Bool { get }; var processIdentifier: Int32? { get }
  var onLine: (@Sendable (String) -> Void)?            // one stderr line, already split, without the newline
  var onExit: (@Sendable (Int32, Process.TerminationReason) -> Void)?
  func start(_ configuration: ServerConfiguration) throws
  func stop() async                                    // SIGINT, then SIGTERM at 10 s, SIGKILL at 15 s
}

// ServerStore.swift
enum ServerRunState: Equatable, Sendable { case stopped, starting, serving, stopping, failed(String) }
struct ServerInfo: Equatable, Sendable { let pid: Int32; let version: String; let python: String; let backend: String; let runtimeVersion: String; let device: String; let cache: String; let transport: String; let port: Int? }
@MainActor @Observable final class ServerStore {
  private(set) var runState: ServerRunState
  private(set) var info: ServerInfo?
  private(set) var link: LinkEvent
  private(set) var engine: EngineEvent
  private(set) var stats: StatsEvent?
  private(set) var startedAt: Date?
  var lastFailure: String?
  func start(); func stop(); func restart()
  func send(_ command: ControlCommand) async throws -> ReplyEvent     // throws ServerStoreError.notRunning when there is no connection
  func startIfNeeded() async throws                                    // start and wait for .serving (used by ModelStore actions)
}

// ModelStore.swift
enum ModelStatus: Equatable, Sendable { case unresolved, notDownloaded, downloading(frac: Double, rateBps: Double), downloaded,
                                        preparing(stage: String, frac: Double, msg: String), prepared, loaded, failed(String) }
struct ModelRow: Identifiable, Equatable, Sendable {
  var id: String                 // sha256 when known, else "ref:<ref>"
  var name: String; var ref: String?; var sha256: String?; var bytes: Int64?; var buildTime: String?
  var status: ModelStatus; var preparedFor: [InventoryArtifact]; var isLoaded: Bool; var isDefault: Bool
  var isRequestedByComma: Bool; var isLocal: Bool; var isOrphan: Bool     // orphan: on disk, in no catalog and no local record
}
@MainActor @Observable final class ModelStore {
  private(set) var catalog: CatalogEvent?
  private(set) var inventory: InventoryEvent?
  private(set) var downloads: [String: DownloadEvent]
  private(set) var imports: [ImportEvent]
  private(set) var rows: [ModelRow]                  // rebuilt from the four above plus ServerStore.engine
  func refreshCatalog(); func download(_ row: ModelRow); func cancelDownload(_ row: ModelRow)
  func prepare(_ row: ModelRow); func unload(); func forget(_ row: ModelRow, artifacts: Bool, model: Bool)
  func importModel(at url: URL)
}

// LogBuffer.swift
@MainActor @Observable final class LogBuffer { private(set) var lines: [String]; private(set) var revision: Int; func append(_ line: String); func clear() }

// AppSettings.swift
@MainActor @Observable final class AppSettings {   // backed by UserDefaults, keys in section 8
  var backend: BackendChoice; var transport: TransportChoice; var tcpPort: Int; var cacheDirectory: URL
  var startServerOnLaunch: Bool; var keepAwakeWhileServing: Bool; var logLevel: String; var pythonOverride: String?
}

// Preview and test factories (C implements, D uses in #Preview and FormattingTests).
// Internal, never called from production paths. They build a store with the given
// state and no live process or socket behind it; actions on such a store are no-ops.
extension ServerStore {
  static func preview(runState: ServerRunState = .serving, info: ServerInfo? = nil, link: LinkEvent, engine: EngineEvent, stats: StatsEvent? = nil) -> ServerStore
}
extension ModelStore {
  static func preview(catalog: CatalogEvent?, inventory: InventoryEvent?, downloads: [String: DownloadEvent] = [:], engine: EngineEvent) -> ModelStore
}
extension LogBuffer { static func preview(lines: [String]) -> LogBuffer }
extension AppSettings { static func preview() -> AppSettings }   // backed by UserDefaults(suiteName: "io.zoompilot.jetlink.preview"), reset on each call
```

The app wires the stores together in `AppState` (C owns it): `@MainActor @Observable final class AppState { let settings: AppSettings; let server: ServerStore; let models: ModelStore; let logs: LogBuffer }`.
D's views take the stores they need as `@Environment(ServerStore.self)`-style
environment objects; `JetlinkApp` (D) creates one `AppState` and injects
`appState.settings`, `.server`, `.models`, `.logs` with `.environment(...)`.
```swift
```

## 8. UserDefaults keys

| Key | Type | Default |
| --- | --- | --- |
| `backend` | String (`auto`,`coreml`,`ane`,`tinygrad`) | `auto` |
| `transport` | String (`usb`,`tcp`) | `usb` |
| `tcpPort` | Int | 5599 |
| `cacheDirectory` | String (absolute path) | `~/Library/Application Support/Jetlink/cache` |
| `startServerOnLaunch` | Bool | true |
| `keepAwakeWhileServing` | Bool | true |
| `logLevel` | String (`INFO`,`DEBUG`) | `INFO` |
| `pythonOverride` | String (path to an interpreter) or absent | absent |

## 9. Server process launch (Swift builds this; Python receives it)

Executable: the embedded `bin/python3` (or `JETLINK_PYTHON` / `pythonOverride`).

Arguments:
```
-m jetlink.server.main
--backend <auto|ort|ort|tinygrad>           auto→auto, coreml→ort, ane→ort, tinygrad→tinygrad
--device <omitted|coreml|ane|METAL>          omitted for auto
--transport <usb|tcp> [--host 0.0.0.0 --port N when tcp]
--cache <cacheDirectory>
--control-socket <controlSocket path>
--parent-pid <the app's pid>
--log-level <INFO|DEBUG>
```

Environment (built from scratch, never inherited wholesale):
```
PATH=/usr/bin:/bin:/usr/sbin:/sbin
HOME, TMPDIR, USER, LANG=en_US.UTF-8
PYTHONNOUSERSITE=1  PYTHONDONTWRITEBYTECODE=1  PYTHONUNBUFFERED=1  PYTHONIOENCODING=utf-8
```
Do not set `PYTHONPATH`, `PYTHONHOME`, `DYLD_*`, `JETLINK_CACHE` or any venv
variable. The embedded prefix finds its own stdlib. With an override
interpreter, the `jetlink` package must already be importable by it (the dev
venv installs the repo editable).

Working directory: the cache directory.

## 10. Exit and failure semantics

- Server exit code 0 after SIGINT, SIGTERM, `shutdown`, or parent death.
- Exit code 1 with the reason on stderr when no backend comes up, the cache is
  not writable, or an argument is wrong. The app shows the last 20 stderr lines
  in this case.
- The app treats an unexpected exit while serving as a crash and restarts with
  backoff (5, 10, 20, 40, 60 s; gives up after five restarts in ten minutes).
