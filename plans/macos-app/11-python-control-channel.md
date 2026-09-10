# 11. Python: the control channel

**Owner: agent B.** Files you create: `jetlink/server/control.py`,
`tests/test_control.py`. Files you edit: `jetlink/server/session.py`,
`jetlink/server/main.py`. Read first: `01-contracts.md` sections 3.1, 4, 5, 6,
9, 10; then all of `session.py` and `main.py`; then `tests/test_session.py`
and `tests/test_sleep.py` (which calls `main._serve` directly, so its
signature must keep working). You depend on `jetlink.registry.Registry` from
`10-python-registry.md`; if it is not merged yet, write against the signature
in `01-contracts.md` section 5 and stub with a `Protocol`.

## What this is

A local socket the server listens on, speaking JSON lines, through which a
management client (the Mac app; also anything else, `socat` included) sees
what the server is doing and asks it to download, prepare, load, unload and
delete models. It runs on every platform. Nothing about it may slow a frame.

## Edits to `session.py`

Keep every existing behaviour and test green. Add:

1. In `EngineHost.__init__`: `self._listeners: list = []`,
   `self.frame_stats = FrameStats()`, `self._last_stage = (None, 0.0, '')`.
2. `subscribe(fn)` and `emit(kind, payload)`: iterate a copy of the list;
   catch `Exception` per listener, log once with `log.exception`, remove that
   listener. `emit` is called from the job thread, the session thread and the
   main thread; listeners must be thread-safe (the control server's is).
3. `_progress(...)`: after the existing throttle check passes, set
   `self._last_stage = (stage, frac, msg)` and `self.emit('progress', {'stage': stage, 'frac': round(frac, 4), 'msg': msg})`
   in addition to the session call. The throttle already caps this at 4 Hz.
4. `snapshot()`: under `self.lock`, derive the `engine` payload from
   `01-contracts.md` 4.2 using `loaded`, `job`, `_last_stage`. Rules:
   `loaded` set → `ready` with its sha (and the job's stage cleared to
   `stage=None, frac=1.0`); else `job` with state `building` → `loading` if
   `job.load_only` else `building`, with the job's sha and `_last_stage`;
   `job.state == 'failed'` → `failed`, `detail=job.detail`, `stage='failed'`;
   otherwise `none`. Include `load_only`.
5. In `_run`'s `finally`, after `session.engine_update()`:
   `self.emit('engine', self.snapshot())`. Also emit `engine` at the end of
   `_start` (state becomes building/loading immediately) and in `_unload` when
   something was unloaded.
6. `request(req, session)`: change `self.session = session` to only assign when
   `session is not None`. A control-channel `prepare` passes None and must not
   detach the comma's session from progress.
7. Public `unload()`: calls `_unload()` and returns. (`_unload` stays for the
   job thread's use.)
8. `FrameStats` (new small class in `session.py` or `control.py`; put it in
   `session.py` so the host owns it): a `collections.deque(maxlen=2000)` of
   `(monotonic, total_us, gpu_us)`. `record(total_us, gpu_us)` appends; it is
   called from `Session._infer` **after** `self._send(...)` and after the
   slow-frame log, never before the send. `window(seconds)` returns the tuples
   newer than `now - seconds`. `summary(seconds=1.0, frames_total=0)` returns
   the `stats` payload (`fps = n / seconds`, mean/p99/max of total in ms,
   mean gpu in ms, `slow` = count of total_us > SLOW_FRAME_US) or None when
   the window is empty. p99 via `sorted(...)[int(0.99 * (n - 1))]`; no numpy
   here, n is at most ~40.
9. `Session._infer`: one line added after the slow-frame block:
   `host.frame_stats.record(total_us, loaded.engine.last_gpu_us)`.
   `self.frames` already counts; the control server reads it through
   `host.session.frames` when a session exists.

## Edits to `main.py`

1. Arguments: `--control-socket ADDR` (default None), `--parent-pid PID`
   (int, default None).
2. `signal.signal(signal.SIGTERM, signal.default_int_handler)` at the top of
   `main()` after parsing, guarded with `try/except (ValueError, OSError,
   AttributeError)` for platforms and non-main threads.
3. `--parent-pid`: start a daemon thread `jetlink-parent-watch` that every
   1.0 s checks `os.getppid() != pid` and then `os.kill(os.getpid(), signal.SIGINT)`
   (on Windows use `signal.CTRL_C_EVENT`? No: use `_thread.interrupt_main()`,
   which raises KeyboardInterrupt in the main thread on every platform). Log
   one warning line "parent process gone, shutting down".
4. `_serve(cache, open_transport, sleeper=None, host=None, control=None)`:
   create the host only when `host is None` (tests pass none). Emit link
   transitions through `host.emit('link', {...})`: `waiting` once when a poll
   first returns None after a connected/disconnected/start state (not on every
   2 s retry), `connected` with `peer` (`'usb'` for the USB opener; the
   `addr` string for TCP, which means `_tcp_opener` has to expose it: set
   `transport.peer = f"{addr[0]}:{addr[1]}"` on the transport object after
   accept, and `transport.peer = 'usb'` in the USB opener; `getattr(transport,
   'peer', None)` in `_serve`), `disconnected` with the LinkError text as
   `detail` when the session ends.
5. In `main()`, after `cache = EngineCache(...)` and before `_serve`:
   ```python
   host = EngineHost(cache, pick_source(backend.name))
   control = None
   if args.control_socket:
     from jetlink.registry import Registry
     from jetlink.server.control import ControlServer
     control = ControlServer(args.control_socket, host, cache, Registry(cache.root), info={
       'version': <importlib.metadata.version('jetlink') or '0.0.0'>, 'python': platform.python_version(),
       'platform': sys.platform, 'cache': str(cache.root), 'transport': args.transport,
       'port': args.port if args.transport == 'tcp' else None})
     control.start()
   ```
   Pass `host` (and `control`) into `_serve`. On `KeyboardInterrupt` and in
   the `finally`, `control.close()` after `host.close()`. `_serve`'s existing
   `host.preload()` stays where it is (after the control server is listening,
   so a client connecting during preload sees `loading`).
6. The `--build` path and `--list-backends` are unchanged.

## `control.py`

```python
class ControlServer:
  def __init__(self, address: str, host: EngineHost, cache: EngineCache, registry: Registry, info: dict): ...
  def start(self) -> None
  def close(self) -> None
  def publish(self, event: str, payload: dict) -> None   # adds "event" and "t", broadcasts
```

### Listening

`address` is a path or `tcp://127.0.0.1:PORT`. For a path: remove a stale
socket file if `connect` to it fails with ECONNREFUSED (a live server there is
an error: log and raise), `socket(AF_UNIX, SOCK_STREAM)`, bind, `os.chmod(path, 0o600)`,
listen(4). For tcp: bind to 127.0.0.1 only; refuse any other host in the
address. Accept loop on a daemon thread `jetlink-control`. Each client gets:
a reader thread (line-buffered `makefile('rb')`, 1 MiB line cap), a
`queue.Queue(maxsize=1000)` of outgoing lines and a writer thread. A full
queue drops the client (close the socket, log at warning). `close()` shuts
the listening socket, closes clients, removes the socket file.

### On connect

Send `hello` (from `info` plus `protocol: 1`, `pid`), `server` (`serving`,
plus `host.backend.describe()`), `link` (last emitted; `waiting` with empty
detail before any), `engine` (`host.snapshot()`), `inventory`, `catalog`
(`registry.catalog()` with no network: `max_age=inf`), then a `download`
event for each active or queued download.

### Events from the host

`host.subscribe(self._on_host)`:
- `'progress'` → publish `engine` with `host.snapshot()` merged with the
  progress fields (snapshot already carries `_last_stage`; just publish it).
- `'engine'` → publish `engine`; if the state is `ready` or `failed`, also
  publish `inventory` (a build changed the disk).
- `'link'` → remember it, publish `link`. On `connected`, start the stats
  ticker; on the others, stop it.

### Stats ticker

A thread that wakes every 1.0 s while a link is connected, calls
`host.frame_stats.summary(1.0, frames_total=<host.session.frames if host.session else 0>)`,
and publishes `stats` when it returns a payload. `window_s: 1.0`.

### Commands

Dispatch table `{'status': self._cmd_status, ...}`. Each handler gets
`(client, msg)` and returns the reply's extra fields (dict) or raises
`ControlError(message)` for an `ok:false` reply. Any other exception is
logged with a traceback and replied as `ok:false` with `type(e).__name__: e`.

Long operations run on executors and reply immediately:
- `self._net = ThreadPoolExecutor(max_workers=1, thread_name_prefix='jetlink-registry')`
  for `catalog` refreshes and `import`.
- `self._downloads = ThreadPoolExecutor(max_workers=1, thread_name_prefix='jetlink-download')`
  so downloads are FIFO, one at a time. Track them in
  `self._active: dict[sha256, DownloadState]` (state, future, cancel flag,
  ref, last event payload) under a lock, so `cancel_download` can flip the flag
  the `should_stop` callback reads, and so the on-connect replay knows what
  is in flight.

Handlers, following `01-contracts.md` 4.3 exactly:

- `catalog`: submit `registry.catalog(refresh=..., max_age=3600)` then
  `registry.resolve_missing([m.ref for m in models if pointer unknown])`, then
  publish `catalog` (from `registry.catalog(max_age=inf)` so it includes the
  new pointers). On exception publish `catalog` with `error` set.
- `download`: validate exactly one of `ref`/`sha256`. Resolve a ref
  synchronously if the pointer is cached, else on the download worker as the
  first step (the reply then carries `sha256: null`? No: the contract says the
  reply carries `sha256`. Resolve synchronously with a 10 s timeout; a failure
  is an error reply. Pointer fetches are 134 bytes.) Refuse if
  `registry.model_path(sha).exists()` with the right size ("already
  downloaded"), or if already active. Submit `registry.fetch(sha_or_ref,
  progress=..., should_stop=...)`; the progress callback publishes `download`
  events at most 4 a second with `rate_bps` computed from bytes over the last
  second; emit `started` before, `done`/`failed`/`cancelled` after, then
  `inventory` on `done`.
- `cancel_download`: set the flag; if queued and not started, cancel the
  future and publish `cancelled`.
- `import`: `path` must exist and be a file; submit `registry.import_model`
  with progress mapped to `hashing` (frac from the hash) then `copying`
  (frac 0 then 1, or by bytes if you copy in chunks yourself); publish `done`
  with the sha; then `inventory`.
- `prepare`: build `Request(sha256, nbytes, frame_skip)` with `nbytes` per the
  contract (model file size, else sidecar `spec.nbytes` from
  `cache.entry(sha).meta()`, else pointer size via
  `registry` lookup by oid, else 0). If neither `cache.entry(sha).exists` nor
  the model file exists → `ControlError("model <sha16> is not downloaded")`.
  Call `host.request(req, None)`; the reply's `state` is
  `host.snapshot()['state']`.
- `unload`: `host.unload()`.
- `forget`: if `host.loaded_sha() == sha256` → `host.unload()` first. Refuse if
  `host.job` is building this sha ("a build for this model is running"). Then
  `registry.remove(...)`, then publish `inventory`.
- `inventory`: publish it. The payload is `registry.inventory(cache)` with
  `loaded` set from `host.loaded_sha()`.
- `shutdown`: reply, publish `server` with `stopping`, then
  `_thread.interrupt_main()`. Do not exit from the control thread.
- `status`: republish the five snapshots.

### Inventory helper

`_inventory()`: `payload = registry.inventory(cache); payload['loaded'] = host.loaded_sha(); return payload`.
Cache the result for 2 s to absorb bursts (a build finishing emits engine then
inventory; the app may also ask).

## Tests (`tests/test_control.py`)

Use `FakeBackend`/`FakeEngine` from `tests/fake_backend.py` and a real
`EngineHost` + `EngineCache` in `tmp_path`, a `Registry` on the same root, and
a `ControlServer` on a unix socket path under `tmp_path` (short path: use
`tmp_path_factory.mktemp` to keep under 104 bytes; on Windows skip the AF_UNIX
tests and run the `tcp://127.0.0.1:0` form: support port 0 meaning "pick one"
and expose `server.address` for the tests). Write a tiny client helper that
connects, reads lines with a timeout, and sends commands.

Cover at least:
1. On connect the first six events arrive in the contract's order, `hello.protocol == 1`,
   `engine.state == 'none'`, `inventory.models == []`.
2. `status` yields a reply with the right `id` plus the five events.
3. `prepare` with a model file present (write the tiny ONNX from
   `tests/tiny_model.py` and give the FakeBackend its spec) leads to `engine`
   events `building` then `ready`, then an `inventory` event listing the
   artifact with `current: true`; `prepare` again replies `state: ready`
   without a new build (`backend.builds` length unchanged).
4. `prepare` for an unknown sha replies `ok:false` with "not downloaded".
5. `unload` → `engine none`; `forget` with `artifacts: true` removes the
   sidecar and artifact and `last-loaded.json`; `forget` while loaded unloads first.
6. `download` with a patched `Registry.fetch` (monkeypatch to write a file
   and call progress a few times) produces `started`, ≥1 `progress`, `done`,
   then `inventory`; `cancel_download` on a fetch whose fake honours
   `should_stop` produces `cancelled` and no file.
7. `catalog` with a fake opener (reuse the registry test's) publishes
   `catalog` with 13 models and resolved pointers for those the opener serves.
8. A frame served through a real `Session` (borrow the `link` fixture pattern
   from `tests/test_session.py`) makes `host.frame_stats.summary(60)` return a
   payload with `frames >= 1`; the ticker publishes a `stats` event within 2 s
   when the link event `connected` has been emitted.
9. A malformed line gets `reply` with `id: null` and `ok: false`; an unknown
   command gets `ok: false`.
10. `--parent-pid`: unit-test the watcher function with a fake `getppid`;
    do not spawn processes.
11. `main.py`: `_serve` with a fake opener and a host emits `link` `waiting`
    exactly once across three None polls, then `connected`, then `disconnected`
    (drive it with a session that raises LinkError on the first recv).
12. The whole `tests/` suite still passes; `ruff check .` clean.

## Manual check before reporting

From the venv, in one terminal:
```
.venv/bin/python -m jetlink.server.main --backend tinygrad --device CPU --transport tcp --port 5599 \
  --cache /tmp/jl-cache --control-socket /tmp/jl.sock --log-level INFO
```
In another: `nc -U /tmp/jl.sock` (or `socat - UNIX-CONNECT:/tmp/jl.sock`),
watch the six events arrive, type `{"id":1,"cmd":"catalog","refresh":true}`
and see a `catalog` event with 13 models and pointers; type
`{"id":2,"cmd":"download","ref":"f877d7a0ccc3cce943c76e285214c020cd65c899"}` and
watch progress; Ctrl-C the download client, confirm the server keeps running;
`{"id":3,"cmd":"shutdown"}` exits it with code 0.

Report: the diff summary of `session.py` and `main.py`, and confirm nothing
was added to the `_infer` path other than the one `record` call after the send.
