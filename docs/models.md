# Models and the model CLI

Two ways to get a driving model onto a server:

- **Let the comma upload it.** This is the default and needs no commands. The
  comma downloads the model over its own connection, sends it over the link,
  and the server prepares it.
- **Prefetch it.** Download the model on the server's own network with
  `jetlink-models fetch`, then prepare it with `jetlink-models prepare` or
  `jetlink-server --build`. This is faster on a Jetson with a slow connection,
  and it means the comma's first request is answered at once.

The commands are part of the Python package, so they work on a Jetson, a Linux
machine, a Windows machine and a Mac terminal. On a Mac the same work is in the
app; see the [Mac guide](macos-app.md).

## Refs, SHA-256, and where files go

A **ref** is a 40-character commit hash from comma's openpilot repository. It
identifies a model in sunnypilot's big-model catalog, the same list the comma
shows under **Settings > Models > Big Model**.

A **SHA-256** is a 64-character hash of the ONNX file itself. This is what the
comma asks the server for over the link, and what the cache is keyed by. One
ref resolves to exactly one SHA-256, and that never changes.

Anywhere a command takes `REF_OR_SHA256`, a 40-hex string is treated as a ref
and a 64-hex string as a SHA-256. Anything else is an error.

Files live under the cache directory:

| Path | What is in it |
| --- | --- |
| `<cache>/models/` | The downloaded ONNX files, about 766 MB each |
| `<cache>/engines/` | The prepared engines and their sidecar files, one per backend and device |
| `<cache>/registry/` | The cached catalog, the resolved pointers, and records of models you imported |

The cache directory is `JETLINK_CACHE` if it is set, otherwise
`/mnt/data/jetlink` on a Jetson, `~/Library/Caches/jetlink` on a Mac terminal,
and `~/.cache/jetlink` elsewhere. Every command takes `--cache DIR` to override
it.

## The commands

```
jetlink-models list      [--refresh] [--json] [--cache DIR]
jetlink-models resolve   REF [--json] [--cache DIR]
jetlink-models fetch     REF_OR_SHA256 [--cache DIR]
jetlink-models import    PATH [--name NAME] [--cache DIR]
jetlink-models inventory [--json] [--cache DIR]
jetlink-models rm        SHA256 [--artifacts] [--model] [--cache DIR]
jetlink-models prepare   REF_OR_SHA256 [--backend auto] [--device auto] [--cache DIR]
```

`python -m jetlink.registry` is the same program, for when the entry point is
not on `PATH`.

### list

Shows the catalog, newest first, with what is on disk for each entry. The
cached copy is used when it is less than an hour old; `--refresh` fetches a new
one.

```bash
jetlink-models list
```

```
  #  Model                                        Ref         Size    On disk
 12  Cinque Terre Model V2 (September 08, 2026)   37bfa1413e  766 MB  no
 11  BMRLNAP Model v4 (August 30, 2026)           f877d7a0cc  766 MB  prepared (default)
```

`--json` prints the `catalog` payload from the control protocol, so a script
and the Mac app read the same shape:

```json
{"fetched_at": 1757440000.0,
 "url": "https://raw.githubusercontent.com/sunnypilot/sunnypilot-models/refs/heads/gh-pages/docs/driving_models_chestnut_v25.json",
 "default_ref": "f877d7a0ccc3cce943c76e285214c020cd65c899", "error": null,
 "models": [{"name": "Cinque Terre Model V2 (September 08, 2026)", "short_name": "CTMV2",
             "ref": "37bfa1413edcdc2e8844984b83727c33f81d8f46", "build_time": "2026-09-08T00:00:00Z",
             "index": 12, "sha256": null, "bytes": null}]}
```

`sha256` and `bytes` are null until that entry's pointer has been resolved.

### resolve

Turns a ref into the SHA-256 and size the comma will ask for. Resolved pointers
are kept forever, so this costs one small request the first time and nothing
afterwards.

```bash
jetlink-models resolve f877d7a0ccc3cce943c76e285214c020cd65c899
```

```
a086d5249fc308bb...  765953504 bytes
```

The hash is printed in full; it is shortened here.

### fetch

Downloads the ONNX file for a ref or a SHA-256. It resolves the pointer if
needed, downloads to a `.part` file, checks the size and the hash, and only
then renames it into place. Progress goes to standard error, one line per whole
percent, so the output can be piped.

```bash
jetlink-models fetch f877d7a0ccc3cce943c76e285214c020cd65c899
```

```
resolving f877d7a0ccc3cce943c76e285214c020cd65c899
downloading a086d5249fc308bb... 765953504 bytes
  1% ... 100%
verified, saved to /mnt/data/jetlink/models/a086d5249fc308bb.onnx
```

An interrupted download leaves the `.part` file behind and starts again from
the beginning next time. A file appears without `.part` only once it has been
verified.

### import

Adds an ONNX file you already have. The file is hashed, copied into
`<cache>/models/`, and recorded with a name so it shows up in listings.

```bash
jetlink-models import ~/Downloads/big_driving_supercombo.onnx --name "My export"
```

```
hashing /home/me/Downloads/big_driving_supercombo.onnx
a086d5249fc308bb...  765953504 bytes
copied to /mnt/data/jetlink/models/a086d5249fc308bb.onnx
```

### inventory

What is on disk: the downloaded models, every prepared engine with the backend
and device it was built for, and the disk usage.

```bash
jetlink-models inventory
```

```
loaded       none
last loaded  a086d5249fc308bb

models
  a086d5249fc308bb  766 MB  BMRLNAP Model v4 (August 30, 2026)

engines
  a086d5249fc308bb.trt10.3.0.cuda-orin  4.1 GB  trt 10.3.0  built 2026-09-08  current

disk  models 766 MB, engines 4.1 GB, 63 GB free
```

`--json` prints the `inventory` payload from the control protocol, the same
shape the app receives.

### rm

Deletes what you name and nothing else. `--model` removes the downloaded ONNX,
`--artifacts` removes every prepared engine for that model. Deleting the
download keeps the prepared engine working, which is the usual way to reclaim a
few hundred megabytes.

```bash
jetlink-models rm a086d5249fc308bb... --model
```

### prepare

Fetches the model if it is not there, then builds an engine for the chosen
backend, exactly as `jetlink-server --build` does. When nothing has been
recorded as last loaded, this model becomes it, so the next server start
preloads it.

```bash
jetlink-models prepare f877d7a0ccc3cce943c76e285214c020cd65c899
```

```
building a086d5249fc308bb... with trt on cuda
  1% ... 100%
built in 166.4 s, saved to /mnt/data/jetlink/engines/a086d5249fc308bb.trt10.3.0.cuda-orin.plan
```

**Do not run `prepare` while a `jetlink-server` is using the same cache.** Two
processes building into one cache is unsupported, and on a small machine the
two builds together run out of memory. The command warns when it thinks a
server is running, but it cannot always tell. On a Jetson, stop the service
first:

```bash
sudo systemctl stop jetlink-server
```

Or leave the server running and ask it to prepare the model over the
[control channel](#the-control-channel), the only safe way to build while it
serves.

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | Wrong usage, or the ref or model was not found |
| 2 | A network request failed |
| 3 | The download failed verification, by size or by hash |

## On a Jetson

The Docker image carries the same package, so nothing has to be installed. Run
the module with the cache bind-mounted:

```bash
sudo docker run --rm -it -v /mnt/data/jetlink:/mnt/data/jetlink \
  --entrypoint python3 jetlink:latest -m jetlink.registry fetch <ref>
```

The image published to GHCR carries the same command; substitute its name for
`jetlink:latest`.

Every subcommand works this way. Remember the rule above: stop
`jetlink-server` before `prepare`, or use the control channel.

## The control channel

For integrators who want to drive a running server from a script. The Mac app
uses nothing else.

### Starting a server with a control socket

`jetlink-server` takes two extra flags:

```
--control-socket ADDR   open a local control channel; ADDR is a filesystem path
                        (AF_UNIX) or tcp://127.0.0.1:PORT (loopback only, for Windows)
--parent-pid PID        exit cleanly when this process is no longer our parent
                        (checked once a second; a dead parent means getppid() changed)
```

SIGTERM is handled like SIGINT: clean shutdown, engine released, exit code 0.

### The protocol

The channel is a stream socket carrying UTF-8 JSON, one object per line,
newline terminated, with no pretty printing. Absent optional fields are `null`,
never missing, so a decoder can be strict.

Client to server: `{"id": <int>, "cmd": "<name>", ...arguments}`. The client
chooses `id`, positive and increasing.

Server to client: `{"event": "<name>", "t": <unix time, float seconds>, ...}`.
There is exactly one `reply` event per command, carrying that command's `id`.
Any other event may arrive at any time, including between a command and its
reply.

Several clients may connect at once, and every event goes to all of them. A
client that does not read fast enough, more than 1000 queued lines, is
disconnected.

### On connect

The server sends, in this order: `hello`, `server`, `link`, `engine`,
`inventory`, `catalog`, then one `download` event per download in progress. A
client needs no command to render its first screen.

### A worked example

Start a server with a socket:

```bash
jetlink-server --transport usb --control-socket /tmp/jetlink-control.sock
```

In another terminal, connect and watch the events:

```bash
nc -U /tmp/jetlink-control.sock
```

Send a command by typing a line into that same connection, or pipe one in:

```bash
printf '{"id":1,"cmd":"download","ref":"f877d7a0ccc3cce943c76e285214c020cd65c899"}\n' \
  | nc -U /tmp/jetlink-control.sock
```

`socat - UNIX-CONNECT:/tmp/jetlink-control.sock` works the same way and is
easier to script. Prepare the downloaded model without stopping the server:

```bash
printf '{"id":2,"cmd":"prepare","sha256":"<sha256>","frame_skip":4}\n' \
  | socat - UNIX-CONNECT:/tmp/jetlink-control.sock
```

### Events

```jsonc
{"event":"hello","t":0,"protocol":1,"pid":4242,"version":"0.2.0","python":"3.14.7",
 "platform":"darwin","cache":"/Users/me/Library/Application Support/Jetlink/cache",
 "transport":"usb","port":null}
// transport is "usb"|"tcp"|"ffs"; port is set for tcp.

{"event":"server","t":0,"state":"serving","detail":"","backend":"ort",
 "runtime_version":"1.29.0","device":"coreml-Apple_M1_Pro"}
// state: "serving" | "stopping". backend/runtime_version/device are what the
// hello to the comma carries.

{"event":"link","t":0,"state":"waiting","detail":"waiting for a jetlink gadget at 1209:0001","peer":null}
// state: "waiting" | "connected" | "disconnected". peer: "usb" or "host:port" when connected.
// Emitted on transitions only, never on every 2 s poll.

{"event":"engine","t":0,"state":"none","sha256":null,"detail":"","stage":null,"frac":0.0,"msg":"","load_only":false}
// state: "none" | "building" | "loading" | "ready" | "failed"
// stage: "upload"|"patch"|"parse"|"build"|"save"|"load"|"failed"|null, frac 0..1, msg free text.
// Emitted on every state change and on progress at most 4 times a second.

{"event":"stats","t":0,"frames":1234,"fps":19.9,"total_ms":{"mean":31.2,"p99":38.0,"max":41.5},
 "gpu_ms":{"mean":21.0},"slow":0,"window_s":1.0}
// Once a second while a link is connected and at least one frame was served in
// the window. slow counts frames over 60 ms in the window.

{"event":"inventory","t":0,"loaded":"<sha256>|null","last_loaded":"<sha256>|null",
 "models":[{"sha256":"…","bytes":765953504,"path":"…/models/a086d5249fc308bb.onnx","name":"BMRLNAP Model v4 (August 30, 2026)","ref":"f877d7a0…|null"}],
 "artifacts":[{"sha256":"…","key":"a086d5249fc308bb.ort1.29.0.coreml-Apple_M1_Pro","path":"…/engines/a086….ortcache",
               "bytes":5900000000,"backend":"ort","runtime_version":"1.29.0","device":"coreml-Apple_M1_Pro",
               "built_at":"2026-09-08T21:19:15Z","build_seconds":548.9,"checkpoint":"b9facbcc-…","current":true}],
 "disk":{"models_bytes":765953504,"engines_bytes":5900000000,"free_bytes":120000000000}}
// models: every complete <sha16>.onnx in models/ (a .part is not listed).
// artifacts: every engines/*.json sidecar with a spec.sha256, any backend.
//   current is true when the key equals this server's cache key for that sha.
// name: from the catalog or from an imported model, else null.
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

### Commands

| cmd | arguments | reply extras | behaviour |
| --- | --- | --- | --- |
| `status` | | | re-sends `server`, `link`, `engine`, `inventory`, `catalog` |
| `catalog` | `refresh: bool` (default false) | `queued: true` | fetch the catalog when refresh is true, the cache is older than 3600 s, or missing; then resolve pointers for refs without one (parallel, 8 at a time); emit `catalog` when done (also when it fails, with `error`). Never blocks the reply. |
| `download` | `ref` or `sha256` (one of them) | `sha256` | resolve the pointer if needed; enqueue a download (one runs at a time, FIFO); `download` events follow. Error if already downloaded, already queued, or the ref is unknown. |
| `cancel_download` | `sha256` | | cancels a running or queued download; `.part` removed; a `download` event with `cancelled` |
| `import` | `path` | `queued: true` | hash the file (streaming), copy it to `models/<sha16>.onnx` via a `.part`, record its name and size; `import` events, then `inventory` |
| `prepare` | `sha256`, `frame_skip` (default 4) | `state` | the path the comma's request takes, without a comma: build if needed, then load and keep loaded. Error when neither an artifact nor the model file exists. `state` is the engine state afterwards. |
| `unload` | | | release the loaded engine; `engine` event with `none` |
| `forget` | `sha256`, `artifacts: bool`, `model: bool` | | unload first if that model is loaded; delete every `engines/<sha16>.*` when artifacts, `models/<sha16>.onnx` (and `.part`) when model; remove `last-loaded.json` if it names this sha and its artifact is gone; then `inventory` |
| `inventory` | | | emit `inventory` |
| `shutdown` | | | reply, emit `server` with `stopping`, then exit cleanly as SIGINT would |

Errors are plain English sentences in `error`. An unknown `cmd` replies
`ok:false`. A malformed line, one that is not JSON or has no `id`, gets
`{"event":"reply","id":null,"ok":false,"error":"…"}`.
