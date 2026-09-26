# Model command reference

For choosing a model or preparing one before a drive, start with
[model management](models.md). This page documents `jetlink-models`.

## Model identifiers and storage

A **ref** is a 40-character commit hash from comma's openpilot repository. It
identifies a model in sunnypilot's big-model catalog, the same list the comma
shows under **Settings > Models > Big Model**.

A **SHA-256** is a 64-character hash of the ONNX file. The comma and server use
it to identify the model and its cached files. One ref resolves to exactly one
SHA-256, and that never changes.

Most refs have the ONNX in their own tree. Newer ones, starting with Cinque
Terre V3, ship only a precompiled tinygrad file; their commit subject names
the export, and its ONNX comes from comma's model repo on Hugging Face
(`commaai/openpilot_driving_models`). Resolving and fetching work the same for
both kinds.

The list is sunnypilot's big-model catalog plus every newer one it has
published since this release, so a model added later shows up without an
update here. A model only a newer catalog has is built by sunnypilot for its
next runtime, which a jetlink server does not need: it runs the commit's ONNX.

Those newer models also keep their history inside the model. The server feeds
each frame's queues back into the next one, on the GPU under TensorRT, so what
crosses the link per frame is the same as for older models.

Anywhere a command takes `REF_OR_SHA256`, use a 40-character hexadecimal ref or
a 64-character hexadecimal SHA-256 hash. Anything else is an error.

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

## Commands

```
jetlink-models list      [--refresh] [--json] [--cache DIR]
jetlink-models resolve   REF [--json] [--cache DIR]
jetlink-models fetch     REF_OR_SHA256 [--cache DIR]
jetlink-models import    PATH [--name NAME] [--cache DIR]
jetlink-models inventory [--json] [--cache DIR]
jetlink-models rm        SHA256 [--artifacts] [--model] [--cache DIR]
jetlink-models prepare   REF_OR_SHA256 [--backend auto] [--device auto] [--cache DIR]
```

Run these commands in the Python environment used to install Jetlink. You can
use `python -m jetlink.registry` instead of `jetlink-models`. In the syntax
above, square brackets mark optional arguments. Replace uppercase placeholders
such as `REF` and `PATH` with your values; omit the brackets.

The output examples below are shortened for readability.

### list

Lists available models, newest first, and their download or preparation status.
The cached copy is used when it is less than an hour old; `--refresh` fetches a
new one.

```bash
jetlink-models list
```

```
  #  Model                  Ref         Size    On disk
 13  Cinque Terre V3 Model   bf3e3631b3  766 MB  no
 12  Cinque Terre Model V2   37bfa1413e  766 MB  no
 11  BMRLNAP Model v4        f877d7a0cc  766 MB  prepared (default)
```

The table abbreviates refs. Use `list --json` to get the full 40-character
`ref` needed by `fetch` and `prepare`.

`--json` prints the `catalog` payload described in the [control
protocol](control-protocol.md#the-protocol). Example:

```json
{"fetched_at": 1757440000.0,
 "url": "https://raw.githubusercontent.com/sunnypilot/sunnypilot-models/refs/heads/gh-pages/docs/driving_models_chestnut_v26.json",
 "default_ref": "f877d7a0ccc3cce943c76e285214c020cd65c899", "error": null,
 "models": [{"name": "Cinque Terre Model V2", "short_name": "CTMV2",
             "ref": "37bfa1413edcdc2e8844984b83727c33f81d8f46", "build_time": "<ISO 8601 timestamp>",
             "index": 12, "sha256": null, "bytes": null}]}
```

`sha256` and `bytes` are null until that entry's pointer has been resolved.

### resolve

Looks up the model SHA-256 and file size for a ref. The result is cached, so
later lookups do not need a network request.

```bash
jetlink-models resolve f877d7a0ccc3cce943c76e285214c020cd65c899
```

```
a086d5249fc308bb...  765953504 bytes
```

The hash is printed in full; it is shortened here.

### fetch

Downloads the ONNX file for a ref or a SHA-256. It resolves the pointer if
needed, downloads to a `.part` file, checks the size and the hash, and only then
renames it into place. Progress goes to standard error, one line per whole
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

An interrupted download leaves the `.part` file behind and starts again from the
beginning next time. A file appears without `.part` only once it has been
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

Lists downloaded models, prepared engines, their backends and devices, and disk
usage.

```bash
jetlink-models inventory
```

```
loaded       none
last loaded  a086d5249fc308bb

models
  a086d5249fc308bb  766 MB  BMRLNAP Model v4

engines
  a086d5249fc308bb.trt10.3.0.cuda-orin  4.1 GB  trt 10.3.0  current

disk  models 766 MB, engines 4.1 GB, 63 GB free
```

`--json` prints the `inventory` payload from the control protocol, the same
shape the app receives.

### rm

`--model` deletes the downloaded ONNX file. `--artifacts` deletes all prepared
engines for the model. You can remove the download and keep using its prepared
engine. Replace `SHA256` below with the full hash from `resolve` or `inventory
--json`:

```bash
jetlink-models rm SHA256 --model
```

### prepare

Fetches the model if it is not there, then builds an engine for the chosen
backend, exactly as `jetlink-server --build` does. If no model is recorded as
last loaded, the server records this model and loads it at the next startup.

**Do not run `prepare` while a `jetlink-server` is using the same cache.**
Concurrent builds in the same cache are unsupported and can exhaust memory. The
command cannot always detect a running server. On a Jetson, stop the service
first:

```bash
jetlink stop
```

Start it again with `jetlink start` after preparation finishes.

Or leave the server running and ask it to prepare the model over the [control
channel](control-protocol.md), the only safe way to build while it serves.

```bash
jetlink-models prepare f877d7a0ccc3cce943c76e285214c020cd65c899
```

```
building a086d5249fc308bb... with trt on cuda
  1% ... 100%
built in 166.4 s, saved to /mnt/data/jetlink/engines/a086d5249fc308bb.trt10.3.0.cuda-orin.plan
```

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | Wrong usage, or the ref or model was not found |
| 2 | A network request failed |
| 3 | The download failed verification, by size or by hash |

## On a Jetson or an installed PC

The installer's `jetlink models` runs this CLI inside the server image, with the
server's models folder. Replace `<ref>` with a 40-character ref from `list --json`:

```bash
jetlink models list --json
jetlink models fetch <ref>
```

Every subcommand works this way. Stop the server with `jetlink stop` before
`prepare`, and start it again afterwards with `jetlink start`, or use the
control channel.
