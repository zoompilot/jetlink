# Model command reference

Reference for `jetlink models`, the installer's command, which runs
`jetlink-server models` on the server's models folder. To choose or prepare a
model before a drive, start with [model management](models.md).

## Model identifiers and storage

- A **ref** is a 40-character commit hash from comma's openpilot repository. It
  names a big model, as listed under **Settings > Models > Big Model** on the
  comma.
- A **SHA-256** is the 64-character hash of the ONNX file. One ref resolves to
  exactly one SHA-256, and that never changes.
- Both come from the **model index**,
  [zoompilot/jetlink-model-index](https://github.com/zoompilot/jetlink-model-index),
  which works out every big model's ONNX once an hour and publishes
  `index/v1.json`: the models to list, in order and by folder, and the ONNX
  and size of every ref it has seen, so an old pick still resolves. New models
  appear without a Jetlink update. The server keeps the last index that parsed
  and works offline with it; one it cannot use (another schema, or nothing
  listed) is refused and the previous one kept.
- A ref the kept index does not know, or a SHA-256 it cannot size, fetches the
  index again once before the command gives up.
- The ONNX itself comes from comma's LFS servers, asked in turn: comma's
  Hugging Face bucket by SHA-256, then the GitLab and Hugging Face LFS
  servers, then comma's model repo (`commaai/openpilot_driving_models`).

| Path | What is in it |
| --- | --- |
| `<cache>/models/` | The downloaded ONNX files, about 766 MB each |
| `<cache>/engines/` | The prepared engines and their sidecar files, one per backend and device |
| `<cache>/registry/` | `index.json`, the last model index that parsed, and `local-models.json`, records of models you imported |

Default cache folders: [model management](models.md#downloads-prepared-engines-and-disk-space).
`--cache DIR` overrides it on every command.

## Commands

```
jetlink models list      [--refresh] [--json]
jetlink models resolve   REF [--json]
jetlink models fetch     REF_OR_SHA256
jetlink models import    PATH [--name NAME]
jetlink models inventory [--json]
jetlink models rm        SHA256 [--artifacts] [--model]
jetlink models prepare   REF_OR_SHA256 [--backend auto|trt|ort] [--device D]
```

- From a checkout, run `jetlink-server models ...` with the same arguments;
  each takes `--cache DIR`. `jetlink models prepare` passes a PC's
  `--tensorrt-libs` itself; `jetlink-server models prepare` on a PC needs it
  ([the server command](installation-reference.md#the-server-command)).
- `REF_OR_SHA256` is a ref, or the SHA-256 of a model the model index knows.
- Output a script reads goes to standard output; progress and warnings go to
  standard error.

| Command | Does |
| --- | --- |
| `list` | The models the index lists, in its order, with each model's size and state (`downloaded`, `prepared`). Uses the kept index if under an hour old; `--refresh` fetches it again. The table shortens refs; `--json` prints the [control protocol](control-protocol.md)'s `catalog` event fields, each model's `folder` included. |
| `resolve` | Prints the SHA-256 and size in bytes the index gives a ref, listed or not. A lookup in the kept index, so it needs no network. |
| `fetch` | Downloads the ONNX to a `.part` file, checks size and hash, renames it into place, and prints its path. An interrupted download starts over. |
| `import` | Hashes and copies an ONNX file you already have into `<cache>/models/`, records its name, and prints the SHA-256 and path. |
| `inventory` | Downloaded models, prepared engines with backend and device, and disk use. `*` marks the engine a server started here would load. `--json` prints the `inventory` event fields. |
| `rm` | `--model` deletes the download, `--artifacts` every prepared engine; name at least one. A prepared engine keeps working after its download is removed. |
| `prepare` | Fetches the model if needed, then builds an engine and loads it once, as `jetlink-server build` does. That makes it the model loaded last, which a server started on this cache preloads. |

**Do not run `prepare` while a server uses the same cache.** Concurrent builds
are unsupported and can exhaust memory. On a Jetson or PC, `jetlink models
prepare` refuses while the service runs:

```bash
jetlink stop
jetlink models prepare f877d7a0ccc3cce943c76e285214c020cd65c899
jetlink start
```

The apps prepare through their running server instead.

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | Refused, or the ref or model was not found |
| 2 | A network request failed |
| 3 | The download failed verification, by size or by hash |
