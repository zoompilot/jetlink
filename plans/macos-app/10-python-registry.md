# 10. Python: the model registry and the `jetlink-models` CLI

**Owner: agent A.** Files you create: `jetlink/registry/__init__.py`,
`jetlink/registry/catalog.py`, `jetlink/registry/lfs.py`,
`jetlink/registry/cli.py`, `jetlink/registry/__main__.py`,
`tests/test_registry.py`, `tests/fixtures/*` (copied from
`plans/macos-app/fixtures/`). File you edit: `pyproject.toml` (one entry
point). Read first: `01-contracts.md` sections 3.2 and 5, then
`jetlink/server/cache.py` (the layout you inventory) and `jetlink/spec.py`
(`sha256_file`). Do not touch anything under `jetlink/server/` except reading.

## Purpose

Everything about "which large models exist, which one is which file, and how
to get the bytes" in one stdlib-only package that every platform can run. It
mirrors, deliberately, what the fork does on the comma
(`openpilot/sunnypilot/accelerators/jetlink/helpers.py` and `lfs.py` in the
`sunnypilot-jetson-trt` worktree beside this repo): same URLs, same filtering,
same endpoint order, same verify rules. Read those two files once for the
reasoning in their comments; do not import them.

## Behaviour

### Catalog (`catalog.py`)

`parse_catalog(data)`:
- Take `data["bundles"]`. Keep a bundle when `ref` matches `^[0-9a-f]{40}$`,
  `int(minimum_selector_version) == REQUIRED_SELECTOR_VERSION` (it is a string
  in the JSON), and `is_big` is true.
- Map to `CatalogModel(name=display_name or ref[:10], short_name, ref, build_time or "", index=int(index))`.
- Sort by `index` descending (newest first). Deduplicate on `ref`, first wins.
- Never raise on a malformed bundle; skip it.

`fetch_catalog(url, timeout, opener)`: GET, `json.loads`, raise `NetworkError`
on any `URLError`, `HTTPError`, timeout or bad JSON, with the URL in the message.

### Pointers (`lfs.py`)

`parse_pointer_text(text)`: lines `oid sha256:<hex>` and `size <int>`; return
`None` if either is missing or malformed, or if the text is longer than 4096
bytes (a real ONNX served by mistake).

`fetch_pointer(ref, timeout, opener)`: GET `POINTER_URL.format(ref=ref)`,
read at most 4096 bytes, parse; `NetworkError` on transport failure,
`RegistryError("<ref[:10]> did not serve an lfs pointer")` when it parses to None.

`lfs_resolve(endpoint, pointer, timeout, opener)`: POST `{endpoint}/objects/batch`
with the body and headers from `00-architecture.md`. Return the href, or `None`
when the reply has an `error` for the object or no `actions.download.href`.
Transport failure at one endpoint is `None` too (the caller tries the next);
log it at warning level.

`lfs_download(href, pointer, dest, progress, should_stop, opener)`:
- Check free space: `shutil.disk_usage(dest.parent).free >= size + 64 MiB`,
  else `RegistryError` naming both numbers in MB.
- Stream to `dest.with_name(dest.name + ".part")` in 4 MiB chunks, updating a
  `hashlib.sha256` as you go. Call `progress(written/size)` at most once per
  whole percent. Poll `should_stop()` per chunk; when true, delete the `.part`
  and raise `RegistryError("download cancelled")`.
- On completion: size must equal `pointer.size`, hex digest must equal
  `pointer.oid`; otherwise delete the `.part` and raise `VerifyError`. Then
  `part.replace(dest)`. Any other exception: delete the `.part`, wrap in
  `NetworkError`.
- Timeouts: 30 s connect/read on the response object (`urlopen(..., timeout=30)`);
  a stalled server surfaces as `NetworkError` rather than hanging forever.

### `Registry` (`__init__.py`)

State directory `<cache_root>/registry/` with three JSON files, each written
atomically (`.tmp` then `os.replace`):

- `catalog.json`: `{"fetched_at": <float>, "url": ..., "raw": <the fetched json>}`.
- `pointers.json`: `{"<ref>": {"oid": ..., "size": ...}, ...}`. A ref's pointer
  never changes; never refetch a known one.
- `local-models.json`: `[{"sha256", "bytes", "name", "added_at"}]`.

`catalog(refresh, max_age, opener)` returns the event payload from
`01-contracts.md` 4.2: `models` from `parse_catalog(raw)`, each entry given
`sha256`/`bytes` from `pointers.json` when known, `default_ref =
DEFAULT_BIG_MODEL_REF`, `fetched_at` (None when nothing is cached), `url`, and
`error` (None, or the message of a failed refresh, in which case `models` is
from the previous cache). It fetches when `refresh` is true, when there is no
cache, or when `time.time() - fetched_at > max_age`. It does **not** resolve
pointers itself; the caller decides (`resolve_missing`) because that is 13
requests and belongs on a worker.

`resolve(ref, opener)`: pointer from the file, or `fetch_pointer` then save.
`resolve_missing(refs, workers, opener)`: `concurrent.futures.ThreadPoolExecutor`,
returns `{ref: Pointer | Exception}`; saves the successes in one write.

`name_for(sha256)`: search `pointers.json` for an oid match, then the cached
catalog for that ref's name; then `local-models.json`. Returns `(name, ref)`,
either may be None.

`model_path(sha256)`: `<root>/models/<sha256[:16]>.onnx` after validating the
sha with the same regex `EngineCache` uses (`^[0-9a-f]{64}$`); reuse
`EngineCache._validate_sha256` or copy the regex, but the path rule must stay
identical to `EngineCache.model_path` because the server looks there.

`fetch(ref_or_sha256, progress, should_stop, opener)`:
- 40 hex: resolve the pointer. 64 hex: the size must be known from
  `pointers.json` (search by oid) or the caller cannot verify; if unknown,
  raise `RegistryError("size for <sha16> unknown; fetch by catalog ref")`.
- If `model_path` exists with the right size, return it without network.
- For each endpoint in `LFS_ENDPOINTS`: `lfs_resolve`; on an href, `lfs_download`
  and return. If none has it: `NetworkError("no LFS server has <sha16>")`.

`import_model(path, name, progress, should_stop)`: `sha256_file`-style
streaming hash with progress by bytes read, then copy to `model_path` through a
`.part` (skip the copy if the destination already exists with the same size),
append or update `local-models.json` (name defaults to the file's stem), return
the `LocalModel`. Refuse a file that is not an ONNX by magic? No: the server
parses it later and reports; just require the suffix `.onnx` to catch the
obvious mistake.

`inventory(cache)`: build the event payload from `01-contracts.md` 4.2:
- `models`: every `<root>/models/*.onnx` whose stem is 16 hex characters. For
  each, the full sha256 is unknown from the filename alone; find it by matching
  the stem against known oids in `pointers.json`, `local-models.json`, or any
  sidecar's `spec.sha256`. If no full sha is known, still list it with
  `sha256` set to the 16-hex stem padded? No: list it with `sha256` equal to
  the stem and `name` null; document that a 16-character `sha256` means
  "identity unknown beyond the prefix". (The Swift side treats a 16-character
  value as an orphan.)
- `artifacts`: every `<root>/engines/*.json`. Read it; skip on any error. Need
  `spec.sha256` (64 hex). `key` is the file stem. `backend` from the sidecar.
  The artifact path is `engines/<key><suffix>` where suffix comes from
  backend: `{'trt': '.plan', 'tinygrad': '.pkl', 'ort': '.ortcache', 'fake': '.fake'}`;
  fall back to the first existing `engines/<key>.*` that is not the json.
  Skip sidecars whose artifact is missing. `bytes` is the file size or the
  directory's total. `runtime_version` is
  `meta.get('onnxruntime') or meta.get('tinygrad') or meta.get('trt_version')`.
  `checkpoint` is `meta['spec'].get('checkpoint')`. `current` is
  `cache is not None and cache.entry(sha).meta_path == json_path` (guard: if
  `cache.backend` raises because no runtime is installed, `current` is false
  for all).
- `loaded` is None here (the control server fills it), `last_loaded` from
  `EngineCache.last_loaded()`-style parsing of `last-loaded.json` (reuse
  `EngineCache(root).last_loaded()` with `backend=None`; it does not touch the
  backend).
- `disk`: sums plus `shutil.disk_usage(root).free`.

`remove(sha256, artifacts, model)`: `artifacts` deletes every
`engines/<sha16>.*` (file or directory, json included); `model` deletes
`models/<sha16>.onnx` and `.onnx.part`. If `last-loaded.json` names this sha and
`artifacts` was requested, delete `last-loaded.json`. Never raise on a missing
file. Does not know about the loaded engine; the caller (control server) unloads first.

### CLI (`cli.py`, `__main__.py`)

`argparse` with subcommands from `01-contracts.md` 3.2. `--cache` default
`jetlink.server.platform.default_cache_dir()`. Output rules:

- `list`: a table on stdout: `#`, `name`, `ref[:10]`, `built` (build_time
  date), `size` (MB or `?`), `state` (`downloaded` / `prepared` / `-` from the
  inventory). `--json` prints the catalog payload. Without `--refresh` it
  uses the cache when fresh. After listing, it resolves missing pointers only
  with `--refresh` (network), so a plain `list` offline works.
- `resolve REF`: prints `oid size`; `--json` prints `{"ref","sha256","bytes"}`.
- `fetch`: progress on stderr as `\r<pct>% <MB>/<MB>`; prints the final path on stdout.
- `import PATH [--name]`: prints the sha256 and path.
- `inventory [--json]`: a human table (models, then artifacts, then disk) or the payload.
- `rm SHA [--artifacts] [--model]`: at least one flag required. Prints what it removed.
- `prepare REF_OR_SHA [--backend] [--device]`: `fetch`, then import
  `jetlink.server.main` and call `main(['--backend', b, '--device', d, '--cache', root, '--build', str(path)])`.
  Before building, warn on stderr if `<root>/registry/server.pid`… no such file
  exists today; instead warn if a `.part` is in flight or if
  `psutil`-free heuristics are impossible. Keep it to a fixed warning line:
  "building outside the server; stop any running jetlink-server that uses this cache first".

Exit codes as the contract says. `main(argv=None) -> int`; `__main__` calls
`sys.exit(main())`. Entry point in `pyproject.toml`:
`jetlink-models = "jetlink.registry.cli:main"`.

## Tests (`tests/test_registry.py`)

Offline. Copy the four fixture files to `tests/fixtures/`. Build a fake
`opener` that maps URLs to responses (an object with `.read(n)`, `.read()`,
context manager, `.status`) and raises `urllib.error.URLError` for unknown URLs.

Cover at least:
1. `parse_catalog` on the fixture: 13 models, newest first, the first is
   `37bfa1413edcdc2e8844984b83727c33f81d8f46`; a bundle with
   `minimum_selector_version: "18"` is dropped; a bad ref is dropped.
2. `parse_pointer_text` on the fixture and on garbage, an oversize body, a
   missing size.
3. `Registry.catalog` caches: second call makes no request; `max_age` expiry
   refetches; a failing refresh returns the old models with `error` set.
4. `resolve` fetches once and never again (opener call count).
5. `resolve_missing` with one ref failing returns the exception for it and the
   pointer for the others.
6. `lfs_resolve` on the response fixture returns the href; on the missing
   fixture returns None; on a transport error returns None.
7. `fetch` end to end with a small fake object (a few KB): the first endpoint
   says missing, the second serves; `.part` gone, file present, progress ended
   at 1.0. Hash mismatch raises `VerifyError` and leaves nothing behind. Size
   short raises `VerifyError`. `should_stop` returning True mid-way raises and
   leaves nothing behind.
8. `import_model` hashes, copies, records; importing the same file twice keeps
   one record.
9. `inventory` against a cache directory built by hand with a fake sidecar
   (use `tests/fake_backend.FakeBackend` and a real `EngineCache` to write one,
   the way `tests/test_cache.py` does) and an ORT-style `.ortcache` directory:
   both listed, `current` true only for the matching backend, bytes summed for
   the directory, a sidecar without a spec skipped, a `.part` not listed.
10. `remove` deletes artifacts across backends, the model, and
    `last-loaded.json` when it named the sha.
11. CLI: `main(['list', '--json', '--cache', tmp])` with a patched opener prints
    valid JSON with 13 models; `main(['rm', sha])` without flags returns 1;
    `main(['resolve', 'nothex'])` returns 1.

Run `ruff check .` and `pytest tests/test_registry.py` before reporting.

## Report back

State: the commands you added, any place the contract had to bend, and the
line counts of new files. Confirm `jetlink-models list --refresh` against the
live network works from the venv (`.venv/bin/jetlink-models list --refresh`;
this needs `pip install -e .` to pick up the new entry point).
