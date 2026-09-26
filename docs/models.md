<a id="models-and-the-model-cli"></a>

# Choose and prepare models

For normal setup, select a model under **Settings > Models > Big Model** on
the comma while parked and online. The comma downloads it, sends it to the
server, and waits for the server to prepare it. Start with the default model.
See [daily use](using-jetlink.md#choose-a-model) for model changes and switching.

## Prepare ahead of time (optional)

Use the server's internet connection to download a model before connecting the
comma. This can save time when the comma is on LTE.

On Mac, use **Models > Download**, then **Prepare**, and wait for **Loaded**.
Select the same model on the comma. See the [Mac guide](macos-app.md#prepare-a-model-before-you-drive).

### On a Jetson or an installed PC

List the available models, then fetch one using its full 40-character ref:

```bash
jetlink models list --json
jetlink models fetch <ref>
```

Replace `<ref>` with the full `ref` value for your chosen model in the JSON.
The plain `list` output abbreviates refs for display. To prepare it as well, stop the server
first. Run these steps while parked:

```bash
jetlink stop
jetlink models prepare <ref>
jetlink start
```

**Do not prepare a model in a separate process while the server uses the same
cache.** Concurrent builds are unsupported and can exhaust memory. The Mac
app prepares through the running server and does not need this stop/start step.

### From a source install

Activate the Python environment used to install Jetlink, then use
`jetlink-models list --json`, `jetlink-models fetch <ref>`, and
`jetlink-models prepare <ref>`. Stop any server using that cache before
running `prepare`.

## Downloads, prepared engines, and disk space

A download is the original ONNX model. A prepared engine is the version built
for your computer's backend and device. Jetlink keeps both so it can reuse them.
A runtime update may require another preparation; it keeps the download.

Most model downloads are about 766 MB. Prepared engine sizes vary; on a Mac,
allow about 3 GB total per model with the default backend.

| Installation | Default cache folder |
| --- | --- |
| Jetson or PC installer | `/mnt/data/jetlink` |
| Mac app | `~/Library/Application Support/Jetlink/cache` |
| Mac source script | `models_cache/` in the checkout |
| Other Mac terminal setup | `~/Library/Caches/jetlink` |
| Other desktop setup | `~/.cache/jetlink` |

The Mac app lets you choose a cache folder in Settings. Source-install commands use
`JETLINK_CACHE` or `--cache DIR` to override their default. The installer wrapper
uses the server's cache automatically.

Use `jetlink models inventory` with the installer, or `jetlink-models inventory`
from a source install, to inspect disk use. The Mac app shows it under **Models**.

## Commands

See the [model command reference](model-cli.md#commands) for listing, fetching,
importing, preparing, and deleting models, including arguments and examples.

## The control channel

For scripts and app development, see the [server control protocol](control-protocol.md).
It lets a client manage downloads and preparation through a running server.

<a id="model-identifiers-and-storage"></a>
<a id="the-protocol"></a>

Model identifiers and cache internals are documented in the
[command reference](model-cli.md#model-identifiers-and-storage); JSON messages
are documented in the [protocol reference](control-protocol.md#the-protocol).

<a id="list"></a>
<a id="resolve"></a>
<a id="fetch"></a>
<a id="import"></a>
<a id="inventory"></a>
<a id="rm"></a>
<a id="prepare"></a>
<a id="exit-codes"></a>

The former command sections are now in the [CLI reference](model-cli.md#commands).

<a id="starting-a-server-with-a-control-socket"></a>
<a id="on-connect"></a>
<a id="example"></a>
<a id="events"></a>
<a id="commands-1"></a>

The former protocol sections are now in the [control reference](control-protocol.md).
