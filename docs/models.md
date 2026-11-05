<a id="models-and-the-model-cli"></a>

# Choose and prepare models

Pick a model on either end: **Settings > Models > Big Model** on the comma,
or **Models** on the Jetson's web page, the Mac app or the phone apps. The
newest pick wins on both. A model picked on the Jetson drives without being
picked on the comma too, and the comma's list shows it after the next
connection. Start with the default.

Only the Jetson needs the model. It downloads the model itself and prepares
it, about 3 minutes, the next time it is on while the car is parked. The
comma downloads the model only when the Jetson cannot reach the internet, then
hands it over. A comma whose Jetson is off while parked downloads it ahead
only if that Jetson had no internet the last time they spoke. Interrupted
downloads pick up where they left off. Until a newly picked model is ready,
the model the device already has loaded drives (on a phone or Mac, the one
picked in the app), else the last one it prepared; with none, the small model
does. See [when it switches](using-jetlink.md#what-to-expect-when-driving).

If the Jetson only has power while the car is on, prepare a new model by
starting the car, forcing offroad, and waiting for the Big Model panel to say
it is ready.

The list is jetlink's model index
([zoompilot/jetlink-model-index](https://github.com/zoompilot/jetlink-model-index)),
refreshed hourly, so new models appear without a Jetlink update. If the list
is empty, use **Refresh Model List**. On Jetson, the 766 MB models
leave more time in the 50 ms frame budget than the 1.7 GB Lebowski.

**Preview Models** are comma models that are not merged into openpilot yet.
comma may still change or withdraw one. When a preview leaves the list, a comma
that had picked it goes back to the default.

## Pick on the Jetson or in the app

Picking here downloads on the server's own connection and prepares the model
before the comma connects. The comma takes the pick at its next connection.

On a Jetson, open the web page's **Models**, choose a model and **Download and
prepare** (or **Prepare it** for one already downloaded).
On Mac, open **Models**, click **Use Model**, and wait for **In Use**.
On iPhone, iPad, or Android, open **Models** and tap **Get**. Keep the iPhone
app open until it finishes.

### On a Jetson or an installed PC

List the models, then fetch one by its full 40-character `ref` from the JSON
(plain `list` shortens refs):

```bash
jetlink models list --json
jetlink models fetch <ref>
```

To prepare it too, stop the server first, with any connected comma offroad:

```bash
jetlink stop
jetlink models prepare <ref>
jetlink start
```

**Do not prepare a model in a separate process while the server uses the same
cache.** Concurrent builds are unsupported and can exhaust memory;
`jetlink models prepare` refuses while the service runs. The apps prepare
through their running server, so they need no stop and start.

<a id="from-a-source-install"></a>

### From a checkout

The same commands on a built `jetlink-server`: `jetlink-server models list
--json`, `jetlink-server models fetch <ref>`, `jetlink-server models prepare
<ref>`. Stop any server using that cache before `prepare`.

## Downloads, prepared engines, and disk space

- A download is the ONNX model, most about 766 MB.
- A prepared engine is built for your backend and device. Jetlink keeps both.
- A runtime update may prepare again; the download is kept.
- On a Mac, allow about 3 GB per model with the default backend.

| Installation | Default cache folder |
| --- | --- |
| Jetson installer | `/mnt/data/jetlink` |
| PC installer | `/var/lib/jetlink` |
| Mac app | `~/Library/Application Support/Jetlink/cache` |
| `jetlink-server` on a Mac | `~/Library/Caches/jetlink` |
| `jetlink-server` on Linux | `/mnt/data/jetlink` on a Jetson, else `/var/lib/jetlink` as root, else `~/.cache/jetlink` |

- Mac app: choose a cache folder in Settings.
- `jetlink-server`: `JETLINK_CACHE` or `--cache DIR`
  ([the rule](installation-reference.md#cache-folder-and-environment)).
- The installer's `jetlink models` uses the server's cache.

Disk use: `jetlink models inventory` (installer), `jetlink-server models
inventory`, or **Models** in the apps.

## Commands

Listing, fetching, importing, preparing and deleting models:
[model command reference](model-cli.md#commands).

<a id="model-identifiers-and-storage"></a>
<a id="the-protocol"></a>
<a id="list"></a>
<a id="resolve"></a>
<a id="fetch"></a>
<a id="import"></a>
<a id="inventory"></a>
<a id="rm"></a>
<a id="prepare"></a>
<a id="exit-codes"></a>
<a id="starting-a-server-with-a-control-socket"></a>
<a id="on-connect"></a>
<a id="example"></a>
<a id="events"></a>
<a id="commands-1"></a>
