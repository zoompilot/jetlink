# Jetlink for Mac

Jetlink for Mac runs the server without terminal commands. For the Jetson, see
the [Jetson guide](jetson.md). For the command line on any platform, see
[platform setup](platforms.md).

## What it does

Jetlink for Mac runs the inference server your comma connects to, and keeps it
running while you drive. It downloads the models over your Mac's network,
prepares them, and keeps the selected model loaded. You can also check model
status and disk use in the app.

## Requirements

| What you need | Why |
| --- | --- |
| A Mac with Apple silicon | The app is arm64 only. Intel Macs are not supported. |
| macOS 15 or later | The app uses system features added in macOS 15. |
| 16 GB of memory recommended | CoreML preparation uses about 3 GB on an M1 Pro; allow memory for other apps. |
| About 3 GB of disk per model | A 766 MB download plus a 2.3 GB CoreML engine. |
| A USB-A port on a hub, dock or adapter | Going through USB-A makes the Mac take the host role reliably. |
| A USB 3 A-to-C data cable | Charge-only cables do not work. |

You also need a comma running a zoompilot build with Jetlink in it, set up with
the steps in the [README](../README.md#quick-start).

## Install

1. Download the Mac ZIP from [Releases](https://github.com/zoompilot/jetlink/releases).
2. Double-click the ZIP to unzip it, then drag Jetlink.app to Applications.
3. Open Jetlink from Applications.

On first launch the server starts by itself and the Status screen says **Waiting
for comma**. Connect the comma to continue.

### If the build is not signed

Pre-release builds and builds from a fork are not signed by an identified
developer. macOS refuses to open them the first time and says:

> Apple could not verify "Jetlink" is free of malware that may harm your Mac or
> compromise your privacy.

For an unsigned build you trust, use either method:

- Right-click Jetlink in Applications, choose **Open**, and confirm. On recent
  macOS versions you instead open **System Settings > Privacy & Security**,
  find the message about Jetlink, and click **Open Anyway**.
- Or remove the quarantine flag in Terminal:

```bash
xattr -d com.apple.quarantine /Applications/Jetlink.app
```

Signed releases need none of this.

## Prepare a model before you drive

Open **Models**. The list is the same one the comma shows under **Settings >
Models > Big Model**, in the same order. Select the same model as your comma.
Choose **Default** if you have not changed the model on the comma.

1. Select the model and choose **Download**. The Status column shows
   download progress and transfer speed.
2. Choose **Prepare**. Wait while the app compiles and loads the model.
3. Wait for **Loaded**.

What the steps mean:

| Status | What is happening |
| --- | --- |
| Not downloaded | The model is in the list, but its file is not on your Mac. |
| Downloading | The Mac is downloading the ONNX model file. |
| Downloaded | The file is on disk and can be prepared. |
| Preparing | The Mac is compiling the model for its own hardware. |
| Prepared | A compiled engine is on disk, but it is not loaded. |
| Loaded | The model is in memory and ready for the comma. |

Preparing with CoreML takes about 10 seconds the first time, and loading a
prepared engine takes about 2 seconds on an M1 Pro.

You can close the window. The server keeps running and the menu bar icon stays.
Quitting Jetlink stops the server, and the next start loads the engine again in
about 2 seconds.

## Plug in

Connect the **Mac's USB-A port to the comma's USB-C port**, using a USB-A port
on a hub or dock, or a USB-C-to-A adapter. A plain C-to-C cable may not give the
Mac the host role.

The Status screen then shows:

- **Connected over USB** on the Link row.
- **Rate**, the frames per second the comma is sending. It should settle near
  20 per second.
- **Frame time**, in milliseconds: mean (average), p99 (99% of frames are at or below this time), and maximum. The budget is 50 ms.
- **Slow frames**, the number of frames over 60 ms in the last second. This should stay at zero. A consistently higher count means the Mac is too slow, and the comma may drop back to its small model.

On the comma, the home-button icon pulses while the model transfers and loads,
then turns green. For driving behavior, see the
[README](../README.md#what-to-expect-when-driving).

## Settings

**General**

| Setting | What it does |
| --- | --- |
| Start server when Jetlink opens | Starts the server as soon as the app launches. On by default. |
| Open Jetlink at login | Adds Jetlink as a login item, so it is running before you get in the car. |
| Keep the Mac awake while serving | Prevents idle sleep when connected to power. On battery, keep the lid open. |
| Cache folder | Stores models and prepared engines. A CoreML engine is about 2.3 GB. Changing it takes effect when the server restarts. |

The cache folder defaults to `~/Library/Application Support/Jetlink/cache`. If
you already used `scripts/run-mac.sh`, you have a `models_cache/` folder in a
checkout. Point the cache folder at it with **Choose**, and nothing is
downloaded or prepared again.

**Server**

| Setting | What it does |
| --- | --- |
| Backend | Which runtime prepares and runs the model. See [Backends](#backends). |
| Connection | **USB (the comma)** for driving, or **TCP** for testing without a comma. |
| Port | The TCP port, 5599 by default. Only shown for TCP. |
| Log level | **Normal (INFO)** or **Verbose (DEBUG)**. Use verbose when reporting a problem. |
| Python interpreter override | For development only. Leave it empty to use the bundled runtime. |

Click **Restart server** to apply these settings.

## Backends

Automatic uses CoreML on the GPU. The measurements below use an M1 Pro;
performance on other Macs may differ. See [backends and
measurements](backends.md#mac-measured).

| Backend | Frame time on an M1 Pro | Prepare and load | Pick it when |
| --- | --- | --- | --- |
| Automatic (recommended) | 43 ms | About 10 seconds to prepare, about 2 seconds each later load | Use this by default. |
| CoreML on the GPU | 43 ms, no frame over budget in 390 | About 10 seconds to prepare, about 2 seconds each later load | Select CoreML explicitly. |
| CoreML with the Neural Engine | 45 ms at 20 Hz, 69 frames of 390 over budget | Not measured | Test Neural Engine performance on your Mac. |
| tinygrad on Metal | 66 ms, over the 50 ms budget every frame | About 15 seconds to prepare, about a second to load | Test tinygrad; it exceeds the driving frame budget on this Mac. |

Disk use depends on the backend: a CoreML engine is about 2.3 GB, a tinygrad
engine is 777 MB.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| The server failed to start | Open **Logs**. The last lines say why. The usual causes are another server already holding the USB device, and a cache folder that is not writable. |
| The app stays on Waiting for comma | Use a USB-A port on a hub, dock or adapter, use a USB 3 data cable, and check that **Accelerator Link** is on under Settings > Models on the comma. |
| Preparing takes a long time | CoreML should take about 10 seconds to prepare and about 2 seconds to load. If it takes minutes, remove the prepared engine under **Models** and prepare it again. Close other large applications to free memory. |
| The comma says **Big Model Lost** | Check the cable first. Then check that the Mac did not sleep: turn on **Keep the Mac awake while serving** and keep the Mac on power. |
| Everything rebuilt after an update | A new runtime version means a new prepared engine, so the model is prepared again. The download is kept and is not fetched twice. |
| The model list is empty | The Mac needs internet for the list. Open **Models** and choose **Refresh**. |
| Frames are slow or the rate is below 20 | Check the cable and the USB port, then check whether another heavy application is using the GPU. |

## Where things live

| What | Where |
| --- | --- |
| Models and prepared engines | `~/Library/Application Support/Jetlink/cache`, or the cache folder you chose |
| Server log | `~/Library/Logs/Jetlink/server.log` |
| The app | `/Applications/Jetlink.app` |

To uninstall, quit Jetlink, then delete `/Applications/Jetlink.app`,
`~/Library/Application Support/Jetlink` and `~/Library/Logs/Jetlink`. If you
turned on **Open Jetlink at login**, remove it in **System Settings > General >
Login Items**.

## For developers

Building the app, the embedded Python runtime, signing and notarizing are
covered in the [Mac developer guide](../macos/README.md).

The same download, prepare and inventory work is available as a command line
tool on every platform, and the running server has a control channel. Both are
documented in [models and the model CLI](models.md).
