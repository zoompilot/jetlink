# Jetlink for Mac

Jetlink for Mac runs the server without terminal commands. For the Jetson, see
the [Jetson guide](jetson.md). For the command line on any platform, see
[platform setup](platforms.md).

<a id="what-it-does"></a>

## Requirements

| What you need | Why |
| --- | --- |
| A Mac with Apple silicon | The app is arm64 only. Intel Macs are not supported. |
| macOS 15 or later | The app uses system features added in macOS 15. |
| 16 GB of memory recommended | CoreML preparation uses about 3 GB on an M1 Pro; allow memory for other apps. |
| About 3 GB of disk per model | A 766 MB download plus a 2.1 GB CoreML engine. |
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

## Plug in

Complete [comma setup](../README.md#comma-setup-all-platforms), including the
branch installation and **Accelerator Link** toggle. Then connect the
**Mac's USB-A port to the comma's USB-C port**, using a USB-A port
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
[daily use guide](using-jetlink.md).

## Everyday use

Keep the Mac powered and awake. You can close the window; the server keeps
running and the menu bar icon stays. Quitting Jetlink stops the server.
The next launch loads the prepared engine again.

## Prepare a model before you drive

This is optional. The comma can download and send the model automatically.
Use these steps to prepare it on the Mac before connecting.

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

Preparing takes about 20 seconds the first time on an M1 Pro, and loading a
prepared engine takes under a second when it was the last model loaded and up to about
10 seconds otherwise.

<details>
<summary>App screenshots</summary>

These screenshots show an earlier app build.

![Server status and model loading](images/mac-status.webp)
![Available models and download status](images/mac-models.webp)

</details>

## Settings

**General**

| Setting | What it does |
| --- | --- |
| Start server when Jetlink opens | Starts the server as soon as the app launches. On by default. |
| Open Jetlink at login | Adds Jetlink as a login item, so it is running before you get in the car. |
| Keep the Mac awake while serving | Prevents idle sleep when connected to power. On battery, keep the lid open. |
| Cache folder | Stores models and prepared engines. A CoreML engine is about 2 GB. Changing it takes effect when the server restarts. |

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

Automatic runs the model's vision layers on the Neural Engine and the rest on
the GPU, the fastest way on a Mac. The measurements below use an M1 Pro;
performance on other Macs may differ. See [backends and
measurements](backends.md#mac-measured).

| Backend | Frame time on an M1 Pro | Pick it when |
| --- | --- | --- |
| Automatic (recommended) | About 30 ms: 32 ms on Cinque Terre V3, 29 ms on V2 | Use this by default. |
| CoreML on the GPU | About 44 ms | Another app keeps the Neural Engine busy. |
| tinygrad on Metal | 66 ms, over the 50 ms budget every frame | Test tinygrad; it exceeds the driving frame budget on this Mac. |

Automatic assumes Jetlink is the only app using the Neural Engine. If you
chose **CoreML on the GPU** in an earlier version, it stays selected; choose
**Automatic** to switch. Disk use depends on the backend: a CoreML engine is
about 2 GB, a tinygrad engine is 777 MB.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| The server failed to start | Open **Logs**. The last lines say why. The usual causes are another server already holding the USB device, and a cache folder that is not writable. |
| The app stays on Waiting for comma | Use a USB-A port on a hub, dock or adapter, use a USB 3 data cable, and check that **Accelerator Link** is on under Settings > Models on the comma. |
| Preparing takes a long time | CoreML should take about 20 seconds to prepare and up to about 10 seconds to load. If it takes minutes, remove the prepared engine under **Models** and prepare it again. Close other large applications to free memory. |
| The comma says **Big Model Lost** | Check the cable first. Then check that the Mac did not sleep: turn on **Keep the Mac awake while serving** and keep the Mac on power. |
| Everything rebuilt after an update | A new runtime version means a new prepared engine, so the model is prepared again. The download is kept and is not fetched twice. |
| The model list is empty | The Mac needs internet for the list. Open **Models** and choose **Refresh**. |
| Frames are slow or the rate is below 20 | Check the cable and the USB port, then check whether another heavy application is using the GPU or the Neural Engine. If one is, choose **CoreML on the GPU** under Settings > Server. |

<details>
<summary>Example server error</summary>

Open **Logs** for the full diagnostic output.

![Server failure and diagnostic output](images/mac-error.webp)

</details>

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
tool on every platform, and the running server has a control channel. See the [model CLI](model-cli.md) and [control protocol](control-protocol.md).
