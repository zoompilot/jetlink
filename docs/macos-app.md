# Jetlink for Mac

The Mac app that runs the server for you, with no terminal. For the Jetson, see
the [Jetson guide](jetson.md). For the command line on any platform, see
[platform setup](platforms.md).

## What it does

Jetlink for Mac runs the inference server your comma connects to, and keeps it
running while you drive. It downloads the models over your Mac's network,
prepares them, and keeps the one you chose loaded, so the comma is ready at
once. It also shows what is on disk, which is worth watching: a prepared model
is about 10 GB.

## Requirements

| What you need | Why |
| --- | --- |
| A Mac with Apple silicon | The app is arm64 only. Intel Macs are not supported. |
| macOS 15 or later | The app uses system features added in macOS 15. |
| 16 GB of memory recommended | Preparing a CoreML model peaks near 8 GB. |
| About 11 GB of disk per model | A 766 MB download plus a 10 GB prepared engine. |
| A USB-A port on a hub, dock or adapter | Going through USB-A makes the Mac take the host role reliably. |
| A USB 3 A-to-C data cable | Charge-only cables do not work. |

You also need a comma running a zoompilot build with Jetlink in it, set up with
the steps in the [README](../README.md#quick-start).

## Install

1. Download the Mac ZIP from [Releases](https://github.com/zoompilot/jetlink/releases).
2. Double-click the ZIP to unzip it, then drag Jetlink.app to Applications.
3. Open Jetlink from Applications.

On first launch the server starts by itself and the Status screen says
**Waiting for comma**. That is the normal resting state until you plug the
comma in.

### If the build is not signed

Pre-release builds and builds from a fork are not signed by an identified
developer. macOS refuses to open them the first time and says:

> Apple could not verify "Jetlink" is free of malware that may harm your Mac or
> compromise your privacy.

Two ways to get past it:

- Right-click Jetlink in Applications, choose **Open**, and confirm. On recent
  macOS versions you instead open **System Settings > Privacy & Security**,
  find the message about Jetlink, and click **Open Anyway**.
- Or remove the quarantine flag in Terminal:

```bash
xattr -d com.apple.quarantine /Applications/Jetlink.app
```

Signed releases need none of this.

## Prepare a model before you drive

Open **Models**. The list is the same one the comma shows under
**Settings > Models > Big Model**, in the same order. Pick the model your comma
is set to use. The model marked **Default** is the fork's default,
and is the right choice if you have not changed it on the comma.

1. Select the model and choose **Download**. The Status column counts up, with
   the transfer rate beside it. How long the 766 MB takes depends on your
   connection.
2. Choose **Prepare**. The status becomes **Preparing** with a progress bar and
   the server's own message under it, then moves to a loading stage.
3. Wait for **Loaded**.

What the steps mean:

| Status | What is happening |
| --- | --- |
| Not downloaded | The model is in the list, but its file is not on your Mac. |
| Downloading | The ONNX file is coming down over your network. |
| Downloaded | The file is on disk and can be prepared. |
| Preparing | The Mac is compiling the model for its own hardware. |
| Prepared | A compiled engine is on disk, but it is not loaded. |
| Loaded | The model is in memory and the comma gets it immediately. |

Preparing with CoreML takes about 10 seconds the first time, and loading a
prepared engine again takes about 2 seconds. It used to be about 9 minutes
each way: the compiled model carried 4 GB of its weights as text, which every
load parsed. The weights now go to the compiled model's weight file, so both
ends are seconds.

Closing the window is fine. The server keeps running and the menu bar icon
stays. Quitting Jetlink stops the server, and the next start loads the engine
again in about 2 seconds.

## Plug in

Connect the **Mac's USB-A port to the comma's USB-C port**, using a USB-A port
on a hub or dock, or a USB-C-to-A adapter. A plain C-to-C cable may not give
the Mac the host role.

The Status screen then shows:

- **Connected over USB** on the Link row.
- **Rate**, the frames per second the comma is sending. It should settle near
  20 per second.
- **Frame time**, as mean, p99 and max in milliseconds. The budget is 50 ms.
- **Slow frames**, the number of frames over 60 ms in the last second. Zero is
  what you want. A number here that stays above zero means the Mac is not
  keeping up, and the comma may drop back to its small model.

On the comma, the home-button icon pulses while it fetches and hands over the
model, then turns green. The rest of the driving behaviour is unchanged and is
described in the [README](../README.md#what-to-expect-when-driving).

## Settings

**General**

| Setting | What it does |
| --- | --- |
| Start server when Jetlink opens | Starts the server as soon as the app launches. On by default. |
| Open Jetlink at login | Adds Jetlink as a login item, so it is running before you get in the car. |
| Keep the Mac awake while serving | Holds off idle sleep. The caption says it plainly: "Only when connected to power. On battery, keep the lid open." |
| Cache folder | Where models and prepared engines live. A CoreML engine is about 10 GB. Changing it takes effect when the server restarts. |

The cache folder defaults to `~/Library/Application Support/Jetlink/cache`. If
you already used `scripts/run-mac.sh`, you have a `models_cache/` folder beside
a checkout. Point the cache folder at it with **Choose**, and nothing is
downloaded or prepared again.

**Server**

| Setting | What it does |
| --- | --- |
| Backend | Which runtime prepares and runs the model. See [Backends](#backends). |
| Connection | **USB (the comma)** for driving, or **TCP (bench client)** for testing without a comma. |
| Port | The TCP port, 5599 by default. Only shown for TCP. |
| Log level | **Normal (INFO)** or **Verbose (DEBUG)**. Use verbose when reporting a problem. |
| Python interpreter override | For development only. Leave it empty to use the bundled runtime. |

Changes to these apply when the server restarts, and the footer has a
**Restart server** button.

## Backends

Automatic is CoreML on the GPU, which is the measured best choice on an M1 Pro.
The numbers below are from that machine; a newer Mac has to be measured, not
assumed. Details and the full method are in
[backends and measurements](backends.md#mac-measured).

| Backend | Frame time on an M1 Pro | Prepare and load | Pick it when |
| --- | --- | --- | --- |
| Automatic (recommended) | 43 ms | About 10 seconds to prepare, about 2 seconds each later load | Always, unless you have a reason not to. |
| CoreML on the GPU | 43 ms, no frame over budget in 390 | About 10 seconds to prepare, about 2 seconds each later load | You want the automatic choice pinned. |
| CoreML with the Neural Engine | 45 ms at 20 Hz, 69 frames of 390 over budget | Not measured again since the weights moved out of the compiled model | You are measuring on a faster Mac. |
| tinygrad on Metal | 66 ms, over the 50 ms budget every frame | About 15 seconds to prepare, about a second to load | You want a quick start for a bench test. |

Disk goes with that choice: a CoreML engine is about 2.3 GB, a tinygrad engine
is 777 MB.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| The server failed to start | Open **Logs**. The last lines say why. The usual causes are another server already holding the USB device, and a cache folder that is not writable. |
| It waits for the comma forever | Use a USB-A port on a hub, dock or adapter, use a USB 3 data cable, and check that **Accelerator Link** is on under Settings > Models on the comma. |
| Preparing takes a long time | CoreML should take about 10 seconds to prepare and about 2 seconds to load. Minutes means an engine prepared before the weights moved out of the compiled model; forget it under **Models** and prepare it again. Memory pressure makes it longer either way, so close other large applications while it prepares. |
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
turned on **Open Jetlink at login**, remove it in
**System Settings > General > Login Items**.

## For developers

Building the app, the embedded Python runtime, signing and notarizing are
covered in `macos/README.md` in the checkout.

The same download, prepare and inventory work is available as a command line
tool on every platform, and the running server has a control channel. Both are
documented in [models and the model CLI](models.md).
