# Set up Jetlink on a Mac

<a id="what-it-does"></a>
<a id="requirements"></a>

## What you need

- An Apple silicon Mac on macOS 15 or later; 16 GB memory recommended.
- About 3 GB of disk space per model.
- A comma 3X or comma 4, powered separately.
- A USB 3 USB-C data cable.

<a id="install"></a>

## 1. Install Jetlink

Set up offroad. Keep the comma and Mac online, and the Mac powered and awake.

1. Download the Mac DMG from [Releases](https://github.com/zoompilot/jetlink/releases).
2. Drag **Jetlink** to **Applications** and open it.
3. Wait for **Waiting for comma**. Leave the settings at their defaults.

<a id="plug-in"></a>

## 2. Connect the comma

The comma runs an openpilot fork with Jetlink support (zoompilot, StarPilot).
The comma's steps and its home-button icon below are zoompilot's; on another
fork, follow its own docs.

1. **Install zoompilot.** After resetting the comma, enter
   **`zoompilot/main`** as the install URL. Already on zoompilot? Select
   **main** in **Settings > Software > Target Branch**.
   Wait for installation and the reboot to finish.
2. Set **Settings > Models > Jetlink** to **USB**.
   Leave **Big Model** at its default.
3. Connect the Mac to the comma with the USB cable.

Stay offroad and online until the comma's home-button icon turns **green**.
It pulses while the model downloads and prepares.

In Jetlink, check for **Connected over USB 3**, a rate near **20 frames per
second**, and **zero slow frames**. If it says USB 2, check the cable and port.
Read [daily use](using-jetlink.md) before driving.

<a id="everyday-use"></a>
<a id="prepare-a-model-before-you-drive"></a>
<a id="use-a-model-before-you-drive"></a>

To download a model ahead of time, see [Models](models.md#prepare-ahead-of-time-optional).

## Troubleshooting

| Problem | First step |
| --- | --- |
| Server failed to start | Open **Logs**. Check that another Jetlink server is not running and the cache folder is writable. |
| Model takes minutes to load | Close large apps. In **Models**, right-click the model, choose **Delete Prepared Engines…**, then use it again. On an M1 Pro, preparing takes about 20 seconds and loading up to 10. |
| Link drops when the Mac sleeps | Enable **Prevent sleep while server is running** and keep it on power, or also enable **Prevent sleep with the lid closed**. |
| Slow frames | Check the cable and port, then close other apps using the GPU or Neural Engine. |
| Network settings show a **jetlink** service | Set the comma's **Jetlink** setting to **USB**. |

[More troubleshooting and logs](troubleshooting.md).

<details>
<summary>Optional settings</summary>

## Settings

| Setting | Use |
| --- | --- |
| Open Jetlink at login | Start the app automatically. |
| Prevent sleep while server is running | Prevent idle sleep on power. On battery, keep the lid open. |
| Prevent sleep with the lid closed | Also on battery, with the lid open or closed. Normal sleep returns when the server stops. |
| Cache folder | Choose where models are stored. Restart the server to apply. |
| Check for updates automatically | Look for a new release once a day. **Check for Updates…**, in the Jetlink menu or the menu bar menu, looks now. |
| Download and install automatically | Download a new release in the background and install it when Jetlink quits. |

Jetlink never shows an update while a comma is connected. A release found then
waits in the menu bar menu as **Update to Jetlink X…**, and appears the next
time you open Jetlink with the comma disconnected.

<a id="backends"></a>
<a id="the-server"></a>

Keep **Backend** on **Automatic**. If another app keeps the Neural Engine busy,
try **GPU Only**. **Neural Engine Only** keeps the GPU idle for
the least power and heat; it runs part of the model in lower precision and
nothing checks its outputs. **Keep CPU Awake** and **Keep GPU Awake** keep a CPU core and the GPU busy
between frames so the next frame does not wait on either; both are on, and
Keep CPU Awake shows as one whole core in Activity Monitor. Changing any of
these restarts the server, unless a comma is connected: then click
**Restart Server** when the drive is over.

**Benchmark** (Command-3) tests the loaded model without a comma connected.

### Bench clients over TCP

Jetlink serves the comma over USB. For bench work it can listen for a
TCP client on every interface instead, with no setting in the app. Quit
Jetlink, then:

```sh
defaults write io.zoompilot.jetlink developerTransport tcp
defaults write io.zoompilot.jetlink tcpPort -int 5599
```

`defaults delete io.zoompilot.jetlink developerTransport` goes back to USB.
For one launch only, pass the same keys as arguments:

```sh
/Applications/Jetlink.app/Contents/MacOS/Jetlink -developerTransport tcp -tcpPort 5599
```

</details>

<details>
<summary>Files and uninstalling</summary>

## Where things live

| What | Where |
| --- | --- |
| Models | `~/Library/Application Support/Jetlink/cache`, or your chosen cache folder |
| Log | `~/Library/Logs/Jetlink/server.log` |
| App | `/Applications/Jetlink.app` |

To uninstall, quit Jetlink and delete the app, `~/Library/Application Support/Jetlink`,
and `~/Library/Logs/Jetlink`. This deletes downloaded models. If you enabled
**Open Jetlink at login**, remove it in **System Settings > General > Login Items**.

</details>
