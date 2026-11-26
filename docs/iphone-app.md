# Set up Jetlink on iPhone or iPad

**Experimental.** Set up offroad with the comma and phone online.

<a id="requirements"></a>

## What you need

- A USB-C iPhone on iOS 26.1 or later, or USB-C iPad on iPadOS 26.1 or later.
  Use a USB 3 model; see [compatible devices](#usb-3-matters).
- Free space: about 5 times the model's size plus 3 GB to download and
  prepare it (7 GB for a 766 MB model, 12 GB for Lebowski), and the model's
  size plus 3 GB kept free while it runs. See [storage](#storage).
- A comma 3X or comma 4, powered separately.
- A USB 3 USB-C data cable. The comma does not charge the phone over it; a
  powered USB 3 hub keeps the phone charging.

<a id="install"></a>

## 1. Install Jetlink

<a href="https://testflight.apple.com/join/DAsYk5sP"><img src="images/testflight-badge.svg" alt="Available on TestFlight" height="40"></a>

1. Open **[the Jetlink beta](https://testflight.apple.com/join/DAsYk5sP)** on the phone.
2. Install **TestFlight** if prompted, then tap **Accept** and **Install**.

<a id="connect-the-comma"></a>

## 2. Connect the comma

The comma runs an openpilot fork with Jetlink support (zoompilot, StarPilot).
The comma's steps and its home-button icon below are zoompilot's; on another
fork, follow its own docs.

1. **Install zoompilot.** After resetting the comma, enter
   **`zoompilot/main`** as the install URL. Already on zoompilot? Select
   **main** in **Settings > Software > Target Branch**.
   Wait for installation and the reboot to finish.
2. Set **Settings > Models > Jetlink** to **iOS**.
   Leave **Big Model** at its default.
3. Open Jetlink on the phone and allow **Local Network** access.
4. Connect the phone to the comma with the cable.
5. Keep Jetlink on screen and wait for **Connected over USB 3**.

Stay offroad and online until the comma's home-button icon turns **green**.
It pulses while the model downloads and prepares.

<a id="status"></a>

Check for a rate near **20 frames per second** and **zero slow frames**.
Run the benchmark below before use. Read [daily use](using-jetlink.md) before driving.

<a id="benchmark"></a>

## 3. Check performance

1. Leave the model loaded and disconnect the comma.
2. Put the phone in its car mount with charging connected.
3. Open **Benchmark** and tap **1 Minute**.

**Fast Enough** leaves time for the cable and comma. **Tight** leaves little
margin; **Too Slow** cannot sustain 20 frames per second. Run **10 Minutes**
to check for slowdowns as the phone heats up. Repeat after changing the model
or processor.

The benchmark excludes cable latency. Passing does not validate operation
with the comma connected.

<a id="limits"></a>
<a id="heat"></a>

## While using Jetlink

Keep the app on screen and the phone unlocked. Switching apps or locking the
phone drops the link. Keep the phone out of direct sun and thick cases to
reduce overheating.

<a id="prepare-a-model-before-you-drive"></a>

To download a model ahead of time, see [Models](models.md#prepare-ahead-of-time-optional).

<a id="storage"></a>
## Storage

A model takes about three times its size once prepared: the download, the
prepared copy, and the Neural Engine's own compiled copy in the app's caches.
Preparing it needs about as much again for a while, and the system needs room
to spare: on a nearly full device it clears the Neural Engine's copies, the
Neural Engine has no room to make them again, and models run on the CPU at
hundreds of milliseconds a frame, so the comma stays on its small model.

- **Models** says when a download or a preparation would not fit, and how much
  each needs, before it starts; the app refuses them rather than fill the
  device. A prepared model with too little room to keep running well asks
  before it is used.
- **Status** shows **Low Storage** while the model in use is short of room,
  and **Big Model Running Slowly** when frames take more than twice the
  50 ms budget.
- **Settings > Storage** shows what the downloads, the prepared copies and
  the Neural Engine's copies take, and what is free.
- Deleting a model removes its download, its prepared copies and the Neural
  Engine copies made for it. Prepared copies the app can no longer load, from
  an older version, go by themselves.

The system's own storage screen counts the app's files under Jetlink; what
it counts as System Data is the system's, and only freeing space or restarting
the device gives it back.

<a id="if-a-direct-cable-does-not-connect"></a>
<a id="if-the-big-model-comes-and-goes"></a>

## Troubleshooting

| Problem | First step |
| --- | --- |
| Build expired | Install the newest available build in TestFlight. Builds expire after 90 days. |
| App cannot find the comma | Allow **Local Network** access and check the comma's **Jetlink** setting is **iOS**. |
| Direct cable fails or comma restarts | Update the comma, or use a powered USB 3 hub. |
| Connection says **USB 2** | Check that the phone, cable, and hub all support USB 3. |
| Slow frames or repeated fallbacks | Let the phone cool, repeat the 10-minute benchmark, and save the logs. |
| Frames take hundreds of milliseconds | The Neural Engine is not running the model, usually for want of free space: see [storage](#storage). Free space, or restart the device if there is plenty. |

<a id="logs"></a>

Logs: **Settings > Help > Logs**. Use the share button when
[asking for help](troubleshooting.md#get-help).

<details>
<summary>USB 3 devices</summary>

### USB 3 matters

Every hop must be USB 3: the phone, the cable and any hub.

| USB 3 | USB 2 |
| --- | --- |
| iPhone 15 Pro and later Pro models | Other USB-C iPhones, and the cable in the iPhone box |
| iPad Pro, iPad Air and iPad mini with USB-C | iPad (10th generation), iPad (A16) |

Sources: [Apple, iPhone](https://support.apple.com/en-us/105099),
[Identify your iPad model](https://support.apple.com/en-us/108043).

</details>

<details>
<summary>Optional settings</summary>

## Settings

Keep **Processor** on **Automatic**. It runs a model on **Neural Engine Only**
unless **Compare Processors** on the Benchmark tab has found another best for
it. The others are there to choose by hand:

| Processor | Runs the model on |
| --- | --- |
| **Neural Engine Only** | the Neural Engine, in lower precision. The coolest, and what Automatic uses. Only Compare Processors checks a model's outputs on it. |
| **Neural Engine + GPU** | the camera half on the Neural Engine, the rest on the GPU |
| **GPU Only** | the GPU, for when another app keeps the Neural Engine busy |

Changing it prepares the model again.

<a id="compare-processors"></a>
**Compare Processors** runs the loaded model on each processor for 20 seconds,
waiting between runs until the device is no warmer than when it started, and
sets Automatic to the first of Neural Engine Only,
Neural Engine + GPU and GPU Only whose median frame is within 2 ms of the fastest with
no more frames over the 50 ms budget. Neural Engine Only is taken only if its
outputs match Neural Engine + GPU's on 32 fixed frames, by the rules of
`scripts/verify_parity.py`. The pick is kept per model. The runs take a
minute; preparing the model for each processor takes minutes more, and the
app says how much space that takes before it starts and refuses it when the
device has too little. Twenty seconds does not show how a processor holds
up once the device is hot; run **10 Minutes** on the pick for that. Run it with the
comma unplugged, charging, in the mount, and share the result from the
toolbar if you can: it tells us what to make the default on your device.

Keep **Keep Screen On** enabled. **Keep CPU Awake** can reduce delays between
frames but uses more power. **Keep GPU Awake** is off by default: on an iPad
Air (M1) it made frames slower and less steady. Try it if frames run late.

Tap **Version** under **About** seven times to show **Developer** settings: a
port that bench tools on a Mac can reach the phone on over Wi-Fi. Driving does
not use it; the phone dials the comma over the cable. Tap **Version** seven
times again to hide it and close the port.

</details>

<a id="build-from-source"></a>

Building your own app: [iPhone development](../ios/README.md#install-on-a-device-from-source).
