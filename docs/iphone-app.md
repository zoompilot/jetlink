# Set up Jetlink on iPhone or iPad

**Experimental.** Set up offroad with the comma and phone online.

<a id="requirements"></a>

## What you need

- A USB-C iPhone on iOS 26.1 or later, or USB-C iPad on iPadOS 26.1 or later.
  Use a USB 3 model; see [compatible devices](#usb-3-matters).
- About 3 GB free per model.
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

Keep **Processor** on **Neural Engine + GPU**. **GPU** is an alternative when
another app keeps the Neural Engine busy. Changing it prepares the model again.

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
