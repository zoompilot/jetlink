# Jetlink

Run openpilot's large driving models on a computer connected to your comma.
The comma handles cameras and vehicle control; Jetlink runs the model and
returns predictions 20 times per second.

**Jetlink is experimental.** It requires zoompilot's
[`jetson-trt` branch](https://github.com/zoompilot/zoompilot/tree/jetson-trt).
The comma uses its small model while the link is unavailable. If the link drops
while engaged, the comma soft-disables and tells you to take over. Read the
[testing status and limitations](docs/status.md) before use.

## Quick start

You need a **comma 3X or comma 4**, a **USB 3 A-to-C data cable**, and
**separate power for the comma and computer**. Stay parked and keep the comma
online during setup.

1. Install Jetlink on your computer using one of the options below.
2. Complete [comma setup](#comma-setup-all-platforms).
3. Wait for the comma's icon to turn green.

### Jetson or Linux PC

For a Jetson Orin or an Ubuntu/Debian PC with an NVIDIA GeForce RTX 20 series
or newer GPU, run:

```bash
curl -fsSL https://raw.githubusercontent.com/zoompilot/jetlink/main/install.sh | bash
```

The installer checks the computer, asks about your setup, installs dependencies,
and starts Jetlink. Allow 10 to 30 minutes, mostly for downloads.

A Jetson needs JetPack first. Follow the [Jetson guide](docs/jetson.md) for
installation and power choices. JetPack 6.2 has in-car test results; the
supported 7.2 image has not yet been tested on hardware. For PC requirements,
see [Linux setup](docs/platforms.md#linux-nvidia-gpu).

When the installer finishes, run `jetlink status` to check the server, then
continue with comma setup below.

### Mac

You need Apple silicon and macOS 15 or later; 16 GB of memory is recommended.
Mac has bench test results; Jetson is the tested in-car setup.

1. Download the Mac ZIP from [Releases](https://github.com/zoompilot/jetlink/releases).
2. Unzip it and drag **Jetlink.app** to **Applications**.
3. Open Jetlink. **Waiting for comma** means the server is ready to connect.
4. Keep the Mac powered and awake, then complete comma setup below.

The app includes its dependencies. If macOS blocks an unsigned build, follow
[the Mac install guide](docs/macos-app.md#if-the-build-is-not-signed).

## Comma setup (all platforms)

1. **Install the branch.** On a comma running zoompilot, open
   **Settings > Software > Target Branch > Non-Prebuilt Branches** and select
   **jetson-trt**. Let it update, reboot, and finish building. Coming from another
   fork? Follow [zoompilot's installation instructions](https://github.com/zoompilot/zoompilot/tree/jetson-trt).
2. **Enable Jetlink.** Open **Settings > Models** and turn on
   **Accelerator Link**. Leave **Big Model** on its default for the first run.
3. **Connect USB.** Connect the computer's **USB-A port** to the comma's
   **USB-C port** with a USB 3 data cable. On Mac, use a USB-A hub, dock, or
   USB-C-to-A adapter. On Jetson, use its USB-A port. Charge-only cables will
   not work, and a plain C-to-C connection may select the wrong USB role.
4. **Wait for green.** The comma's home-button icon pulses during download,
   transfer, and preparation, then turns green when ready. Stay parked and
   online until it finishes. You do not need to download a model manually.

After download and transfer, the default model takes about 3 minutes to prepare
on Jetson or about 20 seconds on an M1 Pro. Later loads use the cached engine.

## What to expect when driving

The small model runs until the large model is ready and can switch. Switching
requires **a stop with cruise off, or lateral control off**. Disengaging alone
is not enough when lateral control is always on. A dimmed green icon means the
model is waiting to switch; **Big Model Ready** means it has switched.

If you hear **Big Model Lost** while engaged, take over. The comma soft-disables
and falls back to the small model. See [daily use and icon meanings](docs/using-jetlink.md)
for startup, model changes, and reconnection behavior.

## If something is wrong

| Problem | First check |
| --- | --- |
| No Accelerator Link toggle | Confirm the `jetson-trt` branch in Settings > Software. |
| Server stays waiting; icon never pulses | Check the server is running, use a USB-A port, and try another USB 3 data cable. |
| Model list is empty | Connect the comma to the internet and use Refresh Model List. |
| Setup alert or orange icon | Read the home-screen alert. Check internet access, then toggle Accelerator Link off and on. |
| Link drops repeatedly | Check the cable, separate power supplies, cooling, and whether the computer slept. |

For logs and more checks, use the [Jetson guide](docs/jetson.md#troubleshooting),
[Mac guide](docs/macos-app.md#troubleshooting), or
[PC guide](docs/platforms.md#troubleshooting).

<a id="more"></a>

## Documentation

[All guides and references](docs/README.md) ·
[Daily use](docs/using-jetlink.md) ·
[Updates and rollback](docs/releasing.md) ·
[Testing status](docs/status.md)

## License

[MIT](LICENSE).
