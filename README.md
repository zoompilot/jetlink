# Jetlink

Run openpilot's large driving models on a computer plugged into your comma.
The comma keeps the cameras and vehicle control. It sends prepared camera
images over USB, the other computer runs the model, and predictions come back
20 times per second.

Jetlink is experimental. It needs a zoompilot build with Jetlink built in. The
comma side lives on the [zoompilot `jetson-trt` branch](https://github.com/zoompilot/zoompilot/tree/jetson-trt).
The small model you picked in sunnypilot keeps driving whenever the link is down. If the link
drops while engaged, the comma soft-disables and tells you to take over. See
[status and known limitations](docs/status.md).

## What you need

- A comma 3X or comma 4.
- A computer to run the model. A **Jetson Orin Nano Super (8 GB)** is the
  tested in-car setup. A **Mac with Apple silicon** works for bench testing.
  Linux and Windows PCs with an NVIDIA GPU are [supported but untested](docs/platforms.md).
- A USB 3 **A-to-C data cable**. Charge-only cables do not work.
- Separate power for both devices. Neither one powers the other.

## Quick start

### 1. Set up the comma

Everything happens on the comma's screen.

1. **Settings > Software > Target Branch > Non-Prebuilt Branches**, select
   **jetson-trt**. Let it update, reboot, and finish building.
2. **Settings > Models**, turn on **Accelerator Link**. The **Big Model** row
   is the model Jetlink runs, the same list a comma with a chestnut board
   picks from. Leave it on the default for your first run.

### 2. Start the server

**Mac (Apple silicon)**, in Terminal:

```bash
brew install python libusb
git clone https://github.com/zoompilot/jetlink.git
cd jetlink
scripts/run-mac.sh
```

The first run installs dependencies. The server then prints that it is waiting
for a gadget, which means it is waiting for the comma. Leave the terminal open.

**Jetson Orin Nano**: follow the [Jetson guide](docs/jetson.md). It is the
same idea in Docker, plus a service that starts the server at boot.

**Other computers**: see [platform setup](docs/platforms.md).

### 3. Plug in and wait

Connect the computer's **USB-A port to the comma's USB-C port**. On a Mac, use
a USB-A port on a hub or dock, or a USB-C-to-A adapter. Going through USB-A
makes the computer take the host role reliably; a plain C-to-C cable may not.
On the Jetson, use a USB-A port, not its USB-C.

Stay parked with the comma online. The home-button icon pulses while the comma
downloads the model, sends it over, and the server prepares it. A Jetson takes
about 3 minutes for the default model. A Mac takes about 9 minutes, and repeats
that wait every time the server restarts, so keep it running.

| Icon | Meaning |
| --- | --- |
| Pulsing | Downloading, transferring, or preparing the model. Keep waiting. |
| Green | Parked: ready. Driving: the large model is driving. |
| Green, dimmed | Driving: ready and waiting for a chance to switch. Stop with cruise off, or turn lateral control off. |
| Orange | Preparation failed. Read the alert on the home screen. |
| Back to normal a minute later | Only with `--sleep-after`: the comma let the link go so the computer can sleep. Otherwise it stays connected the whole time you are parked. |

## What to expect when driving

- The small model drives while the server starts. On a computer that powers up
  with the car, the large model is prepared 65 to 96 seconds later.
- It takes over only when nothing is steering: **at a stop with cruise off, or
  with lateral control off**. Until then the icon is dimmed and the comma says
  **Big Model Available** at every stop. Disengaging alone is not enough on a
  car with lateral control always on.
- A **Big Model Ready** chime means it has taken over.
- Picking a new model needs the comma online once, while parked, to download
  it. After that it is prepared wherever you are: drive off in the middle and
  the small model drives, the panel counts the preparation down, and the large
  model joins at the first chance to switch.
- **Big Model Lost** while engaged is a soft disable. Take over. The small model
  drives, and Jetlink reconnects and switches back at the next chance.

To stop using Jetlink, turn off **Settings > Models > Accelerator Link**.

## If something is wrong

| Problem | Try |
| --- | --- |
| No Accelerator Link toggle | Check the branch in Settings > Software. |
| Toggle is on, nothing happens | Read the setup alert on the home screen. |
| Big Model list is empty | Connect the comma to the internet and use Refresh Model List. |
| Server keeps waiting, icon never pulses | Check the server is running, use a USB-A port, try another USB 3 data cable. |
| Orange icon | Read the alert, check the comma's internet, then toggle Accelerator Link off and on. |
| Model drops out repeatedly | Check the cable, separate power supplies, and cooling. |

The [Jetson guide](docs/jetson.md#troubleshooting) has more, including how to
collect logs when reporting a problem.

## More

- [Jetson setup, boot service, troubleshooting, logs](docs/jetson.md)
- [Mac, Linux, Windows, Docker, and testing without a comma](docs/platforms.md)
- [Status, known limitations, and measured performance](docs/status.md)
- [Updates and rollback](docs/releasing.md)
- [Cables, networking, and power](docs/transport.md)
- [Backends and measurements, for developers](docs/backends.md)

## License

[MIT](LICENSE).
