# Using Jetlink

After [setup](../README.md#quick-start), leave the server running and connect
it to the comma. Keep laptops powered and awake; sleep interrupts the link.

## Check the comma's icon

| Icon | Meaning and next action |
| --- | --- |
| Pulsing | The model is downloading, transferring, or preparing. Wait for it to finish. |
| Green while parked | The large model is ready. |
| Green while driving | The large model is active. |
| Green, dimmed while driving | The large model is ready but cannot switch yet. See below. |
| Orange | Preparation failed. Read the alert on the home screen. |
| Returns to normal after parking | With idle suspend enabled, the comma releases the link so the server can sleep. Otherwise the link stays connected while the comma is awake. |

## What to expect when driving

- The small model drives while the server starts. On the tested Jetson with
  switched power and a cached engine, startup takes 65 to 96 seconds.
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

## Choose a model

While parked and online, open **Settings > Models > Big Model** on the comma.
Start with the default. The comma downloads your selection and sends it to the
server automatically. A new catalog entry appears without a Jetlink update;
use **Refresh Model List** if the list is empty or out of date.

For the Jetson, prefer the 766 MB models. The measured 1.7 GB Lebowski model
leaves little margin within the 50 ms frame budget and needs swap during
preparation. See [measurements and limitations](status.md#measured-performance).

You can also [download and prepare models ahead of time](models.md) using the
server's internet connection. This is optional.

## Update or stop using Jetlink

Follow [updates and rollback](releasing.md) to update both the comma and server.
To stop using Jetlink, turn off **Settings > Models > Accelerator Link**.

If the link does not become ready, start with
[troubleshooting](../README.md#if-something-is-wrong).
