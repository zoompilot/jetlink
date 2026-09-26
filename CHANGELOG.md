Jetlink v0.4.0
==============
One command to install, JetPack 7.2, and a faster Mac
* One command installs Jetlink on a Jetson or a Linux PC with an NVIDIA GPU. It sets up Docker and the GPU runtime, puts a Jetson in its fastest power mode with enough swap, and runs Jetlink as a service that starts on boot.
* `jetlink update` keeps your answers. If an update fails, the previous server comes back.
* JetPack 7.2 support. Its newer TensorRT runs the big models 10-12% faster than JetPack 6.2, which is still supported.
* Linux PCs now need NVIDIA driver 580 or newer.
* Cinque Terre V3 and the models after it, with zoompilot v2026.09.25-16 or newer on the comma. Big models sunnypilot publishes later show up without a Jetlink update.
* Fixed a Jetson set to sleep when the car is off never sleeping on stock JetPack.
* Mac
  * The Neural Engine now runs the model's vision half and the GPU the rest: about 31 ms a frame on an M1 Pro, against about 44 before. Automatic uses it.
  * One button, Use Model, downloads, prepares and loads a model. Models prepared by an earlier version prepare once more.
  * Status shows the frame budget: how much of the 50 ms each frame takes, and the last two minutes as a chart.
  * A new app icon, a menu bar icon that shows when a comma is connected, and a model list that looks like the rest of macOS.
