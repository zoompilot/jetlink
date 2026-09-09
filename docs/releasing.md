# Updates and rollback

The comma build and the Jetlink server have to match. Update both while parked.

## Which Jetlink to run

Clone `main`. The zoompilot fork records the exact Jetlink commit it was tested
with as its `jetlink_repo` submodule, and `main` is kept compatible with the
current `jetson-trt` branch. A protocol mismatch is refused cleanly: the server
rejects the connection and the comma keeps driving on the small model.

If the server refuses the comma after an update, check out the commit the fork
pins and rebuild:

```bash
git -C jetlink fetch
git -C jetlink checkout <commit from the fork's jetlink_repo entry>
```

## Updating

1. Update the comma first from **Settings > Software** and let it reboot.
2. On the server, `git pull`. On a Mac, restart `scripts/run-mac.sh`; delete
   `.venv` first if dependencies changed. On a Jetson, rebuild the image and
   point the service at it:

```bash
sudo docker/build.sh
sudo docker image inspect --format 'JETLINK_IMAGE={{.Id}}' jetlink:latest \
  | sudo tee /etc/jetlink/server.env >/dev/null
sudo systemctl restart jetlink-server
```

3. Plug in while parked and wait for the green icon. A new Jetlink or model may
   need another engine build. Cached engines stay valid across updates that do
   not change the model or runtime.

## Rolling back

Turn off **Settings > Models > Accelerator Link** to stop using Jetlink
immediately. To roll back properly, restore the previous comma build and the
previous server image together; restoring one side can leave them incompatible.
Keep the model cache. On the Jetson, `docker image ls` shows earlier images,
and the previous image ID can be written back into `/etc/jetlink/server.env`.
