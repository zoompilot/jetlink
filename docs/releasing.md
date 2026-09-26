# Updates and rollback

The comma build and the Jetlink server must be compatible. Update both while
parked.

## Which Jetlink to run

The installer follows `main` by default. Mac app releases are built from
that branch. The zoompilot fork records the exact Jetlink commit it was tested with
as its `jetlink_repo` submodule, and `main` is kept compatible with the current
`jetson-trt` branch. If the protocol versions differ, the server rejects the
connection and the comma keeps driving on the small model.

To install a release, or the commit the fork records, instead of `main`, pass
it to the installer. Replace `v0.4.0` with a release tag:

```bash
curl -fsSL https://raw.githubusercontent.com/zoompilot/jetlink/v0.4.0/install.sh | bash -s -- --ref v0.4.0
```

## Updating

1. Update the comma first from **Settings > Software** and let it reboot.
2. Update the server using the method you installed:

   - Jetson or Linux PC with the installer: `jetlink update`. It keeps your
     answers, stops the running server (so a Jetson cannot fall asleep part
     way through), fetches the newest `main` (or your `--ref`), and starts the
     new server. If anything fails before the new server is up, it puts the
     previous one back and starts it again.
   - Mac app: quit Jetlink, replace it with the new release, and reopen it.
   - Source install: run `git pull` from the Jetlink checkout. For the Mac
     script, restart `scripts/run-mac.sh`; recreate `.venv` if dependencies
     changed.

3. Plug in while parked and wait for the green icon. A new Jetlink or model may
   need another engine build. Cached engines stay valid across updates that do
   not change the model or runtime.

## Rolling back

Turn off **Settings > Models > Accelerator Link** to stop using Jetlink
immediately. To roll back, restore the previous comma build and the previous
server together; restoring one side can leave them incompatible. Keep the model
cache.

With the installer, run it with the release or commit to go back to:

```bash
curl -fsSL https://raw.githubusercontent.com/zoompilot/jetlink/v0.4.0/install.sh | bash -s -- --ref v0.4.0
```

Or put an earlier image back by hand: `sudo docker image ls` shows the images on
the machine, and the one to run is `JETLINK_IMAGE` in `/etc/jetlink/server.env`
(an image ID from `sudo docker image inspect --format '{{.Id}}' IMAGE`). Then
`jetlink restart`. Each update keeps the settings it replaced as
`/etc/jetlink/server.env.prev`, so going back one update is
`sudo cp /etc/jetlink/server.env.prev /etc/jetlink/server.env` and
`jetlink restart`.

## Publish a release (maintainers)

Release workflows, container tags, and signing secrets are in the
[publishing guide](publishing.md).

<a id="installing-the-app"></a>
<a id="the-container-images"></a>
<a id="signing-secrets"></a>

See [app artifacts](publishing.md#installing-the-app),
[container images](publishing.md#the-container-images), and
[signing secrets](publishing.md#signing-secrets).
