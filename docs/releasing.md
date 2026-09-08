# Updates and rollback

Update the comma build and Jetlink server as a pair while parked. Use the Jetlink
commit pinned by your fork's `jetlink_repo` entry, then rebuild the server using
your [platform guide](platforms.md) or [Jetson setup](tester-setup.md). For a Jetson
boot service, record the new image ID in `/etc/jetlink/server.env` and restart the
service after stopping the old server. Keep the previous image and configuration
until the new pair is verified.

If setup fails, turn off **Settings > Models > Accelerator Link** to disable
Jetlink. To roll back Jetlink itself, restore the previous fork and server pair;
restoring just one side can leave them incompatible. Keep the model cache.

## Compatibility

Jetlink 0.2.0 uses protocol 2 and rejects protocol 1 peers. The fork's
`jetlink_repo` submodule points to the required Jetlink commit. Use that revision
for the server as well as the client.

Cached engines also depend on the model, GPU, and runtime version. Keep caches
when updating, but allow time for another build if the new setup needs one.

## Check the update

Restart both devices while parked. Confirm model preparation finishes, the
connection becomes ready, and the selected model is correct. If the check fails,
keep Jetlink disabled until you have diagnosed the problem or restored the
previous working pair.
