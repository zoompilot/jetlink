# Paired release and rollback

This is a bench-validation procedure, not authorization to deploy a public
driving release.

Installed pair as of 2026-09-08 20:15 UTC: Jetson image `jetlink:0183b19`, ID
`sha256:e76d57cf8317…`, built from a git clone at `/mnt/data/jetlink-src/0183b19`
(`deployed-image.txt` there records ID and sha), pinned in `/etc/jetlink/server.env`
(previous ID kept in `server.env.prev-0183b19`), host files reinstalled from the
same tree, service enabled, rebooted. The fork is `20557ce5b` (jetson-trt) with
its `jetlink_repo` pin still at `0b02a59` (main): the client, protocol, spec and
queue modules are identical between the two shas, so the pair runs, and the pin
is bumped once the branch is merged. Rollback is `jetlink:819aa374`
(`648fcd8c109d…`), the pair before that is in `server.env.prev-819aa374`.

## Compatibility

Jetlink 0.2.0 uses protocol 2 and rejects protocol 1 peers. Update both ends
while parked; a one-sided update cannot run the accelerator. Existing TensorRT
plans have unchanged model tensors, but still require the same supported
JetPack/TRT/device and model identity. A protocol version is not a complete model
ABI, and source hashes are drift detection, not signed-artifact authentication.

The fork pins the Jetlink package as the `jetlink_repo` submodule; the gitlink
sha is the installed revision. On AGNOS, startup reads `JetlinkEnabled` before
touching the gadget and refuses to present it when the submodule is not checked
out, publishing a setup error instead. Nothing at boot downloads, builds or
installs the package during a driving boot.

## Prepare the pair

1. Commit and test Jetlink. The fork pins the package as the `jetlink_repo`
   submodule, so the commit sha is the release identity; there is no separate
   lock file any more.
2. Bump the submodule pointer in the fork (`git -C jetlink_repo checkout <sha>`,
   commit the gitlink), then commit and test the fork. Any runtime source edit
   is a new sha and a new pointer.
3. On the Jetson, build the image from the pinned source. Record
   `docker image inspect --format '{{.Id}}' <tested-tag>` alongside the fork
   revision, source digest, JetPack/TRT, model hashes and validation results.
   Archive that image for rollback; do not rely on rebuilding a floating base
   image or dependency range to reproduce it.
4. Before installing the new service, create `/etc/jetlink/server.env` containing
   `JETLINK_IMAGE=sha256:<the full 64-hex image ID>`. Keep it root-owned and not
   writable by unprivileged users. The service rejects mutable tags or missing
   configuration. Preserve the previous unit and environment file.

## Install while parked, with stable bench power

- Stop the affected processes before changing package ownership or symlinks.
  Export/copy the pinned tree into a **new staging directory** outside the
  updater-managed fork. Verify it is the pinned sha first:
  `git -C <staging> rev-parse HEAD` must equal `git -C <fork> rev-parse HEAD:jetlink_repo`.
- Preserve the existing `/data/jetlink_repo` as a named rollback directory.
  Move the verified staging directory into that path on the same filesystem.
  Do not rsync over a package that running processes may still import.
- Install the paired fork and Jetson service/configuration, keeping model caches
  intact. Validate the service with `systemd-analyze verify` on the Jetson, reload
  systemd, then restart. Inspect the container's actual image ID, not just its tag.
- Reboot both devices. Confirm the submodule sha, enabled service, actual
  container image, cached engine identity, warmup, and `modelV2.big` only after
  valid inference. Test each boot order, absence, unplug/replug and recovery.
- A failed verification or partial install is a failed rollout, even if the
  small model remains available. Keep the vehicle offroad until diagnosed.

For rollback, stop affected processes and restore the **entire previous pair**:
fork revision, client directory, service unit, environment/image ID. Do not
delete the model cache or previous image during validation. Reboot and verify
the restored pair. An installer that automates staging, interrupted-update
recovery and rollback is still required for a supported public distribution.

## Required hardware evidence

Use the checklist in `release-readiness.md`. At minimum: matching native-build
tests, QCOM reset timing and numerical replay, long-run end-to-end latency under
load, cable/USB fault injection, controls/driver-monitoring noninterference,
cold boot orderings and brownout recovery. Power-role behavior and full
suspend/wake/low-battery/long-parking shutdown are not qualified. Do not enable
real unattended poweroff solely on the basis of the earlier dry-run tests.
