# Paired release and rollback

This is a bench-validation procedure, not authorization to deploy a public
driving release.

Installed pair as of 2026-09-05 17:40 UTC: fork `ee2116199` (jetson-trt) with
Jetlink `58686cf5`, source digest `a2741f0d…`, protocol 2. Comma: `/data/jetlink_repo`
verified against the lock, previous package kept at `/data/jetlink_repo.prev-5f06840`,
fork applied through the updater and a reboot. Jetson: image `jetlink:58686cf`,
ID `sha256:c300ccd1d053…`, built from `/mnt/data/jetlink-src/58686cf`, pinned in
`/etc/jetlink/server.env` (previous ID kept in `server.env.prev-5f06840`),
service enabled, rebooted. Rollback is the previous pair: fork `7ba3b9df50`,
`jetlink_repo.prev-5f06840`, `jetlink:5f06840` (`4d0cec7051bd`).

## Compatibility

Jetlink 0.2.0 uses protocol 2 and rejects protocol 1 peers. Update both ends
while parked; a one-sided update cannot run the accelerator. Existing TensorRT
plans have unchanged model tensors, but still require the same supported
JetPack/TRT/device and model identity. A protocol version is not a complete model
ABI, and source hashes are drift detection, not signed-artifact authentication.

The fork's `openpilot/sunnypilot/accelerators/jetlink/release.json` records the
Jetlink git revision and runtime source SHA-256. On AGNOS, startup verifies the
installed package before gadget setup. Missing/mismatched sources publish a
setup error and prevent Jetlink selection; they must not trigger downloads,
builds or an installation attempt during a driving boot.

## Prepare the pair

1. Commit and test Jetlink. Export a clean tree from that commit, not a dirty
   worktree. Run `python3 scripts/verify_release.py .` from that tree.
2. Put the exact revision, source digest and protocol version into the fork's
   release lock, then commit and test the fork. The digest includes package
   Python sources, immediate script files, pyproject and Dockerfile; it excludes
   logs, tests and documentation. Any runtime source edit requires a new lock.
3. On the Jetson, build the image from the locked source. Record
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
  Export/copy the locked tree into a **new staging directory** outside the
  updater-managed fork. Verify it against the fork lock there first:
  `python3 <staging>/scripts/verify_release.py <staging> <fork>/openpilot/sunnypilot/accelerators/jetlink/release.json`.
- Preserve the existing `/data/jetlink_repo` as a named rollback directory.
  Move the verified staging directory into that path on the same filesystem.
  Do not rsync over a package that running processes may still import.
- Install the paired fork and Jetson service/configuration, keeping model caches
  intact. Validate the service with `systemd-analyze verify` on the Jetson, reload
  systemd, then restart. Inspect the container's actual image ID, not just its tag.
- Reboot both devices. Confirm source-lock success, enabled service, actual
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
