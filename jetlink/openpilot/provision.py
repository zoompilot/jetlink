"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

One provisioning run: `python -m jetlink.openpilot.provision --adapter MODULE`.

The heavy half of jetlink, numpy and the client, which is why it is a run and
not a daemon. The owner (jetlink.openpilot.owner, as jetlinkd) holds the gadget
for the whole time the link is enabled and starts a run when something
changes; the run borrows the endpoint files from it exactly as modeld does
(jetlink.comma.lending), so the gadget never leaves the bus and a parked car
keeps one resident jetlink process of about 13 MB instead of a run's 47. Only
the owner ever holds ep0: a run that gets no loan logs it and exits, and the
owner starts another.

The result is cached on the Jetson, recorded in the spec record and left
loaded on the server. The owner stops a run at the onroad transition with
SIGTERM, so every long wait polls `stop`; the server's build thread carries on
regardless and modeld picks the engine up over its own link.

A server whose hello lists `addons` is sent the fork's add-on models it has
neither ready nor failed (Stock.addon_models), once the driving model is
ready, over the same link and the same upload, and builds them there; the
hello after says what it has now, so the owner's record does too.

What the owner cannot learn for itself it hears from the run: the server's
hello, passed on over the loan (lending.Loan.note_server), and whether the
round left work undone, which is the exit status: 0 when nothing is left,
anything else to be tried again once the owner's backoff has passed.
"""
from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

from jetlink import protocol as P
from jetlink.comma import gadget, lending
from jetlink.openpilot import link

# how long to wait for the Jetson to enumerate before giving up on this run.
# The owner presented the gadget; a box that is asleep answers the bind in
# about 8 s, one that is off never does and the next run will find it
WAKE_TIMEOUT = 20.0
# an add-on model is megabytes, built in seconds to a minute or two
ADDON_BUILD_TIMEOUT = 600.0


class ProvisioningRun:
  def __init__(self, parts):
    self.parts = parts
    self.log = parts.log
    self.client = None
    self.loan = None
    self.stop = False
    self.fetch_failed = False
    # what came of each add-on model this run, by sha256 (note_addons)
    self.addon_results: dict[str, dict] = {}

  # -- lifecycle ------------------------------------------------------------

  def request_stop(self, *_) -> None:
    self.stop = True

  def close_link(self) -> None:
    """Always go through this: a FunctionFS owner that exits without closing
    can wedge the driver until a reboot."""
    client, self.client = self.client, None
    if client is not None:
      try:
        client.leave(P.LEAVE_PROVISIONED)
        client.close()
      except Exception:
        self.log.exception("jetlink: error closing the link")

  def open_link(self) -> bool:
    """Borrow the link from the owner that started this run. Without a loan
    there is nothing to open: only the owner ever holds ep0."""
    if self.client is not None:
      return True
    try:
      self.loan = lending.borrow('provision')
      if self.loan is None:
        self.log.error("jetlink: the owner lent no link, nothing to provision over")
        return False
      self.client = link.connect(self.log, self.loan, deadline=5.0, name='provision')
      return True
    except Exception:
      self.log.exception("jetlink: could not open the link")
      return False

  # -- provisioning ---------------------------------------------------------

  def fetch_model(self):
    """Download the picked large model, if the comma has not got it whole.

    Minutes on a slow link, so it reports progress and stops when the owner
    stops the run; it is the one call in a run that blocks for long. A stall is
    retried within the run and resumed from where it stopped, by this run or
    the next (Models.fetch_shipped_model); a run gives up only once those
    retries are spent, and the .part waits for the next one.
    """
    if self.fetch_failed:
      return None
    parts = self.parts
    model = parts.models.selected_model() or {}
    size = model.get('size') or 0
    got = 0.0

    def progress(frac: float) -> None:
      nonlocal got
      got = frac
      got_mb, total_mb = int(frac * size) >> 20, size >> 20
      parts.progress.report('download', frac, f"downloading {got_mb} of {total_mb} MB" if size else 'downloading',
                            code='downloading', args={'got_mb': got_mb, 'total_mb': total_mb} if size else None)

    def retrying(error, delay: float) -> None:
      parts.progress.report('download', got, f"download stalled, retrying in {delay:.0f} s",
                            code='download_stalled', args={'retry_s': round(delay)})

    try:
      path = parts.models.fetch_shipped_model(progress=progress, should_stop=lambda: self.stop, retrying=retrying)
    except Exception:
      if self.stop:
        self.log.warning("jetlink: download stopped, the next run carries on from where it got to")
      else:
        # the owner starts another run after its backoff, which resumes the .part
        self.log.exception("jetlink: could not fetch the large model")
        parts.progress.report('failed', 1.0, 'download failed, retrying later', code='download_failed')
      self.fetch_failed = True
      return None
    if path is not None and model.get('oid'):
      link.remember_hash(path, model['oid'])   # the download proved it; the upload need not read it again
    return path

  def needs_download(self) -> bool:
    """Could the Jetson need the picked model from the comma? Only while its
    engine is not known to be built and the file is not here."""
    parts = self.parts
    entry = parts.models.selected_model()
    if entry is None:
      return False
    sha256, _ = link.identity(parts, entry)
    if parts.spec.engine_ready_for(sha256):
      return False
    return parts.models.shipped_model_path() is None

  def provision(self) -> bool:
    """Make the Jetson ready for the picked model; a host is attached. The
    hello settles which pick that is, the comma's or a newer one made on the
    server. A server without the model downloads it itself; only one that
    cannot (no internet, or too old) gets the bytes from the comma, fetched
    here then."""
    from jetlink.client import EngineMissing
    parts = self.parts
    parts.progress.report('connect', 0.0, 'connecting', code='connecting')
    hello = self.hello()
    self.log.warning("jetlink: server %s trt %s", hello.get('device'), hello.get('trt_version'))
    parts.models.adopt()
    entry = parts.models.selected_model()
    if entry is None:
      # no catalog yet; not an error
      parts.spec.clear_ready()
      parts.progress.clear()
      return False
    sha256, nbytes = link.identity(parts, entry)

    # only needed if the server turns out not to have this model and cannot
    # get it; None is a legitimate state here, see EngineMissing below
    model_path = parts.models.shipped_model_path()

    self.log.warning("jetlink: provisioning %s (%d MB, sha %s)",
                     entry.get('name', sha256[:16]), nbytes >> 20, sha256[:16])

    def ensure(path):
      return link.ensure(parts, self.client, sha256, nbytes, path, progress=parts.progress.report_with_eta,
                         should_stop=lambda: self.stop, ref=entry.get('ref'))

    try:
      spec = ensure(model_path)
    except EngineMissing:
      # the server has nothing to build from and could not download it, and
      # neither have we: fetch the model and hand it over in this run. The
      # owner lends the link for the whole run; leaving after the download
      # would put the build a WORKER_BACKOFF later
      if model_path is not None or (model_path := self.fetch_model()) is None:
        raise
      # a download of minutes can outlast the session; a hello starts a new one
      self.hello()
      spec = ensure(model_path)

    self.log.warning("jetlink: engine ready for %s", spec.sha256[:16])
    finished = self.provide_addons(hello)
    parts.progress.report('ready', 1.0, 'engine ready', code='engine_ready')
    return finished

  def provide_addons(self, hello: dict) -> bool:
    """Send the server the add-on models it lacks, one at a time, then say
    hello so the owner keeps what it has now. False when one is left for the
    next run: a link that failed, or a stop."""
    missing = addons_missing(addon_models(self.parts), hello)
    if not missing:
      self.note_addons('JetLink has every add-on model' if isinstance(hello.get('addons'), dict)
                       else 'JetLink is too old for add-ons')
      return True
    done = True
    for addon in missing:
      if self.stop:
        self.note_addons('stopped: the car started')
        return False
      done = self.provide_addon(addon) and done
    self.hello()
    self.note_addons('sent the add-on models' if done else 'some add-on models are left for the next park')
    return done

  def note_addons(self, result: str) -> None:
    """Say what this run made of the add-ons (gadget.ADDON_RUN), for a fork's
    panel: the run's result and each model it sent. Only with an add-on on."""
    models = addon_models(self.parts)
    if not models:
      return
    names = {m.sha256: m.name for m in models}
    record = {'at': time.time(), 'result': result,
              'models': {sha: {'name': names.get(sha, sha[:16]), **r} for sha, r in self.addon_results.items()}}
    self.log.warning("jetlink: add-ons: %s", result)
    try:
      gadget.write_record(gadget.ADDON_RUN, record)
    except OSError:
      self.log.exception("jetlink: could not write the add-on record")

  def provide_addon(self, addon) -> bool:
    """One add-on model: uploaded and built, or failed on the server, which
    then says so in its hello and is not asked again. False only for what the
    next run may fix."""
    parts = self.parts
    path = Path(addon.path)
    try:
      nbytes = path.stat().st_size
    except OSError:
      self.log.error("jetlink: add-on %s has no model at %s", addon.name, path)
      self.addon_results[addon.sha256] = {'state': 'failed', 'detail': 'the comma has no copy of the model'}
      return True
    upload = link.verified_upload(self.log, path, addon.sha256, nbytes)
    if upload is None:
      self.addon_results[addon.sha256] = {'state': 'failed', 'detail': "the comma's copy is not the file it names"}
      return True

    def progress(stage: str, frac: float, msg: str) -> None:
      text, args = f"{addon.name}: {stage} {frac:.0%}", {'name': addon.name, 'stage': stage}
      if stage == 'build':
        # TensorRT says little while it builds: a percentage would sit at one value for minutes
        text = f"{addon.name}: building on JetLink, several minutes"
      if stage == 'upload':
        parts.progress.report(stage, frac, text, code='addon_upload', args=args)
      else:
        parts.progress.report(stage, frac, text, code='addon_build', args=args)

    from jetlink.client import AddonFailed, LinkError
    self.log.warning("jetlink: sending the jetson add-on %s (%d KB, sha %s)", addon.name, nbytes >> 10, addon.sha256[:16])
    try:
      self.client.ensure_addon(addon.sha256, nbytes, upload, addon.function.iface_class, name=addon.name, progress=progress,
                               build_timeout=ADDON_BUILD_TIMEOUT, should_stop=lambda: self.stop, port=addon.function.port)
    except AddonFailed as e:
      self.log.error("jetlink: the jetson cannot run add-on %s: %s", addon.name, e)
      self.addon_results[addon.sha256] = {'state': 'failed', 'detail': str(e)}
      return True
    except LinkError as e:
      self.log.warning("jetlink: add-on %s is left for the next run: %s", addon.name, e)
      self.addon_results[addon.sha256] = {'state': 'left', 'detail': str(e)}
      return False
    self.log.warning("jetlink: add-on %s ready on the jetson", addon.name)
    self.addon_results[addon.sha256] = {'state': 'ready', 'detail': ''}
    return True

  def hello(self) -> dict:
    """Say hello with our pick, and pass the answer on to the owner, which
    needs its sleep_after to decide whether letting go of the gadget is worth
    what it costs, never speaks the protocol to learn it, and keeps the
    server's pick and whether it downloads for the comma (Models.heard)."""
    models = self.parts.models
    sent = models.hello_pick()
    # the add-ons turned off go: a server that serves them deletes what this leaves out
    keep = addon_keep(self.parts)
    extra = {'addons_keep': keep} if keep is not None else {}
    hello = models.heard(self.client.hello(timeout=10.0, pick=sent, **extra), sent)
    if self.loan is not None:
      self.loan.note_server(hello)
    return hello

  # -- the parked car -------------------------------------------------------

  def without_jetson(self) -> None:
    """No Jetson came to the bind: a car whose Jetson is switched with the
    ignition has none while parked. Only the Jetson needs the model, and one
    that can download it gets it itself when it is on; the comma fetches it
    now only for a Jetson that cannot (no internet when it was last here, or
    too old to say), so the build can start the moment it is."""
    if not self.has_work():
      return
    if self.needs_download():
      if server_fetches():
        self.parts.progress.report('waiting', 0.0, 'the jetson downloads it when it is on', code='waiting_for_host')
        return
      if self.fetch_model() is None:
        return
    self.parts.progress.report('waiting', 0.0, 'downloaded, builds when the jetson is on',
                               code='downloaded_waiting_host')

  def has_work(self) -> bool:
    """Is there a reason to wake the Jetson? Only things the link can fix count."""
    spec = self.parts.spec.load()
    if spec is None or not self.parts.spec.engine_ready_for(spec.sha256):
      return True
    selected = self.parts.models.selected_model()
    if selected is not None and selected.get('oid') != spec.sha256:
      return True
    models, server = addon_models(self.parts), gadget.read_record(gadget.SERVER)
    # no hello since boot: whether the server has them is not known until one
    if models and not server:
      return True
    return bool(addons_missing(models, server)) or addons_extra(addon_keep(self.parts), server)

  def shutdown_jetson(self, reason: str) -> None:
    """hardwared is shutting the comma down and wants the Jetson off too.
    The request file is removed whatever happens: hardwared is waiting on it."""
    self.log.warning("jetlink: shutting the jetson down: %s", reason)
    try:
      if not self.open_link():
        raise RuntimeError("could not open the link")
      if not self.wait_for_jetson():
        raise TimeoutError(f"no jetson attached within {WAKE_TIMEOUT:.0f} s")
      # a hello first: the gadget can have stayed bound since the last
      # borrower, and the server's session with it. This client's seqs start
      # at 1 again, and the session drops anything at or below the last seq
      # it saw as a replay; only a hello starts it over
      self.hello()
      resp = self.client.shutdown(reason, timeout=5.0)
      self.log.warning("jetlink: jetson answered the shutdown request: %s", resp)
    except Exception:
      self.log.exception("jetlink: could not shut the jetson down")
    finally:
      gadget.finish_shutdown()

  # -- one run --------------------------------------------------------------

  def wait_for_jetson(self) -> bool:
    """Has a host enumerated within WAKE_TIMEOUT? A sleeping Jetson wakes to the bind."""
    return gadget.wait_for_host(WAKE_TIMEOUT, bounce=self.bounce, should_stop=lambda: self.stop,
                                mode=self.parts.settings.mode())

  def bounce(self) -> bool:
    """Ask the owner to bounce the gadget, over the lease. On a phone's cable
    the client's rebind is a no-op: nothing is stuck in an endpoint file."""
    try:
      return bool(self.client.rebind()) if self.client is not None else False
    except Exception:
      self.log.exception("jetlink: could not bounce the gadget")
      return False

  def run(self) -> bool:
    """One provisioning round. True when there is nothing left to do, which
    main() makes the exit status the owner reads."""
    # the setting alone, as the owner that started this run reads it
    if self.parts.settings.mode() == 'off':
      return True

    reason = gadget.pending_shutdown()
    if reason is not None:
      self.shutdown_jetson(reason)
      return True

    # without a warp for this camera the engine would never run; the offroad
    # alert says so, and waking the Jetson to build one would not change it
    if not self.parts.warps.built():
      self.log.warning("jetlink: no warp built for this camera, nothing to provision for")
      self.note_addons('waiting for the camera to be calibrated')
      return True

    # Nothing to build, unless a model was picked on the server since: a
    # Jetson awake on the bus is asked, which costs a hello and no wake. Not
    # a phone's cable, which is there only once the app dials in
    if not self.has_work() and not (gadget.host_attached() and gadget.link_kind(self.parts.settings.mode()) == 'usb'):
      self.log.warning("jetlink: nothing to provision")
      self.note_addons('JetLink has every add-on model')
      return True

    finished = False
    try:
      if not self.open_link():
        return False
      if not self.wait_for_jetson():
        self.log.warning("jetlink: no jetson within %.0f s, leaving it for the next run", WAKE_TIMEOUT)
        self.note_addons(f'no JetLink computer answered within {WAKE_TIMEOUT:.0f} s')
        self.close_link()
        self.without_jetson()
        return False
      finished = self.provision()
    except Exception:
      self.log.exception("jetlink: provisioning failed")
      self.parts.progress.report('failed', 1.0, 'see the log', code='failed')
      self.note_addons('the run failed; see the log')
    finally:
      self.close_link()
    return finished


def addon_models(parts) -> tuple:
  """The fork's add-on models turned on now (Stock.addon_models); none on any doubt."""
  return _addon_models(parts) or ()


def _addon_models(parts) -> tuple | None:
  """addon_models, or None when the fork has no hook or it failed."""
  hook = getattr(parts.op, 'addon_models', None)
  if not callable(hook):
    return None
  try:
    return tuple(hook(parts.settings.params))
  except Exception:
    parts.log.exception("jetlink: addon_models failed, sending the jetson no add-ons")
    return None


def addon_keep(parts) -> dict | None:
  """What a server may keep on each of the fork's add-on interfaces, for the
  hello (addons_keep): class triple, then the sha256s turned on, an empty
  list for an interface whose add-ons are all off. A server deletes the rest
  there. None on any doubt, a hook missing or failing, or no add-on
  declared: the server then deletes nothing."""
  from jetlink.addons.wire import interface_triple
  models = _addon_models(parts)
  hook = getattr(parts.op, 'gadget_functions', None)
  if models is None or not callable(hook):
    return None
  try:
    functions = tuple(hook(parts.settings.params))
  except Exception:
    parts.log.exception("jetlink: gadget_functions failed, the jetson keeps every add-on")
    return None
  keep = {interface_triple(f.iface_class): [] for f in functions}
  for m in models:
    triple = interface_triple(m.function.iface_class)
    if triple in keep and m.sha256 not in keep[triple]:
      keep[triple].append(m.sha256)
  return keep or None


def addons_extra(keep: dict | None, server: dict | None) -> bool:
  """Does the server serve a model on one of keep's interfaces that is not
  in it, and delete one on a hello's word (`forgets`)? Never for an older
  server: it would keep it, and the Jetson would be woken every park."""
  addons = (server or {}).get('addons')
  if not keep or not isinstance(addons, dict) or addons.get('forgets') is not True:
    return False
  serving = addons.get('serving') if isinstance(addons.get('serving'), dict) else {}
  return any(set(serving.get(triple) or ()) - set(shas) for triple, shas in keep.items())


def _serves(addons: dict, sha256: str, iface_class) -> bool:
  """Does the server serve sha256 on iface_class's interface? Its hello's
  `serving` says per interface: a model built for one add-on is not served on
  another's until it is asked for there. A server too old to say has only
  `ready`, and served each model on one interface."""
  serving = addons.get('serving')
  if isinstance(serving, dict) and iface_class is not None:
    from jetlink.addons.wire import interface_triple
    return sha256 in (serving.get(interface_triple(iface_class)) or ())
  return sha256 in (addons.get('ready') or ())


def addons_missing(models, server: dict | None) -> list:
  """The add-on models a server that serves add-ons has neither ready on
  their own interface nor failed. None for one whose hello lists no
  `addons`: an older server would take an add-on for its driving model."""
  addons = (server or {}).get('addons')
  if not isinstance(addons, dict):
    return []
  failed = set(addons.get('failed') or ())
  return [m for m in models if m.sha256 not in failed and not _serves(addons, m.sha256, m.function.iface_class)]


def addon_state(sha256: str, server: dict | None, iface_class=None) -> str | None:
  """An add-on model on the server the comma last spoke to, as the owner
  keeps its hello: ready (on iface_class's interface, given one), failed,
  missing, or unsupported for a server too old to serve add-ons. None before
  any hello."""
  if not server:
    return None
  addons = server.get('addons')
  if not isinstance(addons, dict):
    return 'unsupported'
  if _serves(addons, sha256, iface_class):
    return 'ready'
  if sha256 in (addons.get('failed') or ()):
    return 'failed'
  return 'missing'


def server_fetches() -> bool:
  """Did the server the comma last spoke to say it downloads a model itself
  (its hello's `fetches`, as the owner keeps it)? No for one too old to say."""
  return (gadget.read_record(gadget.SERVER) or {}).get('fetches') is True


def main(argv: list[str] | None = None) -> None:
  """One provisioning round over the fork's adapter, as the owner starts it.
  Exits 0 when nothing is left to do, and 1 when the owner should try again."""
  from jetlink.openpilot.interface import load_adapter
  from jetlink.openpilot.parts import for_this_process
  p = argparse.ArgumentParser(prog='python -m jetlink.openpilot.provision', description=main.__doc__)
  p.add_argument('--adapter', required=True, help="the fork's adapter module")
  args = p.parse_args(argv)
  d = ProvisioningRun(for_this_process(load_adapter(args.adapter)))
  # the owner stops this at the onroad transition; closing the link properly is
  # what keeps the driver healthy for modeld
  signal.signal(signal.SIGTERM, d.request_stop)
  signal.signal(signal.SIGINT, d.request_stop)
  raise SystemExit(0 if d.run() else 1)


if __name__ == "__main__":
  main()
