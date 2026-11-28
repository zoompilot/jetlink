"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The comma's one root script and its wrapper. The wrapper against a fake
subprocess.run; the script's port and vm subcommands against a fake debugfs
and /proc/sys (tests/comma_fakes.py), run without sudo through its override
variables. gadget, net, check and teardown need a real configfs and are not
tested here.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from jetlink.comma import root
from tests.comma_fakes import (STOCK, TUNED, cable_netdev, dual_role, ffs_log_off, pe_params, proc_sys, read_all, read_sys, record,
                                run_script, udc_glue, usb_icl, usbpd, voter)

REPO = Path(__file__).resolve().parents[1]


class FakeRun:
  def __init__(self, returncode=0, stderr='', raises=None):
    self.returncode, self.stderr, self.raises = returncode, stderr, raises
    self.calls = []

  def __call__(self, argv, **kwargs):
    self.calls.append((argv, kwargs))
    if self.raises is not None:
      raise self.raises
    return subprocess.CompletedProcess(argv, self.returncode, None, self.stderr)


@pytest.fixture
def fake_run(monkeypatch):
  """A fake subprocess.run, on AGNOS whatever this machine is."""
  def install(**kwargs):
    fake = FakeRun(**kwargs)
    monkeypatch.setattr(root.subprocess, 'run', fake)
    monkeypatch.setattr(root, 'AGNOS', True)
    return fake
  return install


# -- the wrapper ------------------------------------------------------------

def test_the_script_is_the_packages():
  assert root.SCRIPT.resolve() == REPO / 'jetlink' / 'comma' / 'jetlink-root.sh'
  assert root.SCRIPT.is_file()
  assert os.access(root.SCRIPT, os.X_OK)


@pytest.mark.skipif(shutil.which('git') is None or not (REPO / '.git').exists(), reason='needs a git checkout')
def test_the_script_is_executable_in_git():
  # so a checkout's copy, and the stub under scripts/comma, run as themselves too
  for path in (root.SCRIPT.resolve(), REPO / 'scripts' / 'comma' / 'jetlink-root.sh'):
    entry = subprocess.run(['git', 'ls-files', '-s', str(path.relative_to(REPO))], cwd=REPO, capture_output=True, text=True,
                           check=True).stdout
    assert entry.split(' ')[0] == '100755', (path, entry)


def test_the_old_path_forwards_to_the_script(tmp_path):
  # an owner from before the move still runs scripts/comma/jetlink-root.sh; after v0.9.x this goes
  old = REPO / 'scripts' / 'comma' / 'jetlink-root.sh'
  assert os.access(old, os.X_OK)
  proc_sys(tmp_path, STOCK)
  assert run_script(tmp_path, 'vm', 'apply', script=old).returncode == 0
  assert read_all(tmp_path, TUNED) == TUNED


@pytest.mark.parametrize('args, timeout', [
  (('gadget',), None),
  (('gadget', '--ios'), None),
  (('net',), None),
  (('teardown',), None),
  (('vm', 'apply'), None),
  (('port', 'hold'), root.PORT_TIMEOUT),
])
def test_run_is_sudo_bash_script_args(fake_run, args, timeout):
  fake = fake_run()
  assert (root.run(*args) if timeout is None else root.run(*args, timeout=timeout)) is True
  [(argv, kwargs)] = fake.calls
  assert argv == ['sudo', '-n', 'bash', str(root.SCRIPT), *args]
  assert kwargs['timeout'] == (root.TIMEOUT if timeout is None else timeout)
  assert kwargs['stdout'] == subprocess.DEVNULL
  assert kwargs['stderr'] == subprocess.PIPE


def test_off_agnos_nothing_runs_and_nothing_is_logged(fake_run, monkeypatch, caplog):
  fake = fake_run()
  monkeypatch.setattr(root, 'AGNOS', False)
  with caplog.at_level(logging.INFO, logger='jetlink.comma'):
    for args in (('gadget', '--ios'), ('net',), ('port', 'off'), ('vm', 'apply'), ('vm', 'restore')):
      assert root.run(*args) is False
  assert fake.calls == []
  assert caplog.records == []


def test_a_failure_logs_the_last_stderr_line_once(fake_run, caplog):
  fake_run(returncode=1, stderr='a warning first\njetlink: could not release the voter\n')
  with caplog.at_level(logging.INFO, logger='jetlink.comma'):
    assert root.run('port', 'off', timeout=root.PORT_TIMEOUT) is False
  [record] = caplog.records
  assert record.levelno == logging.ERROR
  assert 'port off' in record.getMessage()
  assert 'exit 1' in record.getMessage()
  assert 'could not release the voter' in record.getMessage()
  assert 'a warning first' not in record.getMessage()


def test_a_failure_with_nothing_on_stderr_still_logs(fake_run, caplog):
  fake_run(returncode=1)
  with caplog.at_level(logging.INFO, logger='jetlink.comma'):
    assert root.run('vm', 'restore') is False
  [record] = caplog.records
  assert 'no output' in record.getMessage()


@pytest.mark.parametrize('error, words', [
  (subprocess.TimeoutExpired(['sudo'], 2.0), 'timed out'),
  (FileNotFoundError(2, 'No such file or directory', 'sudo'), 'did not run'),
  (PermissionError(13, 'Permission denied'), 'did not run'),
])
def test_run_never_raises(fake_run, caplog, error, words):
  fake_run(raises=error)
  with caplog.at_level(logging.INFO, logger='jetlink.comma'):
    assert root.run('port', 'hold', timeout=root.PORT_TIMEOUT) is False
  [record] = caplog.records
  assert words in record.getMessage()


# -- the script -------------------------------------------------------------

def test_the_script_parses():
  subprocess.run(['bash', '-n', str(root.SCRIPT)], check=True)


@pytest.mark.skipif(shutil.which('shellcheck') is None, reason='needs shellcheck')
def test_the_script_passes_shellcheck():
  # at the default severity, as the CI job runs it
  subprocess.run(['shellcheck', str(root.SCRIPT)], check=True)


# what apply records for restore: stock AGNOS runs the dirty limits in ratio
# mode and the kernel drops a 0 written to a *_bytes key, so their ratio keys
# stand in for them
# the dirty limits by their ratio keys (stock is ratio mode), the socket caps as they are
RECORDED = ('vm.dirty_ratio', 'vm.dirty_background_ratio', 'vm.extra_free_kbytes', 'net.core.wmem_max', 'net.core.rmem_max')


def test_vm_apply_records_the_stock_values_once_and_applies_ours(tmp_path):
  proc_sys(tmp_path, STOCK)
  assert run_script(tmp_path, 'vm', 'apply').returncode == 0
  # what restore writes back, a line each
  stock = ''.join(f'{k}={STOCK[k]}\n' for k in RECORDED)
  assert record(tmp_path).read_text() == stock
  assert read_all(tmp_path, TUNED) == TUNED
  # a second apply keeps the first record, not our own values
  assert run_script(tmp_path, 'vm', 'apply').returncode == 0
  assert record(tmp_path).read_text() == stock


def test_a_key_the_last_apply_did_not_record_is_recorded_while_stock(tmp_path):
  # an update adds a key while the link is on: the record from this boot's
  # first apply lacks it, and restore must still put it back
  proc_sys(tmp_path, STOCK)
  older = [k for k in RECORDED if k != 'vm.extra_free_kbytes']
  record(tmp_path).write_text(''.join(f'{k}={STOCK[k]}\n' for k in older))
  assert run_script(tmp_path, 'vm', 'apply').returncode == 0
  assert record(tmp_path).read_text().splitlines() == [f'{k}={STOCK[k]}' for k in older] + ['vm.extra_free_kbytes=0']
  assert read_sys(tmp_path, 'vm.extra_free_kbytes') == TUNED['vm.extra_free_kbytes']
  assert run_script(tmp_path, 'vm', 'restore').returncode == 0
  assert read_sys(tmp_path, 'vm.extra_free_kbytes') == '0'


def test_vm_restore_goes_back_to_ratio_mode_and_drops_the_record(tmp_path):
  proc_sys(tmp_path, STOCK)
  assert run_script(tmp_path, 'vm', 'apply').returncode == 0
  assert run_script(tmp_path, 'vm', 'restore').returncode == 0
  # the fake /proc cannot zero the bytes keys as the kernel does
  for key in RECORDED:
    assert read_sys(tmp_path, key) == STOCK[key]
  assert not record(tmp_path).exists()


def test_bytes_that_were_set_are_put_back_as_bytes(tmp_path):
  proc_sys(tmp_path, {**STOCK, 'vm.dirty_bytes': '33554432'})
  assert run_script(tmp_path, 'vm', 'apply').returncode == 0
  assert 'vm.dirty_bytes=33554432\n' in record(tmp_path).read_text()
  assert run_script(tmp_path, 'vm', 'restore').returncode == 0
  assert read_sys(tmp_path, 'vm.dirty_bytes') == '33554432'


def test_vm_restore_without_a_record_changes_nothing(tmp_path):
  proc_sys(tmp_path, TUNED)
  assert run_script(tmp_path, 'vm', 'restore').returncode == 0
  assert read_all(tmp_path, TUNED) == TUNED


def test_restore_writes_only_vm_keys_and_numbers(tmp_path):
  # /dev/shm is anyone's to write, and restore runs as root
  proc_sys(tmp_path, TUNED)
  record(tmp_path).write_text('garbage\nkernel.core_pattern=|/bin/sh\nvm.dirty_bytes=x\nvm.dirty_background_bytes=10\n')
  result = run_script(tmp_path, 'vm', 'restore')
  assert result.returncode == 1
  assert result.stderr.count('not restoring') == 3
  assert read_sys(tmp_path, 'vm.dirty_background_bytes') == '10'
  assert read_sys(tmp_path, 'vm.dirty_bytes') == TUNED['vm.dirty_bytes']
  assert not (tmp_path / 'sys' / 'kernel').exists()
  assert not record(tmp_path).exists()


def test_vm_apply_fails_when_a_key_will_not_take(tmp_path):
  proc_sys(tmp_path, STOCK)
  last = tmp_path / 'sys' / 'vm' / 'dirty_background_bytes'
  last.chmod(0o444)
  if os.access(last, os.W_OK):
    pytest.skip('running as root')
  result = run_script(tmp_path, 'vm', 'apply')
  assert result.returncode == 1
  assert result.stderr.strip().splitlines()[-1] == \
    f'jetlink: could not set vm.dirty_background_bytes={TUNED["vm.dirty_background_bytes"]}: Permission denied'
  # the others still took
  assert read_sys(tmp_path, 'vm.dirty_bytes') == TUNED['vm.dirty_bytes']


@pytest.mark.parametrize('lever, force, value', [
  (voter, ('port', 'hold'), '1'),
  # the input current limit: 0 suspends the input
  (usb_icl, ('draw', 'off'), '0'),
])
def test_a_voter_is_forced_and_let_go(tmp_path, lever, force, value):
  lever = lever(tmp_path)
  lever.mkdir()
  assert run_script(tmp_path, *force).returncode == 0
  assert (lever / 'force_val').read_text().strip() == value
  assert (lever / 'force_active').read_text().strip() == '1'
  release = {'hold': 'off', 'off': 'on'}[force[1]]
  assert run_script(tmp_path, force[0], release).returncode == 0
  assert (lever / 'force_active').read_text().strip() == '0'
  assert (lever / 'force_val').read_text().strip() == '0'


@pytest.mark.parametrize('setting, cut', [('1', True), ('2', True), ('0', False), ('3', False), ('', False), (None, False)])
def test_boot_cuts_the_draw_while_the_link_is_on(tmp_path, setting, cut):
  # the launch script, before manager: usb and ios hold the port; off, Wi-Fi
  # (3, which never touches it) and no setting do not
  lever = usb_icl(tmp_path)
  lever.mkdir()
  path = tmp_path / 'JetlinkLink'
  if setting is not None:
    path.write_text(setting)
  assert run_script(tmp_path, 'boot', str(path)).returncode == 0
  assert (lever / 'force_active').is_file() == cut
  if cut:
    assert (lever / 'force_val').read_text().strip() == '0'


@pytest.mark.parametrize('args, made, says', [
  (('port', 'hold'), None, 'could not force'),
  (('port', 'off'), None, 'could not release'),
  # a refusal, or PD not ready, is a failed write; so is a missing class
  (('port', 'device'), None, 'did not take the host role'),
  (('port', 'source'), None, 'did not give up the source role'),
  (('port', 'reset'), None, 'could not reset USB PD'),
  (('draw', 'off'), None, 'could not force'),
  (('draw', 'on'), None, 'could not release'),
  (('udc', 'start'), None, 'could not start the USB device controller'),
  (('udc', 'stop'), None, 'could not stop the USB device controller'),
  # a kernel without the glue's knob still gets the policy engine's
  (('udc', 'apply'), pe_params, 'a600000.ssusb/usb_compliance_mode'),
])
def test_a_lever_the_kernel_lacks_fails(tmp_path, args, made, says):
  # both commas have them all, so an absence is a failure like any other
  if made is not None:
    made(tmp_path).mkdir()
  result = run_script(tmp_path, *args)
  assert result.returncode == 1
  assert says in result.stderr
  if made is not None:
    assert (made(tmp_path) / 'usb_compliance_mode').read_text().strip() == 'Y'


@pytest.mark.parametrize('args, where, name, value', [
  # the kernel turns this write into a USB PD DR_Swap
  (('port', 'device'), dual_role, 'data_role', 'device'),
  # and this one into a PR_Swap: the far end charges from the comma
  (('port', 'source'), dual_role, 'power_role', 'source'),
  (('port', 'reset'), usbpd, 'hard_reset', '1'),
  (('udc', 'start'), udc_glue, 'mode', 'peripheral'),
  (('udc', 'stop'), udc_glue, 'mode', 'none'),
])
def test_a_lever_is_one_write(tmp_path, args, where, name, value):
  where(tmp_path).mkdir()
  assert run_script(tmp_path, *args).returncode == 0
  assert (where(tmp_path) / name).read_text().strip() == value


def test_a_refusal_says_why(tmp_path):
  # the kernel's errno is what tells PD not ready from a refusal
  dual_role(tmp_path).mkdir()
  (dual_role(tmp_path) / 'power_role').mkdir()   # a write fails, as a refused one does
  result = run_script(tmp_path, 'port', 'source')
  assert result.returncode == 1
  line = result.stderr.strip().splitlines()[-1]
  assert line.startswith('jetlink: the far end did not give up the source role')
  assert line.endswith(': Is a directory'), line


def test_udc_apply_keeps_the_device_and_restore_puts_it_back(tmp_path):
  for d in (udc_glue(tmp_path), pe_params(tmp_path)):
    d.mkdir()
  for command, value, log_off in (('apply', 'Y', '1'), ('restore', 'N', '0')):
    assert run_script(tmp_path, 'udc', command).returncode == 0
    for d in (udc_glue(tmp_path), pe_params(tmp_path)):
      assert (d / 'usb_compliance_mode').read_text().strip() == value
    assert ffs_log_off(tmp_path).read_text().strip() == log_off


def test_rps_puts_the_cables_receive_work_on_core_6(tmp_path):
  mask = cable_netdev(tmp_path)
  assert run_script(tmp_path, 'rps').returncode == 0
  assert mask.read_text().strip() == '40'


def test_rps_fails_when_the_mask_does_not_read_back(tmp_path):
  # the kernel keeps only the cores online at the write: parked, core 6 reads 00
  mask = cable_netdev(tmp_path)
  mask.unlink()
  mask.symlink_to('/dev/null')
  result = run_script(tmp_path, 'rps')
  assert result.returncode == 1
  assert 'reads nothing, not 40' in result.stderr


def test_teardown_takes_our_conntrack_rules_out_and_leaves_the_rest(tmp_path):
  # the netdev's name moves between binds, so ours are found by their comment
  calls = tmp_path / 'iptables.calls'
  fake = tmp_path / 'iptables'
  fake.write_text(f"""#!/bin/sh
case "$*" in
  *" -S"*) printf '%s\\n' '-P OUTPUT ACCEPT' \\
    '-A PREROUTING -i usb1 -m comment --comment jetlink -j CT --notrack' \\
    '-A OUTPUT -o usb1 -m comment --comment jetlink -j CT --notrack' \\
    '-A OUTPUT -o wlan0 -m comment --comment someone -j CT --notrack' ;;
  *) echo "$*" >> {calls} ;;
esac
""")
  fake.chmod(0o755)
  result = run_script(tmp_path, 'teardown', iptables=fake)
  assert result.returncode == 0, result.stderr
  assert calls.read_text().splitlines() == [
    '-w -t raw -D PREROUTING -i usb1 -m comment --comment jetlink -j CT --notrack',
    '-w -t raw -D OUTPUT -o usb1 -m comment --comment jetlink -j CT --notrack',
  ]


def test_rps_without_the_cable_network_says_so(tmp_path):
  result = run_script(tmp_path, 'rps')
  assert result.returncode == 1
  assert 'no cable network interface' in result.stderr


@pytest.mark.parametrize('args', [(), ('setup',), ('--ios',), ('gadget', '--net'), ('port',), ('port', 'on'), ('udc',),
                                  ('udc', 'on'), ('vm',), ('vm', 'undo'),
                                  ('draw',), ('draw', 'cut'), ('boot',), ('boot', 'a', 'b')])
def test_anything_else_is_usage(tmp_path, args):
  result = run_script(tmp_path, *args)
  assert result.returncode == 2
  assert 'usage:' in result.stderr
