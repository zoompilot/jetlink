"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

What the comma imports, and that it loads nothing else.

openpilot runs jetlink from a checkout on the comma (only jetlink/ and
scripts/comma/ ship), imports the names below and patches some of them in its
tests. Each module is imported in a fresh interpreter, so one module's imports
cannot hide another's: none may load a package beyond numpy and the standard
library. tests/openpilot/test_api.py pins the signatures of the API itself.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from jetlink import protocol as P

ROOT = Path(__file__).resolve().parents[1]

# What the fork imports from each module, and the members it calls, reads or
# patches: its adapter (openpilot/sunnypilot/jetlink_adapter), which calls
# jetlink.openpilot's API, and its tests, which also run jetlink's comma side
# on the fork's own tinygrad and openpilot (fork branch jetlink-host-adapter,
# 2026-09-29). The fork runs against whatever jetlink_repo is pinned: a name
# dropped here breaks that build.
FORK_IMPORTS = {
  'jetlink.openpilot': ('API', 'MODES', 'STATES', 'Status', 'bind'),
  'jetlink.openpilot.interface': ('Keys', 'ModelFace', 'Openpilot', 'OwnerConfig', 'conformance', 'load_adapter'),
  'jetlink.openpilot.owner': ('main', 'worker'),
  'jetlink.openpilot.settings': ('FileParams', 'Settings'),
  # its tests only: the seam guard, and the comma side on the real tinygrad
  'jetlink.openpilot.joining': ('JoiningModelState',),
  'jetlink.openpilot.model_state': ('JetlinkModelState',),
  'jetlink.openpilot.warp': ('Warp', 'Warps', 'init_device', 'prepare_reset'),
  'jetlink.spec': ('ModelSpec',),
  'jetlink.queues': ('PolicyQueues',),
  'jetlink.comma': ('gadget',),
  'jetlink.comma.gadget': ('gadget_error', 'link_configured'),
  'jetlink.registry.catalog': ('fetch_catalogs',),
}
# What jetlink.openpilot calls on a client and reads off one, off a loan, and
# off the transport a client rides on: the comma's side of the link
CLIENT_CALLS = ('open_socket', 'open_borrowed_ffs', 'open_ffs', 'hello', 'ensure_engine', 'infer_begin',
                'infer_end', 'ping', 'shutdown', 'leave', 'rebind', 'close')
CLIENT_FIELDS = ('t', 'dead', 'deadline', 'last_timings', 'last_state')
LOAN_MEMBERS = ('cable', 'mount', 'udc', 'bounce', 'accept', 'closed', 'note_server', 'close')
TRANSPORT_CALLS = ('link_info',)
# and the rest of what runs there
COMMA_MODULES = tuple(dict.fromkeys((*FORK_IMPORTS, 'jetlink.protocol', 'jetlink.comma.gadget', 'jetlink.comma.lending',
                                     'jetlink.comma.port', 'jetlink.comma.root', 'jetlink.transport.ffs')))

PROBE = '''
import importlib, json, sys
before = set(sys.modules)
module = importlib.import_module(sys.argv[1])

def resolves(name):
  if hasattr(module, name):
    return True
  try:   # `from package import submodule`
    importlib.import_module(f'{sys.argv[1]}.{name}')
    return True
  except ImportError:
    return False

missing = [n for n in sys.argv[2:] if not resolves(n)]
loaded = set(sys.modules) - before
print(json.dumps({
  'missing': missing,
  'foreign': sorted({n.split('.')[0] for n in loaded} - set(sys.stdlib_module_names) - {'jetlink', 'numpy'}),
}))
'''


@pytest.mark.parametrize('module', COMMA_MODULES)
def test_a_comma_module_loads_only_what_the_comma_has(module):
  # cwd first on the path: this checkout's jetlink, whatever is installed
  run = subprocess.run([sys.executable, '-c', PROBE, module, *FORK_IMPORTS.get(module, ())],
                       cwd=ROOT, capture_output=True, text=True, check=True)
  found = json.loads(run.stdout)
  assert found['missing'] == [], f"the fork imports {found['missing']} from {module}"
  assert found['foreign'] == [], f"{module} needs {found['foreign']}; the comma has numpy and the standard library"


def test_the_client_has_what_openpilot_calls():
  from jetlink.client import JetlinkClient
  assert [n for n in CLIENT_CALLS if not callable(getattr(JetlinkClient, n, None))] == []
  client = JetlinkClient(SimpleNamespace())
  assert [n for n in CLIENT_FIELDS if not hasattr(client, n)] == []


def test_a_loan_has_what_openpilot_reads():
  import socket

  from jetlink.comma.lending import Loan
  a, b = socket.socketpair()
  try:
    loan = Loan(a, bytearray(), '/dev/ffs-jetlink', 'udc0')
    assert [n for n in LOAN_MEMBERS if not hasattr(loan, n)] == []
  finally:
    a.close()
    b.close()


def test_every_transport_says_what_it_is():
  from jetlink.transport.ffs import FfsTransport
  from jetlink.transport.tcp import TcpTransport
  for transport in (FfsTransport, TcpTransport):
    assert [n for n in TRANSPORT_CALLS if not callable(getattr(transport, n, None))] == [], transport


def test_the_gadget_presents_what_a_host_looks_for():
  # The server finds the comma by these; nothing on the comma reads protocol.py's copy.
  script = (ROOT / 'scripts' / 'comma' / 'jetlink-root.sh').read_text()
  ids = {k: int(v, 16) for k, v in re.findall(r'^(VID|PID)=(0x[0-9a-fA-F]+)', script, re.M)}
  assert ids == {'VID': P.USB_VID, 'PID': P.USB_PID}
