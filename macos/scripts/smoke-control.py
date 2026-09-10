#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Prove a Python runtime can actually run the server and speak the control channel.

CI runs this against the interpreter embedded in Jetlink.app, under the stripped
environment from 01-contracts.md section 9, so it fails when the bundle is
missing a module, when the relocated prefix cannot find its stdlib, or when the
control channel regresses. It starts the server on a free TCP port with a
control socket in a temporary directory, asserts the six on-connect events
arrive in order, sends `status`, sends `shutdown`, and requires exit code 0.

No comma and no model are involved: the point is the runtime and the channel,
which is the part of the bundle nothing else checks.

  macos/scripts/smoke-control.py                     the app's embedded interpreter
  macos/scripts/smoke-control.py --python .venv/bin/python   a development venv

A development venv usually needs the repo on PYTHONPATH; this script passes
PYTHONPATH through when it is set, and drops it otherwise so a bundle test is
never fooled by the shell it ran from.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_PYTHON = REPO_ROOT / 'macos' / 'build' / 'Jetlink.app' / 'Contents' / 'Resources' / 'python' / 'bin' / 'python3'

# 01-contracts.md section 4.1: this is the order, and a client renders its first
# screen from it without sending a command.
ON_CONNECT = ('hello', 'server', 'link', 'engine', 'inventory', 'catalog')

CONNECT_TIMEOUT = 120.0   # a cold backend selection can be slow; ort on CoreML is minutes
REPLY_TIMEOUT = 30.0
EXIT_TIMEOUT = 15.0
STDERR_KEEP = 40


class SmokeError(Exception):
  pass


def free_port() -> int:
  """A port nothing holds. Racy in principle, fine for a serving loop that
  binds a second later, and better than a fixed port on a shared runner."""
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(('127.0.0.1', 0))
    return int(sock.getsockname()[1])


def clean_env() -> dict[str, str]:
  """The environment the app builds for the server, never inherited wholesale."""
  env = {
    'PATH': '/usr/bin:/bin:/usr/sbin:/sbin',
    'HOME': os.environ.get('HOME', str(Path.home())),
    'TMPDIR': os.environ.get('TMPDIR', '/tmp'),
    'LANG': 'en_US.UTF-8',
    'PYTHONNOUSERSITE': '1',
    'PYTHONDONTWRITEBYTECODE': '1',
    'PYTHONUNBUFFERED': '1',
    'PYTHONIOENCODING': 'utf-8',
  }
  user = os.environ.get('USER')
  if user:
    env['USER'] = user
  # The one deliberate leak: a development venv reaches the jetlink package
  # through it. The embedded runtime has the package installed and is run
  # without it. Entries are made absolute because the server runs in the cache
  # directory, so a `PYTHONPATH=.` from the repo root would point at the cache.
  pythonpath = os.environ.get('PYTHONPATH')
  if pythonpath:
    env['PYTHONPATH'] = os.pathsep.join(str(Path(part).absolute()) for part in pythonpath.split(os.pathsep) if part)
  return env


class Reader:
  """JSON lines off the control socket, each with its own deadline.

  A buffer of its own rather than socket.makefile: a timeout in the middle of a
  buffered readline leaves the file object holding a partial line it will not
  give back, and every read here is on a deadline.
  """

  def __init__(self, sock: socket.socket):
    self._sock = sock
    self._buf = b''

  def event(self, deadline: float) -> dict:
    while True:
      line, sep, rest = self._buf.partition(b'\n')
      if sep:
        self._buf = rest
        text = line.decode('utf-8').strip()
        if not text:
          continue
        try:
          return json.loads(text)
        except ValueError as e:
          raise SmokeError(f'the server sent a line that is not JSON: {text[:200]}') from e
      timeout = deadline - time.monotonic()
      if timeout <= 0:
        raise SmokeError('timed out waiting for an event')
      self._sock.settimeout(timeout)
      try:
        chunk = self._sock.recv(65536)
      except OSError as e:
        # A timed out recv raises socket.timeout, which is TimeoutError only
        # from Python 3.10 on; this script may be run by whatever python3 the
        # runner has, so catch the base class and tell them apart by the clock.
        if time.monotonic() >= deadline:
          raise SmokeError('timed out waiting for an event') from e
        raise SmokeError(f'the control channel failed: {e}') from e
      if not chunk:
        raise SmokeError('the control channel closed early')
      self._buf += chunk

  def reply(self, wanted_id: int, deadline: float) -> dict:
    """The reply to one command, skipping the events that may arrive first."""
    while True:
      event = self.event(deadline)
      if event.get('event') == 'reply' and event.get('id') == wanted_id:
        return event


def send(sock: socket.socket, message: dict) -> None:
  sock.sendall((json.dumps(message, separators=(',', ':')) + '\n').encode('utf-8'))


def pump_stderr(proc: subprocess.Popen, tail: deque, echo: bool) -> threading.Thread:
  def run() -> None:
    assert proc.stderr is not None
    for raw in proc.stderr:
      line = raw.decode('utf-8', 'replace').rstrip('\n')
      tail.append(line)
      if echo:
        print(f'    server: {line}', file=sys.stderr)

  thread = threading.Thread(target=run, daemon=True, name='smoke-stderr')
  thread.start()
  return thread


def connect(path: Path, proc: subprocess.Popen, deadline: float) -> socket.socket:
  while True:
    if proc.poll() is not None:
      raise SmokeError(f'the server exited with {proc.returncode} before the control socket appeared')
    if time.monotonic() > deadline:
      raise SmokeError(f'no control socket at {path} after {CONNECT_TIMEOUT:.0f} s')
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
      sock.connect(str(path))
    except OSError:
      sock.close()
      time.sleep(0.25)
      continue
    return sock


def smoke(python: Path, backend: str, device: str | None, echo: bool) -> None:
  workdir = Path(tempfile.mkdtemp(prefix='jetlink-smoke-'))
  cache = workdir / 'cache'
  cache.mkdir()
  # AF_UNIX paths are limited to 104 bytes on macOS, so the name stays short.
  control = workdir / 'c.sock'
  port = free_port()

  argv = [
    str(python), '-m', 'jetlink.server.main',
    '--backend', backend,
    '--transport', 'tcp', '--host', '127.0.0.1', '--port', str(port),
    '--cache', str(cache),
    '--control-socket', str(control),
    '--parent-pid', str(os.getpid()),
    '--log-level', 'INFO',
  ]
  if device:
    argv += ['--device', device]

  print(f'==> {python}')
  print(f'    backend {backend}{" device " + device if device else ""}, port {port}, cache {cache}')
  tail: deque = deque(maxlen=STDERR_KEEP)
  proc = subprocess.Popen(argv, cwd=str(cache), env=clean_env(),
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
  pump_stderr(proc, tail, echo)
  sock = None
  reader = None
  try:
    sock = connect(control, proc, time.monotonic() + CONNECT_TIMEOUT)
    reader = Reader(sock)

    print('==> on connect')
    deadline = time.monotonic() + REPLY_TIMEOUT
    events = [reader.event(deadline) for _ in range(len(ON_CONNECT))]
    names = tuple(e.get('event') for e in events)
    if names != ON_CONNECT:
      raise SmokeError(f'on connect the server sent {names}, expected {ON_CONNECT}')
    hello = events[0]
    if hello.get('protocol') != 1:
      raise SmokeError(f'protocol {hello.get("protocol")}, expected 1')
    if hello.get('transport') != 'tcp' or hello.get('port') != port:
      raise SmokeError(f'hello says transport {hello.get("transport")} port {hello.get("port")}, expected tcp {port}')
    for event in events:
      if not isinstance(event.get('t'), (int, float)):
        raise SmokeError(f'the {event.get("event")} event has no timestamp')
    print(f'    {", ".join(names)}')
    print(f'    jetlink {hello.get("version")} on python {hello.get("python")} ({hello.get("platform")}), pid {hello.get("pid")}')
    server = events[1]
    print(f'    backend {server.get("backend")} {server.get("runtime_version")} on {server.get("device")}')
    print(f'    link {events[2].get("state")}, engine {events[3].get("state")}')

    print('==> status')
    send(sock, {'id': 1, 'cmd': 'status'})
    reply = reader.reply(1, time.monotonic() + REPLY_TIMEOUT)
    if not reply.get('ok'):
      raise SmokeError(f'status failed: {reply.get("error")}')
    print('    ok')

    print('==> shutdown')
    send(sock, {'id': 2, 'cmd': 'shutdown'})
    reply = reader.reply(2, time.monotonic() + REPLY_TIMEOUT)
    if not reply.get('ok'):
      raise SmokeError(f'shutdown failed: {reply.get("error")}')
    print('    ok')

    try:
      code = proc.wait(timeout=EXIT_TIMEOUT)
    except subprocess.TimeoutExpired as e:
      raise SmokeError(f'the server was still running {EXIT_TIMEOUT:.0f} s after shutdown') from e
    if code != 0:
      raise SmokeError(f'the server exited with {code}, expected 0')
    print(f'    the server exited with {code}')
  except Exception:
    if tail:
      print('--- the last server output ---', file=sys.stderr)
      for line in tail:
        print(line, file=sys.stderr)
    raise
  finally:
    if sock is not None:
      with contextlib.suppress(OSError):
        sock.close()
    if proc.poll() is None:
      proc.kill()
      with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=5)
    shutil.rmtree(workdir, ignore_errors=True)


def main(argv=None) -> int:
  p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[6],
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--python', default=None, metavar='PATH',
                 help='the interpreter to run the server with (default: the app bundle\'s)')
  p.add_argument('--backend', default='ort',
                 help='which backend to bring up. ort on the CPU is the default: it needs no '
                      'GPU and it spawns the onnxruntime worker, which is the part of the '
                      'bundle most likely to break. --backend tinygrad is the quick alternative')
  p.add_argument('--device', default='cpu',
                 help='backend specific device; empty means the backend chooses')
  p.add_argument('--quiet', action='store_true', help='do not echo the server log')
  args = p.parse_args(argv)
  # The server log we echo goes to stderr, which is unbuffered; without this the
  # two streams arrive in the wrong order in a CI log.
  sys.stdout.reconfigure(line_buffering=True)

  # absolute(), not resolve(): a venv interpreter is a symlink to the base
  # installation and following it loses the venv's site-packages.
  python = Path(args.python).absolute() if args.python else APP_PYTHON
  if not python.exists():
    print(f'error: no interpreter at {python}', file=sys.stderr)
    if args.python is None:
      print('       run: make -C macos app', file=sys.stderr)
    return 1

  try:
    smoke(python, args.backend, args.device or None, echo=not args.quiet)
  except SmokeError as e:
    print(f'error: {e}', file=sys.stderr)
    return 1
  print('==> the control channel smoke test passed')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
