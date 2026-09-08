"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.
"""
from __future__ import annotations

from pathlib import Path

from jetlink.server import platform as P


def test_jetson_detection_reads_the_device_tree(tmp_path, monkeypatch):
  monkeypatch.setattr(P, '_TEGRA_SIGNS', (str(tmp_path / 'compatible'),))
  monkeypatch.setattr(P, '_TEGRA_FILES', (str(tmp_path / 'nope'),))
  assert not P.is_jetson()
  (tmp_path / 'compatible').write_bytes(b'nvidia,p3768-0000+p3767-0005\0nvidia,tegra234\0')
  assert P.is_jetson()


def test_jetson_detection_accepts_the_release_file(tmp_path, monkeypatch):
  monkeypatch.setattr(P, '_TEGRA_SIGNS', ())
  monkeypatch.setattr(P, '_TEGRA_FILES', (str(tmp_path / 'nv_tegra_release'),))
  assert not P.is_jetson()
  (tmp_path / 'nv_tegra_release').write_text('# R36 (release), REVISION: 4.0\n')
  assert P.is_jetson()


def test_cache_dir_env_wins_then_the_jetson_mount_then_the_user_cache(tmp_path, monkeypatch):
  monkeypatch.setenv('JETLINK_CACHE', str(tmp_path / 'env'))
  assert P.default_cache_dir() == tmp_path / 'env'
  monkeypatch.delenv('JETLINK_CACHE')
  monkeypatch.setattr(P, 'JETSON_CACHE', tmp_path / 'mnt')
  monkeypatch.setattr(P, 'is_jetson', lambda: False)
  assert P.default_cache_dir() != tmp_path / 'mnt'
  (tmp_path / 'mnt').mkdir()
  assert P.default_cache_dir() == tmp_path / 'mnt'
  (tmp_path / 'mnt').rmdir()
  monkeypatch.setattr(P, 'is_jetson', lambda: True)
  assert P.default_cache_dir() == tmp_path / 'mnt'


def test_user_cache_dir_is_under_home(monkeypatch):
  monkeypatch.delenv('JETLINK_CACHE', raising=False)
  monkeypatch.setattr(P, 'JETSON_CACHE', Path('/nonexistent/jetlink'))
  monkeypatch.setattr(P, 'is_jetson', lambda: False)
  d = P.default_cache_dir()
  assert d.name == 'jetlink'
  assert str(d).startswith(str(Path.home())) or 'XDG' in str(d) or 'LOCALAPPDATA' in str(d)


def test_available_bytes_never_raises_and_is_not_negative():
  assert P.available_bytes() >= 0


def test_gpu_name_is_a_string_whatever_the_host():
  assert isinstance(P.gpu_name(), str) and P.gpu_name()


def test_can_suspend_follows_sys_power(monkeypatch, tmp_path):
  assert isinstance(P.can_suspend(), bool)
