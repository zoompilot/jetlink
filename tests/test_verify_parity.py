"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The parity gate's statistics, on synthetic heads.

Shapes follow Cinque Terre's spec: plan is 990 values, 33 rows by 15 columns for
mu and again for std; euler is 3 and 3; lead_prob is three logits. The noise is
float16 sized, ~0.01 absolute. The gate must ride through that on columns too
small or too flat to correlate, and still fail a column that is wired wrong.
"""
import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_parity.py'
spec_ = importlib.util.spec_from_file_location('verify_parity', SCRIPT)
vp = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(vp)

PLAN = slice(0, 990)
EULER = slice(990, 996)
LEAD_PROB = slice(996, 999)
SPEC = SimpleNamespace(output_slices={'plan': PLAN, 'wide_from_device_euler': EULER, 'lead_prob': LEAD_PROB})
N_FRAMES = 20


# Scale of each plan column, from the model's own predicted std: metres, m/s,
# radians and rad/s do not share a magnitude, which is why columns are checked alone.
PLAN_SCALE = np.array([150, 1, 0.05, 15, 0.05, 0.05, 0.3, 0.01, 0.01, 0.0006, 0.005, 0.01, 0.004, 0.006, 0.009],
                      np.float32)


def reference_frames(seed=0):
  rng = np.random.default_rng(seed)
  frames = []
  t = np.linspace(0, 10, 33, dtype=np.float32)
  for i in range(N_FRAMES):
    out = np.zeros(999, np.float32)
    mu = np.empty((33, 15), np.float32)
    for j in range(15):
      mu[:, j] = PLAN_SCALE[j] * (np.sin(0.3 * t + 0.7 * i + j) + 0.3 * i)
    # ay barely moves within one frame and drifts between frames: on one frame
    # its spread is under the noise, pooled it is not.
    mu[:, 7] = 0.02 * (i / N_FRAMES) + 0.0005 * np.sin(t)
    std = -2.0 + 0.25 * rng.standard_normal((33, 15)).astype(np.float32)
    out[PLAN] = np.concatenate([mu.reshape(-1), std.reshape(-1)])
    # roll is pinned by the hardware and the net learned ~0; pitch and yaw are real
    out[EULER] = [1e-6 * rng.standard_normal(), 0.01 * i, -0.02 * i,
                  -7.1 + 0.3 * np.sin(i), -3.7 + 0.3 * np.cos(i), -3.2 + 0.3 * np.sin(2 * i)]
    out[LEAD_PROB] = [-2.5 + 0.5 * i, -2.9 + 0.3 * i, -2.3 - 0.2 * i]
    frames.append(out)
  return frames


def fp16_noisy(frames, seed=1):
  """What the other float16 implementation returns: every value off by a tenth
  of a percent of its own magnitude, then rounded to float16.

  Proportional, not uniform: the noise is the final layer's accumulation order,
  so an output near zero carries noise near zero (roll, 1e-6 rad, off by 1e-7).
  """
  rng = np.random.default_rng(seed)
  out = []
  for f in frames:
    rel = np.full(f.shape, 0.001, np.float32)
    # ay disagreed by 5% of its magnitude on the real capture, ten times the
    # rest of the head; 2% here is enough to sink a per-frame correlation.
    rel[PLAN][:495].reshape(33, 15)[:, 7] = 0.02
    noisy = f * (1 + rel * rng.standard_normal(f.shape).astype(np.float32))
    out.append(noisy.astype(np.float16).astype(np.float32))
  return out


def gate(links, refs):
  with redirect_stdout(io.StringIO()) as buf:
    passed = vp.report_slices(SPEC, links, refs)
  return passed, buf.getvalue()


def test_float16_noise_passes_pooled():
  refs = reference_frames()
  links = fp16_noisy(refs)
  passed, text = gate(links, refs)
  assert all(passed.values()), text
  assert '(1 flat)' in text, text


def test_why_pooling_the_per_frame_column_would_have_failed():
  # ay spreads 0.0005 across one frame against noise of 0.0002: correlation on
  # one frame is meaningless there, and it is what the gate used to demand.
  refs = reference_frames()
  links = fp16_noisy(refs)
  per_frame = []
  for link, ref in zip(links, refs, strict=True):
    ca, cb = vp.columns('plan', link[PLAN]), vp.columns('plan', ref[PLAN])
    per_frame.append(vp._corr(ca['mu[7]'], cb['mu[7]']))
  assert min(per_frame) < vp.MIN_CORR
  passed, text = gate(links, refs)
  assert passed['plan'], text


def test_three_logits_are_not_correlated_per_frame():
  refs = reference_frames()
  links = fp16_noisy(refs)
  per_frame = [vp._corr(link[LEAD_PROB], ref[LEAD_PROB]) for link, ref in zip(links, refs, strict=True)]
  assert min(per_frame) < 1.0
  assert LEAD_PROB.stop - LEAD_PROB.start < vp.MIN_SAMPLES
  passed, _ = gate(links, refs)
  assert passed['lead_prob']


def test_swapped_columns_fail():
  refs = reference_frames()
  links = []
  for f in fp16_noisy(refs):
    f = f.copy()
    mu = f[PLAN][:495].reshape(33, 15)
    mu[:, [0, 1]] = mu[:, [1, 0]]
    links.append(f)
  passed, text = gate(links, refs)
  assert not passed['plan'], text
  assert 'mu[0]' in text or 'mu[1]' in text


def test_off_by_one_slice_fails():
  refs = reference_frames()
  links = []
  for f in fp16_noisy(refs):
    f = f.copy()
    f[PLAN.start:PLAN.stop - 1] = f[PLAN.start + 1:PLAN.stop].copy()
    links.append(f)
  passed, _ = gate(links, refs)
  assert not passed['plan']


def test_flat_column_with_a_real_error_fails():
  refs = reference_frames()
  links = []
  for f in fp16_noisy(refs):
    f = f.copy()
    f[EULER.start] += 0.05   # roll off by 0.05 rad on a column that should be ~0
    links.append(f)
  passed, text = gate(links, refs)
  assert not passed['wide_from_device_euler'], text
  assert 'flat' in text


def test_negated_small_slice_fails_pooled():
  refs = reference_frames()
  links = []
  for f in fp16_noisy(refs):
    f = f.copy()
    f[LEAD_PROB] = -f[LEAD_PROB]
    links.append(f)
  passed, _ = gate(links, refs)
  assert not passed['lead_prob']


def test_single_frame_still_works_for_verify_engine():
  refs = reference_frames()
  links = fp16_noisy(refs)
  passed, text = gate(links[0], refs[0])
  assert set(passed) == set(SPEC.output_slices)
  # one value per column: reported as not gated, never correlated
  assert f'6 under {vp.MIN_SAMPLES} samples, not gated' in text, text
  assert 'no column gated' in text


def test_too_few_frames_do_not_gate_thin_columns():
  refs = reference_frames()
  links = []
  for f in fp16_noisy(refs):
    f = f.copy()
    f[EULER.start + 1] = -f[EULER.start + 1]   # a wrong pitch would be caught with enough frames...
    links.append(f)
  passed, _ = gate(links[:4], refs[:4])
  assert passed['wide_from_device_euler']       # ...but four samples are not a verdict either way
  passed, text = gate(links, refs)
  assert not passed['wide_from_device_euler'], text
