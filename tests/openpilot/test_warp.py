"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

When modeld's warp is trusted, and when it is not; what a frame does with it;
and the small model's reset for a fallback.

The warp is openpilot's own pickle and runs on the fork's tinygrad, and the
fork's tests run it for real (the frame path on tinygrad's CPU device,
prepare_reset on a real TinyJit). What is left here is the file on disk being
wrong, each a raise into the small-model fallback, and the plumbing, over a
tinygrad made of numpy.
"""
import pickle
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from jetlink.openpilot import warp
from tests.openpilot import fakes
from tests.openpilot.fakes import OpenpilotTest

GEOM = fakes.TICI
FRAME = 1234
SPECS = {'input_frame': ((2, FRAME), '|u1', 'QCOM'), 'M_inv': ((2, 3, 3), '<f4', 'QCOM')}


class FakeCaptured:
  def __init__(self, names, made=(2, 6, 128, 256)):
    self.expected_names = names
    self.ret = SimpleNamespace(shape=made)


class FakeJit:
  """Just enough of a TinyJit to be pickled and inspected."""

  def __init__(self, names=warp.WARP_INPUT_NAMES, **kwargs):
    self.captured = FakeCaptured(names, **kwargs) if names is not None else None


def pickled(jit=None, specs=SPECS) -> bytes:
  """A warp as openpilot pickles it."""
  return pickle.dumps({'metadata': {}, 'run': jit or FakeJit(), 'input_specs': specs})


class WarpTest(OpenpilotTest):
  def setUp(self):
    super().setUp()
    self.warps = self.parts.warps

  def write(self, geom=GEOM, body=b'pickle'):
    pkl = self.warps.path(*geom)
    pkl.parent.mkdir(parents=True, exist_ok=True)
    pkl.write_bytes(body)
    return pkl


class TestValidity(WarpTest):
  def test_a_warp_that_is_there_is_used(self):
    self.write()
    self.assertTrue(self.warps.is_cached(*GEOM))

  def test_nothing_there_is_a_miss(self):
    self.assertFalse(self.warps.is_cached(*GEOM))
    self.assertFalse(self.warps.built())

  def test_where_it_lives_is_the_adapters_to_say(self):
    # modeld's own, in the fork's tree, never inside the jetlink submodule
    self.assertEqual(self.warps.path(*GEOM), self.op.warp_path(*GEOM))

  def test_another_camera_does_not_answer_for_this_one(self):
    """A device that changed camera. The warp is baked against the frame
    geometry, so the wrong one is silently wrong rather than an error."""
    self.write(geom=fakes.MICI)
    self.assertFalse(self.warps.is_cached(*GEOM))
    self.assertFalse(self.warps.built())

  def test_another_model_input_size_does_not_answer_either(self):
    self.write(geom=(1928, 1208, 256, 128))
    self.assertFalse(self.warps.is_cached(*GEOM))

  def test_built_is_for_this_devices_camera_and_holds_for_the_process(self):
    # nothing makes one at runtime, and the checkout's is there before manager
    self.write()
    self.op.geometry = fakes.TICI
    self.assertTrue(self.warps.built())
    self.warps.path(*GEOM).unlink()
    self.assertTrue(self.warps.built())

  def test_the_geometry_is_the_adapters(self):
    self.op.geometry = fakes.MICI
    self.assertEqual(self.warps.geometry(), fakes.MICI)


class TestLoad(WarpTest):
  def test_a_miss_raises_rather_than_returning_none(self):
    """modeld's big-model load is wrapped in the fallback to the small model.
    Raising lands there; returning None would reach the car."""
    with self.assertRaisesRegex(RuntimeError, 'no warp for 1928x1208 -> 512x256'):
      self.warps.load(*GEOM)

  def test_a_pickle_that_will_not_load_raises(self):
    """The incompatible-tinygrad case: unpickling fails and modeld's big-model load falls back."""
    self.write(body=b'not a pickle at all')
    with self.assertRaises(pickle.UnpicklingError):
      self.warps.load(*GEOM)


class TestLoadValidation(WarpTest):
  """A pickle that loads is not yet a warp that can be trusted."""

  def refused(self, body, why):
    self.write(body=body)
    with self.assertRaisesRegex(RuntimeError, why):
      self.warps.load(*GEOM)

  def test_a_good_warp_loads(self):
    self.write(body=pickled())
    loaded = self.warps.load(*GEOM)
    self.assertIsInstance(loaded['run'], FakeJit)
    self.assertEqual(loaded['input_specs'], SPECS)

  def test_the_jit_jetlink_once_built_itself_is_refused(self):
    # a bare TinyJit, left by a build from before
    self.refused(pickle.dumps(FakeJit(['big_frame', 'big_tfm', 'frame', 'tfm'])), 'not a driving warp')

  def test_an_uncaptured_jit_is_refused(self):
    # pickled before TinyJit captured: loads fine, computes nothing
    self.refused(pickled(FakeJit(None)), 'computes nothing')

  def test_a_capture_under_other_names_is_refused(self):
    # captured positionally, called by keyword: JitError on the first frame of a drive
    self.refused(pickled(FakeJit([0, 1])), 'the frame loop passes')

  def test_a_warp_of_one_frame_is_refused(self):
    # dm_warp's shape: one frame a call
    one = {'input_frame': ((FRAME,), '|u1', 'QCOM'), 'M_inv': ((3, 3), '<f4', 'QCOM')}
    self.refused(pickled(specs=one), 'not two of each')

  def test_a_warp_for_another_model_input_is_refused(self):
    self.refused(pickled(FakeJit(made=(2, 6, 64, 128))), "not a 512x256 model's input")


class TestInitDevice(unittest.TestCase):
  """prepare() runs this before modeld goes realtime: tinygrad's compile pool
  is otherwise created on the warp's first call, and its handler threads then
  sit at FIFO 54 on the frame loop's core."""

  def setUp(self):
    self.pool = mock.Mock(name='get_worker_pool')
    p = mock.patch.dict(sys.modules, fakes.fake_tinygrad(get_worker_pool=self.pool))
    p.start()
    self.addCleanup(p.stop)
    self.log = fakes.RecordingLog()

  def test_the_compile_pool_is_created_with_the_device(self):
    warp.init_device(self.log)
    self.pool.assert_called_once_with()
    self.assertEqual(self.log.records, [])

  def test_a_pool_that_will_not_start_is_logged_not_raised(self):
    # An older tinygrad without the module, or PARALLEL=0, must not veto the accelerator.
    self.pool.side_effect = RuntimeError('no pool')
    warp.init_device(self.log)
    self.assertEqual(len(self.log.lines('exception')), 1)

  def test_a_device_that_will_not_come_up_is_logged_not_raised(self):
    with mock.patch.object(fakes.FakeTensor, 'realize', side_effect=RuntimeError('no gpu')):
      warp.init_device(self.log)
    self.assertTrue(self.log.has('could not bring the gpu up', 'exception'))
    self.pool.assert_called_once_with()


class FakeBuffer:
  """A tinygrad Buffer as Warp reads it: the CPU's view of its bytes."""

  def __init__(self, array):
    self.array = array

  def as_memoryview(self, allow_zero_copy=False):
    assert allow_zero_copy, 'a copy, which start() would write into for nothing'
    return memoryview(self.array.reshape(-1).view(np.uint8))


class BufferTensor(fakes.FakeTensor):
  """A FakeTensor over a buffer of its own, which is what a capture is
  handed (uop.base)."""

  def __init__(self, data=None, device=None, dtype=None):
    super().__init__(data, device, dtype)
    self.uop = SimpleNamespace(base=SimpleNamespace(buffer=FakeBuffer(self.array)))


class FakeQcomDevice:
  """A tinygrad device as Warp uses it. An allocation's record holds the
  flags the kernel kept: the request masked by `keep`, plus some of its own."""

  def __init__(self):
    self.keep = ~0
    self.allocs, self.freed = [], []
    self.synchronize = mock.Mock(name='synchronize')

  def _gpu_alloc(self, size, flags=0):
    mem = SimpleNamespace(size=size, meta=(SimpleNamespace(flags=flags & self.keep | 0x100c0000), True))
    self.allocs.append(mem)
    return mem

  def _gpu_free(self, mem):
    self.freed.append(mem)


class FakeOutput:
  """The warp JIT's output Buffer, write-combined as tinygrad allocates it."""

  def __init__(self, device):
    self.device, self.nbytes = device, 64
    self._storage = SimpleNamespace(meta=(SimpleNamespace(flags=0x100c0000), True))
    self.bytes = bytearray(64)

  def deallocate(self):
    self._storage = None

  def allocate(self, opaque=None):
    assert self._storage is None, "can't allocate already allocated buffer"
    self._storage = opaque
    return self

  def as_memoryview(self, allow_zero_copy=False):
    assert allow_zero_copy, 'a copy, not the bytes the next frame writes'
    return memoryview(self.bytes)


class FakeCapture:
  """A captured warp: what went in through TinyJit (each call with the
  output's allocation as it found it), and what replays after, with what the
  input buffers held then."""

  def __init__(self, out):
    self.out = out
    self.ret = SimpleNamespace(uop=SimpleNamespace(base=SimpleNamespace(buffer=out)))
    self.calls, self.replays = [], []

  def jit(self, **kwargs):
    self.calls.append((kwargs, self.out._storage))
    return self.ret

  def __call__(self, inputs, var_vals):
    self.replays.append((inputs, var_vals, [b.buffer.array.copy() for b in inputs]))
    return self.ret


class TestWarp(unittest.TestCase):
  """The warp as the frame loop runs it: its output in memory the CPU reads
  through its cache, its inputs checked once and replayed after, and the
  camera buffers copied in."""

  def setUp(self):
    self.devices = {'QCOM': FakeQcomDevice(), 'CPU': FakeQcomDevice()}
    modules = fakes.fake_tinygrad()
    modules['tinygrad.device'].Device = self.devices
    modules['tinygrad.tensor'].Tensor = BufferTensor
    p = mock.patch.dict(sys.modules, modules)
    p.start()
    self.addCleanup(p.stop)

  def make(self, device='QCOM', buffer=FRAME):
    capture = FakeCapture(FakeOutput(device))
    jit = mock.Mock(side_effect=capture.jit)
    jit.captured = capture
    specs = {name: (shape, dtype, device) for name, (shape, dtype, _) in SPECS.items()}
    return warp.Warp({'run': jit, 'input_specs': specs}, buffer), capture

  def test_the_output_is_coherent_before_the_first_call(self):
    # the first call links the JIT to its buffers' addresses
    w, capture = self.make()
    qcom = self.devices['QCOM']
    self.assertEqual(len(qcom.allocs), 1)
    self.assertTrue(all(found is qcom.allocs[0] for _, found in capture.calls))
    self.assertEqual(qcom.allocs[0].meta[0].flags & warp.COHERENT_WRITEBACK, warp.COHERENT_WRITEBACK)
    self.assertEqual(w.output.nbytes, 64)
    self.assertIs(w.wait, qcom.synchronize)

  def test_memory_the_kernel_would_not_make_coherent_is_refused(self):
    # write-back memory that is not coherent read stale frames, 396 of 400
    for withheld in (1 << 31, 1 << 26):   # coherency; write-back
      self.devices['QCOM'].keep = ~withheld
      with self.assertRaisesRegex(RuntimeError, 'IO-coherent'):
        self.make()
      self.assertIs(self.devices['QCOM'].freed[-1], self.devices['QCOM'].allocs[-1])

  def test_a_cpu_warp_keeps_its_output(self):
    # the fork's frame-path test runs the real warp on tinygrad's CPU device
    self.make('CPU')
    self.assertEqual(self.devices['CPU'].allocs, [])

  def test_it_is_warmed_through_tinygrad_on_zero_inputs_as_the_pickle_says(self):
    _, capture = self.make()
    self.assertEqual(len(capture.calls), 2)
    kwargs, _ = capture.calls[0]
    self.assertEqual(sorted(kwargs), warp.WARP_INPUT_NAMES)
    for name, (shape, dtype, device) in SPECS.items():
      got = kwargs[name]
      self.assertEqual((got.array.shape, got.array.dtype, got.device), (shape, np.dtype(dtype), device))
      self.assertFalse(got.array.any())
    self.assertEqual(capture.replays, [])
    self.devices['QCOM'].synchronize.assert_called_once_with()

  def test_a_frame_copies_the_cameras_in_and_replays_the_capture(self):
    # a camera buffer is longer than what the warp reads of it
    w, capture = self.make(buffer=FRAME + 100)
    rng = np.random.default_rng(0)
    frame, big_frame = (rng.integers(0, 256, FRAME + 100, dtype=np.uint8) for _ in range(2))
    tfm, big_tfm = np.eye(3) * 2, np.eye(3) * 3
    w.start(frame.ctypes.data, big_frame.ctypes.data, tfm, big_tfm)
    (inputs, var_vals, held), = capture.replays
    self.assertEqual(var_vals, {})
    # the warm-up's buffers, in the capture's order: sorted names
    kwargs, _ = capture.calls[0]
    self.assertEqual([b.buffer for b in inputs], [kwargs[name].uop.base.buffer for name in warp.WARP_INPUT_NAMES])
    got_tfm, got_frames = held
    np.testing.assert_array_equal(got_frames, np.stack([frame[:FRAME], big_frame[:FRAME]]))
    np.testing.assert_array_equal(got_tfm, np.stack([tfm, big_tfm]))
    self.assertEqual(len(capture.calls), 2, 'no call through TinyJit after the warm-up')

  def test_every_frame_replays_the_same_buffers(self):
    w, capture = self.make()
    frame = np.zeros(FRAME, np.uint8)
    for _ in range(3):
      w.start(frame.ctypes.data, frame.ctypes.data, np.eye(3), np.eye(3))
    first, *rest = [inputs for inputs, _, _ in capture.replays]
    for inputs in rest:
      self.assertTrue(all(a is b for a, b in zip(first, inputs, strict=True)))

  def test_a_warp_that_would_read_past_the_camera_buffer_is_refused(self):
    with self.assertRaisesRegex(RuntimeError, 'past a 1000 byte camera buffer'):
      self.make(buffer=1000)


class TestPrepareReset(unittest.TestCase):
  """The small model's history is zeroed in place on a fallback, as modeld's
  warmup leaves it: the JIT's buffers keep their identities and nothing is
  compiled on the failure frame. The fork runs this against a real TinyJit."""

  def setUp(self):
    self.modules = fakes.fake_tinygrad()
    p = mock.patch.dict(sys.modules, self.modules)
    p.start()
    self.addCleanup(p.stop)

  def queue(self, device='CPU'):
    return fakes.FakeTensor(np.ones((4, 8), np.float32), device=device)

  def test_stock_modelds_model(self):
    # the recurrent state is zeroed, the views of the packed upload through
    # it, and the warp's output is not history
    packed = np.ones(64, np.uint8)
    model = SimpleNamespace(input_queues={'state_img_q': self.queue(), 'state_feat_q': self.queue(),
                                          'desire': self.queue(), 'new_img': self.queue()},
                            state_pairs={'state_img_q': 'next_state_img_q', 'state_feat_q': 'next_state_feat_q'},
                            packed_input=packed, prev_desire=np.ones(8))
    warp.prepare_reset(model)()
    for name in ('state_img_q', 'state_feat_q'):
      np.testing.assert_array_equal(model.input_queues[name].array, 0)
    for name in ('desire', 'new_img'):
      np.testing.assert_array_equal(model.input_queues[name].array, 1)
    np.testing.assert_array_equal(packed, 0)
    np.testing.assert_array_equal(model.prev_desire, 0)

  def test_a_modeld_v2_native_bundle(self):
    packed = np.ones(64, np.uint8)
    adapter = SimpleNamespace(is_native=True, input_queues={'state_img_q': self.queue(), 'desire': self.queue()},
                              state_pairs={'state_img_q': 'next_state_img_q'}, packed_input=packed)
    model = SimpleNamespace(adapter=adapter, prev_desire=np.ones(8))
    warp.prepare_reset(model)()
    np.testing.assert_array_equal(adapter.input_queues['state_img_q'].array, 0)
    np.testing.assert_array_equal(adapter.input_queues['desire'].array, 1)
    np.testing.assert_array_equal(packed, 0)
    np.testing.assert_array_equal(model.prev_desire, 0)

  def test_a_modeld_v2_legacy_bundle_and_its_npy_tensor(self):
    # numpy inputs on the adapter; the NPY tensor is not a queue, and its
    # numpy views are what zero it
    packed = np.ones(16, np.float32)
    npy = self.queue(device='NPY')
    adapter = SimpleNamespace(is_native=False, input_queues={'img_q': self.queue(), 'packed_npy_inputs': npy},
                              numpy_inputs={'desire': packed[:8], 'rest': packed[8:]})
    model = SimpleNamespace(adapter=adapter, prev_desire=np.ones(8))
    with mock.patch.object(npy, 'assign', side_effect=AssertionError('an NPY tensor is not a queue')):
      warp.prepare_reset(model)()
    np.testing.assert_array_equal(adapter.input_queues['img_q'].array, 0)
    np.testing.assert_array_equal(packed, 0)
    np.testing.assert_array_equal(model.prev_desire, 0)

  def test_a_model_with_no_state_on_the_gpu_captures_nothing(self):
    # a TinyJit with nothing to run raises at its capture
    self.modules['tinygrad'].TinyJit = lambda fn: mock.Mock(side_effect=AssertionError('nothing to capture'))
    packed = np.ones(8, np.uint8)
    adapter = SimpleNamespace(is_native=True, input_queues={}, state_pairs={}, packed_input=packed)
    model = SimpleNamespace(adapter=adapter, prev_desire=np.ones(8))
    warp.prepare_reset(model)()
    np.testing.assert_array_equal(packed, 0)
    np.testing.assert_array_equal(model.prev_desire, 0)


if __name__ == "__main__":
  unittest.main()
