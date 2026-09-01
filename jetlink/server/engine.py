"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

TensorRT execution.

Everything is preallocated at load time - device buffers, pinned host staging
buffers and the stream - so the steady state does no allocation at all. That is
what keeps the per-frame time predictable, which matters more here than raw
throughput: the budget is 50 ms per frame and the tail is what breaks a drive.

Callers write straight into the pinned input arrays exposed by `host_input()`
to avoid a second copy on the way to the GPU.
"""
from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass

import numpy as np
import tensorrt as trt

from jetlink.server import cudart

TRT_TO_NP = {
  trt.DataType.FLOAT: np.float32,
  trt.DataType.HALF: np.float16,
  trt.DataType.INT8: np.int8,
  trt.DataType.INT32: np.int32,
  trt.DataType.INT64: np.int64,
  trt.DataType.BOOL: np.bool_,
  trt.DataType.UINT8: np.uint8,
}


@dataclass
class Binding:
  name: str
  shape: tuple[int, ...]
  dtype: np.dtype
  nbytes: int
  device_ptr: int
  host_ptr: int
  host: np.ndarray
  is_input: bool


def _np_from_ptr(ptr: int, shape, dtype) -> np.ndarray:
  """Writable numpy view over pinned host memory, so writes need no further copy.

  Goes through a raw byte buffer rather than np.ctypeslib.as_ctypes_type,
  which cannot represent float16 - the dtype most of this model uses.
  """
  dtype = np.dtype(dtype)
  nbytes = int(np.prod(shape)) * dtype.itemsize
  buf = (ctypes.c_char * nbytes).from_address(ptr)
  return np.frombuffer(buf, dtype=dtype).reshape(shape)


class TrtEngine:
  def __init__(self, plan_path: str, log_severity=trt.Logger.WARNING):
    self.logger = trt.Logger(log_severity)
    trt.init_libnvinfer_plugins(self.logger, '')
    self.runtime = trt.Runtime(self.logger)
    with open(plan_path, 'rb') as f:
      self.engine = self.runtime.deserialize_cuda_engine(f.read())
    if self.engine is None:
      raise RuntimeError(f"failed to deserialize engine {plan_path}")
    self.context = self.engine.create_execution_context()
    self.stream = cudart.stream_create()

    self.bindings: dict[str, Binding] = {}
    for i in range(self.engine.num_io_tensors):
      name = self.engine.get_tensor_name(i)
      is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
      shape = tuple(self.engine.get_tensor_shape(name))
      if any(d < 0 for d in shape):
        raise RuntimeError(f"tensor {name} has a dynamic shape {shape}; "
                           "jetlink builds fixed-shape engines")
      dtype = np.dtype(TRT_TO_NP[self.engine.get_tensor_dtype(name)])
      nbytes = int(np.prod(shape)) * dtype.itemsize
      dev = cudart.malloc(nbytes)
      host = cudart.host_alloc(nbytes)
      self.bindings[name] = Binding(name, shape, dtype, nbytes, int(dev), int(host),
                                    _np_from_ptr(int(host), shape, dtype), is_input)
      self.context.set_tensor_address(name, int(dev))

    self.inputs = {n: b for n, b in self.bindings.items() if b.is_input}
    self.outputs = {n: b for n, b in self.bindings.items() if not b.is_input}
    self.last_gpu_us = 0

  # -- introspection --------------------------------------------------------

  @property
  def input_shapes(self) -> dict[str, tuple[int, ...]]:
    return {n: b.shape for n, b in self.inputs.items()}

  @property
  def output_shapes(self) -> dict[str, tuple[int, ...]]:
    return {n: b.shape for n, b in self.outputs.items()}

  def host_input(self, name: str) -> np.ndarray:
    """Pinned array for an input. Write into it, then call run()."""
    return self.inputs[name].host

  def host_output(self, name: str) -> np.ndarray:
    return self.outputs[name].host

  # -- execution ------------------------------------------------------------

  def load_inputs(self, values: dict[str, np.ndarray]) -> None:
    for name, value in values.items():
      b = self.inputs.get(name)
      if b is None:
        raise KeyError(f"engine has no input {name!r}; has {sorted(self.inputs)}")
      if int(np.prod(value.shape)) != int(np.prod(b.shape)):
        raise ValueError(f"{name}: {value.shape} has {value.size} elements, "
                         f"engine wants {b.shape} ({int(np.prod(b.shape))})")
      # copyto casts if needed; the queues already produce the engine's dtype.
      np.copyto(b.host, value.reshape(b.shape), casting='unsafe')

  def run(self) -> dict[str, np.ndarray]:
    """Run one frame. Returns views over pinned output memory, valid until the next run."""
    t0 = time.perf_counter()
    for b in self.inputs.values():
      cudart.memcpy_h2d_async(b.device_ptr, b.host_ptr, b.nbytes, self.stream)
    if not self.context.execute_async_v3(self.stream):
      raise RuntimeError("execute_async_v3 failed")
    for b in self.outputs.values():
      cudart.memcpy_d2h_async(b.host_ptr, b.device_ptr, b.nbytes, self.stream)
    cudart.stream_sync(self.stream)
    self.last_gpu_us = int((time.perf_counter() - t0) * 1e6)
    return {n: b.host for n, b in self.outputs.items()}

  def infer(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    self.load_inputs(values)
    return self.run()

  # Deliberately no __del__. host_input() hands out numpy views over
  # cudaHostAlloc memory that hold no reference back here, so a GC-driven
  # close() would free pages another object is still writing into - a segfault
  # rather than an exception. Ownership is explicit: whoever swaps an engine
  # out closes it.
  def close(self) -> None:
    for b in self.bindings.values():
      try:
        cudart.free(b.device_ptr)
        cudart.host_free(b.host_ptr)
      except Exception:
        pass
    self.bindings.clear()
    try:
      cudart.stream_destroy(self.stream)
    except Exception:
      pass

