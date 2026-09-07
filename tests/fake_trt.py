"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Just enough of tensorrt and cuda-python to import the server off a Jetson.

The server modules touch these at import time (dtype tables, a base class for
the build progress monitor). Install with install_stubs() before importing
anything under jetlink.server.
"""
from __future__ import annotations

import sys
import types


def _enum(**kw):
  ns = types.SimpleNamespace(**kw)
  return ns


def install_stubs() -> None:
  if 'tensorrt' in sys.modules:
    return

  trt = types.ModuleType('tensorrt')
  trt.__version__ = '10.3.0-stub'

  class Logger:
    WARNING = 1
    INFO = 2

    def __init__(self, severity=WARNING):
      self.severity = severity

  class IProgressMonitor:
    def __init__(self):
      pass

  trt.Logger = Logger
  trt.IProgressMonitor = IProgressMonitor
  trt.DataType = _enum(FLOAT='f4', HALF='f2', INT8='i1', INT32='i4',
                       INT64='i8', BOOL='b1', UINT8='u1')
  trt.TensorIOMode = _enum(INPUT='in', OUTPUT='out')
  trt.MemoryPoolType = _enum(WORKSPACE='ws')
  trt.BuilderFlag = _enum(FP16='fp16')
  trt.init_libnvinfer_plugins = lambda *a, **k: True
  trt.Runtime = lambda *a, **k: None
  trt.Builder = lambda *a, **k: None
  trt.OnnxParser = lambda *a, **k: None
  sys.modules['tensorrt'] = trt

  # cuda-python: jetlink.server.cudart reads a few enum members at import.
  runtime = types.ModuleType('cuda.bindings.runtime')
  runtime.cudaError_t = _enum(cudaSuccess=0)
  runtime.cudaMemcpyKind = _enum(cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2)
  for name in ('cudaMalloc', 'cudaFree', 'cudaHostAlloc', 'cudaFreeHost',
               'cudaStreamCreate', 'cudaStreamDestroy', 'cudaStreamSynchronize',
               'cudaMemcpyAsync', 'cudaMemGetInfo', 'cudaGetDeviceProperties',
               'cudaGetErrorName', 'cudaGetErrorString'):
    setattr(runtime, name, lambda *a, **k: (0, 0))

  bindings = types.ModuleType('cuda.bindings')
  bindings.runtime = runtime
  cuda = types.ModuleType('cuda')
  cuda.bindings = bindings
  sys.modules.setdefault('cuda', cuda)
  sys.modules.setdefault('cuda.bindings', bindings)
  sys.modules.setdefault('cuda.bindings.runtime', runtime)
