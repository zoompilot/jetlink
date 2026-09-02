"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

TensorRT's ONNX parser rejects UINT8 graph inputs outright, so the head of the
model has to be retyped before it will build. Exporters disagree about where
the cast goes, and the patcher has to cope with either.
"""
from __future__ import annotations

import pytest

onnx = pytest.importorskip('onnx')

from onnx import TensorProto, helper  # noqa: E402

from jetlink.onnx_patch import (  # noqa: E402
  TINYGRAD_DOMAIN,
  needs_patch,
  patch_uint8_inputs,
  strip_tinygrad_ops,
)

SHAPE = [1, 12, 128, 256]


def _image_inputs():
  return [helper.make_tensor_value_info(n, TensorProto.UINT8, SHAPE) for n in ('img', 'big_img')]


def _model(nodes, value_info=()):
  graph = helper.make_graph(nodes, 'test', _image_inputs(),
                            [helper.make_tensor_value_info('out', TensorProto.FLOAT16, SHAPE)],
                            value_info=list(value_info))
  return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 20)])


def _cast_after_concat():
  """comma's shape: join the uint8 inputs, then cast the result."""
  return _model([
    helper.make_node('Concat', ['img', 'big_img'], ['cat'], axis=1),
    helper.make_node('Cast', ['cat'], ['cast_out'], to=TensorProto.FLOAT16),
    helper.make_node('Identity', ['cast_out'], ['out']),
  ], value_info=[helper.make_tensor_value_info('cat', TensorProto.UINT8, SHAPE)])


def _cast_per_input():
  """sunnypilot's shape: cast each input, then join."""
  return _model([
    helper.make_node('Cast', ['img'], ['a'], to=TensorProto.FLOAT16),
    helper.make_node('Cast', ['big_img'], ['b'], to=TensorProto.FLOAT16),
    helper.make_node('Concat', ['a', 'b'], ['cat'], axis=1),
    helper.make_node('Identity', ['cat'], ['out']),
  ])


@pytest.mark.parametrize(("name", "build"), [
  ('cast after concat', _cast_after_concat),
  ('cast per input', _cast_per_input),
])
class TestBothGraphShapes:
  def test_is_recognised_as_needing_a_patch(self, name, build):
    assert needs_patch(build())

  def test_inputs_come_out_fp16(self, name, build):
    g = patch_uint8_inputs(build()).graph
    assert all(vi.type.tensor_type.elem_type == TensorProto.FLOAT16 for vi in g.input)

  def test_the_casts_are_gone(self, name, build):
    g = patch_uint8_inputs(build()).graph
    assert not [n for n in g.node if n.op_type == 'Cast']

  def test_nothing_still_reads_a_dropped_output(self, name, build):
    g = patch_uint8_inputs(build()).graph
    produced = {o for n in g.node for o in n.output}
    inputs = {vi.name for vi in g.input}
    for n in g.node:
      for i in n.input:
        assert i in produced or i in inputs, f"{n.op_type} reads dangling {i!r}"

  def test_the_result_is_a_valid_model(self, name, build):
    # A stale uint8 value_info is what onnxruntime refuses to load, and the
    # checker is the cheapest thing that notices.
    onnx.checker.check_model(patch_uint8_inputs(build()))

  def test_patching_twice_is_refused_rather_than_silent(self, name, build):
    once = patch_uint8_inputs(build())
    assert not needs_patch(once)
    with pytest.raises(ValueError):
      patch_uint8_inputs(once)


def test_a_model_without_image_inputs_says_so():
  graph = helper.make_graph([helper.make_node('Identity', ['x'], ['out'])], 'g',
                            [helper.make_tensor_value_info('x', TensorProto.UINT8, SHAPE)],
                            [helper.make_tensor_value_info('out', TensorProto.UINT8, SHAPE)])
  with pytest.raises(ValueError, match="none of"):
    patch_uint8_inputs(helper.make_model(graph))


def test_a_cast_to_the_wrong_type_is_refused():
  model = _model([
    helper.make_node('Cast', ['img'], ['a'], to=TensorProto.FLOAT),
    helper.make_node('Cast', ['big_img'], ['b'], to=TensorProto.FLOAT),
    helper.make_node('Concat', ['a', 'b'], ['out'], axis=1),
  ])
  with pytest.raises(ValueError, match="expected FLOAT16"):
    patch_uint8_inputs(model)


class TestTinygradPassthrough:
  """comma's 2026-09-01 export carries one org.tinygrad Contiguous node, which
  TensorRT's parser rejects outright. It is a layout hint, so it comes out."""

  def _with_contiguous(self, op='Contiguous', **kw):
    node = helper.make_node(op, ['cast_out'], ['hint'], domain=TINYGRAD_DOMAIN, **kw)
    graph = helper.make_graph([
      helper.make_node('Concat', ['img', 'big_img'], ['cat'], axis=1),
      helper.make_node('Cast', ['cat'], ['cast_out'], to=TensorProto.FLOAT16),
      node,
      helper.make_node('Identity', ['hint'], ['out']),
    ], 'test', _image_inputs(),
      [helper.make_tensor_value_info('out', TensorProto.FLOAT16, SHAPE)],
      value_info=[helper.make_tensor_value_info('hint', TensorProto.FLOAT16, SHAPE)])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 20),
                                                   helper.make_opsetid(TINYGRAD_DOMAIN, 1)])

  def test_the_node_goes_and_its_consumer_reads_the_source(self):
    model = self._with_contiguous()
    assert strip_tinygrad_ops(model) == 1
    ops = [(n.domain, n.op_type) for n in model.graph.node]
    assert (TINYGRAD_DOMAIN, 'Contiguous') not in ops
    identity = next(n for n in model.graph.node if n.op_type == 'Identity')
    assert list(identity.input) == ['cast_out']

  def test_the_opset_import_goes_with_it(self):
    # Left behind, it tells TensorRT the graph still needs a domain it has
    # never heard of, which is the failure we are removing.
    model = self._with_contiguous()
    strip_tinygrad_ops(model)
    assert all(o.domain != TINYGRAD_DOMAIN for o in model.opset_import)
    assert 'hint' not in [vi.name for vi in model.graph.value_info]

  def test_a_clean_model_is_untouched(self):
    model = _cast_after_concat()
    before = len(model.graph.node)
    assert strip_tinygrad_ops(model) == 0
    assert len(model.graph.node) == before

  def test_an_unknown_tinygrad_op_is_refused(self):
    # Dropping something that actually did work would change what the car sees,
    # so anything not known to be a passthrough has to stop the build.
    model = self._with_contiguous(op='SomethingElse')
    with pytest.raises(ValueError, match='unknown'):
      strip_tinygrad_ops(model)

  def test_an_op_carrying_attributes_is_refused(self):
    model = self._with_contiguous(axis=1)
    with pytest.raises(ValueError, match='passthrough'):
      strip_tinygrad_ops(model)

  def test_it_survives_the_checker_and_the_uint8_patch(self):
    model = self._with_contiguous()
    strip_tinygrad_ops(model)
    patch_uint8_inputs(model)
    onnx.checker.check_model(model, full_check=False)


def test_a_passthrough_feeding_a_graph_output_keeps_the_output_name():
  """The output name is what openpilot's output_slices and the parity tools
  address, so the producer has to take it over. An earlier version renamed both
  ends and left the graph output with no producer, which the checker catches."""
  x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2])
  y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2])
  relu = helper.make_node('Relu', ['x'], ['r'])
  hint = helper.make_node('Contiguous', ['r'], ['y'], domain=TINYGRAD_DOMAIN)
  model = helper.make_model(helper.make_graph([relu, hint], 'g', [x], [y]),
                            opset_imports=[helper.make_opsetid('', 17),
                                           helper.make_opsetid(TINYGRAD_DOMAIN, 1)])
  assert strip_tinygrad_ops(model) == 1
  assert [o.name for o in model.graph.output] == ['y']
  assert [list(n.output) for n in model.graph.node] == [['y']]
  onnx.checker.check_model(model, full_check=True)
