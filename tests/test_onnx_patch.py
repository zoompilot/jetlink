"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

TensorRT's ONNX parser rejects UINT8 graph inputs outright, so the head of the
model has to be retyped before it will build. Exporters disagree about where
the cast goes, and the patcher has to cope with either.
"""
from __future__ import annotations

import numpy as np
import pytest

onnx = pytest.importorskip('onnx')

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from jetlink.onnx_patch import (  # noqa: E402
  TINYGRAD_DOMAIN,
  gemm_with_transposed_weight,
  layernorm_in_fp32,
  needs_patch,
  normalize_gather_indices,
  patch_uint8_inputs,
  strip_tinygrad_ops,
  vision_nodes,
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


class TestNegativeGatherIndices:
  """Apple's Neural Engine gathers garbage at a negative scalar index (measured
  on the driving model, correlation 0.03); the same index counted from the
  front is exact. The rewrite must change no value and touch nothing it
  cannot prove."""

  def _model(self, index, axis=1, with_shape=True, share=False):
    inp = helper.make_tensor_value_info('x', TensorProto.FLOAT16, [1, 288, 512])
    mid = helper.make_tensor_value_info('y', TensorProto.FLOAT16, [1, 288, 512] if with_shape else None)
    nodes = [helper.make_node('Identity', ['x'], ['y']),
             helper.make_node('Gather', ['y', 'idx'], ['sel'], axis=axis)]
    inits = [numpy_helper.from_array(np.array(index, np.int64), 'idx')]
    outs = [helper.make_tensor_value_info('sel', TensorProto.FLOAT16, [1, 512])]
    if share:
      nodes.append(helper.make_node('Gather', ['x', 'idx'], ['sel2'], axis=axis))
      outs.append(helper.make_tensor_value_info('sel2', TensorProto.FLOAT16, [1, 512]))
    graph = helper.make_graph(nodes, 'g', [inp], outs, initializer=inits, value_info=[mid] if with_shape else [])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])

  def _index_of(self, model, output):
    node = next(n for n in model.graph.node if n.output[0] == output)
    init = {t.name: t for t in model.graph.initializer}
    return numpy_helper.to_array(init[node.input[1]]).tolist()

  def test_minus_one_becomes_the_last_index(self):
    m = self._model(-1)
    assert normalize_gather_indices(m) == 1
    assert self._index_of(m, 'sel') == 287
    onnx.checker.check_model(m)

  def test_a_shared_index_is_not_rewritten_under_the_other_node(self):
    m = self._model(-1, share=True)
    assert normalize_gather_indices(m) == 2
    assert self._index_of(m, 'sel') == 287 and self._index_of(m, 'sel2') == 287
    init = {t.name for t in m.graph.initializer}
    assert 'idx' in init and 'sel__index' in init and 'sel2__index' in init

  def test_a_positive_index_and_a_vector_without_negatives_are_untouched(self):
    for index in (3, [0, 5, 287]):
      m = self._model(index)
      assert normalize_gather_indices(m) == 0
      assert self._index_of(m, 'sel') == index

  def test_a_vector_with_a_negative_is_rewritten_elementwise(self):
    m = self._model([0, -1, -288])
    assert normalize_gather_indices(m) == 1
    assert self._index_of(m, 'sel') == [0, 287, 0]

  def test_an_index_out_of_range_is_refused(self):
    with pytest.raises(ValueError, match='out of range'):
      normalize_gather_indices(self._model(-289))

  def test_an_unknown_size_is_left_alone(self):
    """No value_info and nothing the shape inferrer can see: the node stays,
    because a guessed size would change the value."""
    m = self._model(-1, with_shape=False)
    # the inferrer sees through Identity here, so take it away to make the size unknowable
    m.graph.node[0].op_type = 'Contiguous'
    m.graph.node[0].domain = 'org.tinygrad'
    m.opset_import.append(helper.make_opsetid('org.tinygrad', 1))
    assert normalize_gather_indices(m) == 0
    assert self._index_of(m, 'sel') == -1

  def test_the_shape_inferrer_fills_a_missing_value_info(self):
    m = self._model(-1, with_shape=False)
    assert normalize_gather_indices(m) == 1
    assert self._index_of(m, 'sel') == 287

  def test_values_are_preserved(self):
    # onnx's own reference evaluator, not onnxruntime: a runtime session in
    # the test process is what made the suite abort at exit beside tinygrad.
    from onnx.reference import ReferenceEvaluator
    m = self._model(-1, axis=1)
    x = np.random.default_rng(0).standard_normal((1, 288, 512)).astype(np.float16)
    before = ReferenceEvaluator(m).run(None, {'x': x})[0]
    normalize_gather_indices(m)
    after = ReferenceEvaluator(m).run(None, {'x': x})[0]
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(after, x[:, -1, :])


class TestLayerNormInFp32:
  """The Neural Engine's fp16 LayerNormalization loses the parity gate on this
  model's residual stream; in fp32 CoreML runs it elsewhere and exactly."""

  def _model(self, dtype=TensorProto.FLOAT16, with_bias=True):
    x = helper.make_tensor_value_info('x', dtype, [1, 4, 8])
    y = helper.make_tensor_value_info('y', dtype, [1, 4, 8])
    np_dtype = np.float16 if dtype == TensorProto.FLOAT16 else np.float32
    inits = [numpy_helper.from_array(np.linspace(0.5, 1.5, 8).astype(np_dtype), 'scale')]
    inputs = ['x', 'scale']
    if with_bias:
      inits.append(numpy_helper.from_array(np.linspace(-1, 1, 8).astype(np_dtype), 'bias'))
      inputs.append('bias')
    nodes = [helper.make_node('Identity', ['x'], ['h']),
             helper.make_node('LayerNormalization', ['h'] + inputs[1:], ['n'], axis=-1, epsilon=1e-5, name='ln'),
             helper.make_node('Identity', ['n'], ['y'])]
    return helper.make_model(helper.make_graph(nodes, 'g', [x], [y], initializer=inits),
                             opset_imports=[helper.make_opsetid('', 17)])

  def test_the_node_runs_in_fp32_between_casts(self):
    m = self._model()
    assert layernorm_in_fp32(m) == 1
    ops = [n.op_type for n in m.graph.node]
    assert ops == ['Identity', 'Cast', 'LayerNormalization', 'Cast', 'Identity']
    ln = m.graph.node[2]
    init = {t.name: t for t in m.graph.initializer}
    assert all(init[i].data_type == TensorProto.FLOAT for i in ln.input[1:])
    assert init['scale'].data_type == TensorProto.FLOAT16, 'the original initializer must stay for anything else that reads it'
    onnx.checker.check_model(m)

  def test_values_match_an_fp32_layernorm_where_fp16_overflows(self):
    """Residuals in the hundreds: their squares overflow fp16 inside the
    normalisation, which is what the Neural Engine gets wrong. The fp32 node
    matches numpy's float32 answer to fp16 output rounding."""
    from onnx.reference import ReferenceEvaluator
    m = self._model()
    x = (np.random.default_rng(1).standard_normal((1, 4, 8)) * 300).astype(np.float16)
    x32 = x.astype(np.float32)
    mean = x32.mean(-1, keepdims=True)
    var = ((x32 - mean) ** 2).mean(-1, keepdims=True)
    want = (x32 - mean) / np.sqrt(var + 1e-5) * np.linspace(0.5, 1.5, 8, dtype=np.float32) + np.linspace(-1, 1, 8, dtype=np.float32)
    with np.errstate(over='ignore'):
      before = ReferenceEvaluator(m).run(None, {'x': x})[0].astype(np.float32)
    assert not np.allclose(before, want, atol=0.05), 'fp16 should have overflowed here; the test proves nothing otherwise'
    assert layernorm_in_fp32(m) == 1
    after = ReferenceEvaluator(m).run(None, {'x': x})[0]
    assert after.dtype == np.float16 and m.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT16
    np.testing.assert_allclose(after.astype(np.float32), want, atol=5e-3, rtol=5e-3)

  def test_layernorms_sharing_an_input_share_one_cast(self):
    """The driving model's five head MLPs each normalise the same selected
    token; a cast per node would define the same name five times."""
    m = self._model()
    g = m.graph
    g.node.append(helper.make_node('LayerNormalization', ['h', 'scale', 'bias'], ['n2'], axis=-1, epsilon=1e-5, name='ln2'))
    g.node.append(helper.make_node('Add', ['n', 'n2'], ['y2']))
    g.output[0].name = 'y2'
    g.node[2].input[0] = 'n'   # the old Identity now unused; keep the graph simple
    del g.node[2]
    assert layernorm_in_fp32(m) == 2
    casts_in = [n for n in g.node if n.op_type == 'Cast' and n.input[0] == 'h']
    assert len(casts_in) == 1
    onnx.checker.check_model(m)

  def test_an_fp32_layernorm_and_one_without_a_bias_are_handled(self):
    m = self._model(dtype=TensorProto.FLOAT)
    assert layernorm_in_fp32(m) == 0
    m = self._model(with_bias=False)
    assert layernorm_in_fp32(m) == 1
    onnx.checker.check_model(m)


class TestVisionNodes:
  def test_the_trunk_and_a_head_off_it_are_vision_and_the_rest_is_not(self, tmp_path):
    from tests import tiny_model
    model = onnx.load(str(tiny_model.write(tmp_path / 'tiny.onnx', with_contiguous=False)))
    g = model.graph
    for i, n in enumerate(g.node):
      n.name = f'n{i}_{n.op_type}'
    g.node.append(helper.make_node('Relu', ['img_mean'], ['head'], name='head'))
    names = vision_nodes(model)
    assert names == {'n0_Cast', 'n1_Cast', 'n2_Concat', 'n3_ReduceMean', 'head'}

  def test_only_restricts_the_fp32_rewrite(self, tmp_path):
    m = TestLayerNormInFp32()._model()
    m.graph.node[1].name = 'ln'
    assert layernorm_in_fp32(m, only={'somewhere else'}) == 0
    assert layernorm_in_fp32(m, only={'ln'}) == 1

  def test_a_model_without_image_inputs_says_so(self):
    g = helper.make_graph([helper.make_node('Identity', ['x'], ['y'])], 'g',
                          [helper.make_tensor_value_info('x', TensorProto.FLOAT16, [1])],
                          [helper.make_tensor_value_info('y', TensorProto.FLOAT16, [1])])
    with pytest.raises(ValueError, match='graph inputs'):
      vision_nodes(helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)]))


class TestGemmWithTransposedWeight:
  """The CoreML EP writes a Gemm's weight into the MIL as text unless it
  arrives already transposed, at about six bytes per fp16 value. These check
  the rewrite that hands it one, and that it leaves everything else alone.
  """

  @staticmethod
  def _matmul_add(a_shape, k=64, n=32, bias_dims=None, with_add=True):
    w = np.arange(k * n, dtype=np.float16).reshape(k, n) / (k * n)
    b = np.arange(n, dtype=np.float16) / n
    out_shape = list(a_shape[:-1]) + [n]
    nodes = [helper.make_node('MatMul', ['a', 'W'], ['mm'], name='mm')]
    inits = [numpy_helper.from_array(w, 'W')]
    if with_add:
      nodes.append(helper.make_node('Add', ['mm', 'B'], ['out'], name='add'))
      inits.append(numpy_helper.from_array(
        b.reshape(bias_dims) if bias_dims is not None else b, 'B'))
    else:
      nodes.append(helper.make_node('Identity', ['mm'], ['out'], name='id'))
    graph = helper.make_graph(
      nodes, 'g', [helper.make_tensor_value_info('a', TensorProto.FLOAT16, list(a_shape))],
      [helper.make_tensor_value_info('out', TensorProto.FLOAT16, out_shape)], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])

  def test_a_rank_2_matmul_add_becomes_one_gemm(self):
    m = self._matmul_add([8, 64])
    assert gemm_with_transposed_weight(m) == 1
    assert [n.op_type for n in m.graph.node] == ['Gemm']
    gemm = m.graph.node[0]
    assert next(a.i for a in gemm.attribute if a.name == 'transB') == 1
    assert gemm.output[0] == 'out'
    onnx.checker.check_model(m)

  def test_the_weight_is_stored_transposed_and_the_original_is_dropped(self):
    m = self._matmul_add([8, 64])
    before = numpy_helper.to_array(next(t for t in m.graph.initializer if t.name == 'W'))
    gemm_with_transposed_weight(m)
    assert 'W' not in {t.name for t in m.graph.initializer}
    stored = numpy_helper.to_array(next(t for t in m.graph.initializer if t.name.endswith('__wt')))
    assert stored.shape == (before.shape[1], before.shape[0])
    np.testing.assert_array_equal(stored, before.T)

  @pytest.mark.parametrize('a_shape', [[1, 8, 64], [1, 4, 8, 64]])
  def test_a_higher_rank_matmul_gets_the_reshape_pair_onnxruntime_would_add(self, a_shape):
    m = self._matmul_add(a_shape)
    assert gemm_with_transposed_weight(m) == 1
    assert [n.op_type for n in m.graph.node] == ['Reshape', 'Gemm', 'Reshape']
    flat = numpy_helper.to_array(
      next(t for t in m.graph.initializer if t.name.endswith('__flat_shape')))
    back = numpy_helper.to_array(
      next(t for t in m.graph.initializer if t.name.endswith('__out_shape')))
    np.testing.assert_array_equal(flat, [-1, 64])
    np.testing.assert_array_equal(back, list(a_shape[:-1]) + [32])
    assert m.graph.node[-1].output[0] == 'out'
    onnx.checker.check_model(m)

  @pytest.mark.parametrize('a_shape', [[8, 64], [1, 8, 64], [1, 4, 8, 64]])
  def test_the_rewrite_keeps_the_arithmetic(self, a_shape):
    # onnx's own evaluator, never onnxruntime: importing that into the test
    # process aborts the suite at exit (see backends.ort.quiet)
    from onnx.reference import ReferenceEvaluator
    m = self._matmul_add(a_shape)
    size = int(np.prod(a_shape))
    a = ((np.arange(size, dtype=np.float32).reshape(a_shape) / size) - 0.5).astype(np.float16)
    before = ReferenceEvaluator(m).run(None, {'a': a})[0]
    gemm_with_transposed_weight(m)
    after = ReferenceEvaluator(m).run(None, {'a': a})[0]
    assert before.shape == after.shape
    np.testing.assert_allclose(before.astype(np.float32), after.astype(np.float32),
                               rtol=1e-3, atol=1e-3)

  def test_a_matmul_with_no_add_is_left_alone(self):
    # a bare MatMul already lowers to a CoreML matmul that reads the weight file
    m = self._matmul_add([8, 64], with_add=False)
    assert gemm_with_transposed_weight(m) == 0

  def test_a_weight_below_the_threshold_is_left_alone(self):
    m = self._matmul_add([4, 8], k=8, n=8)
    assert gemm_with_transposed_weight(m) == 0

  def test_an_add_that_is_not_a_bias_is_left_alone(self):
    # a full-width Add is elementwise arithmetic, not a Gemm's C
    m = self._matmul_add([8, 64], bias_dims=(1, 32))
    assert gemm_with_transposed_weight(m) == 0

  def test_a_matmul_whose_output_leaves_the_graph_is_left_alone(self):
    m = self._matmul_add([8, 64])
    m.graph.output.append(helper.make_tensor_value_info('mm', TensorProto.FLOAT16, [8, 32]))
    assert gemm_with_transposed_weight(m) == 0

  def test_a_matmul_without_a_static_shape_is_left_alone(self):
    m = self._matmul_add([8, 64])
    del m.graph.input[0].type.tensor_type.shape.dim[:]
    assert gemm_with_transposed_weight(m) == 0

  def test_a_shared_weight_is_transposed_once_per_use(self):
    m = self._matmul_add([8, 64])
    g = m.graph
    g.node.extend([helper.make_node('MatMul', ['a', 'W'], ['mm2'], name='mm2'),
                   helper.make_node('Add', ['mm2', 'B'], ['out2'], name='add2')])
    g.output.append(helper.make_tensor_value_info('out2', TensorProto.FLOAT16, [8, 32]))
    assert gemm_with_transposed_weight(m) == 2
    assert sum(1 for t in g.initializer if t.name.endswith('__wt')) == 2
    assert 'W' not in {t.name for t in g.initializer}
    onnx.checker.check_model(m)
