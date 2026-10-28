"""Exact causal past/current partition of the audited Cinque V3 fp16 graph.

Vision and past-token computation are independent in the same frame. The decoder
waits for both and uses every original past/current attention key and value.
No model parameters, recurrent feedback, frame IDs, or accuracy gates change.
"""

from pathlib import Path
import copy
import hashlib
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from onnx.utils import Extractor

import argparse
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=Path, required=True, help="original Cinque V3 model")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--base-tool", type=Path, required=True, help="built jetlink-prefill-base executable")
args = parser.parse_args()
if not __debug__:
  parser.error("Python -O disables the model-contract assertions; run without -O")


def sha256(path):
  digest = hashlib.sha256()
  with path.open('rb') as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


assert onnx.__version__ == "1.23.2", "reproduction is pinned to onnx==1.23.2"
assert np.__version__ == "2.5.4", "reproduction is pinned to numpy==2.5.4"
assert sha256(args.model) == "404a18cfd86d29637d20c697dfde245bb47c666ae016730ab674c65f4d1e1aa4", (
  "only the validated Cinque V3 export is supported"
)
OUT = args.out.resolve()
OUT.mkdir(parents=True, exist_ok=True)
assert not any(OUT.iterdir()), "choose an empty output directory"
temporary = tempfile.TemporaryDirectory(prefix="jetlink-prefill-base-")
source = Path(temporary.name) / "model.onnx"
subprocess.run([str(args.base_tool.resolve()), str(args.model.resolve()), temporary.name], check=True)
assert sha256(source) == "4ef3b6f248bf8d3ae0e1f76cfd0d5fcc4f079b6d910f735bd4cc22172b0d0166", (
  "base graph differs from the phone-validated graph"
)
m = onnx.load(source)
ex = Extractor(m)
plan = json.loads(Path(__file__).with_name("causal-prefill-cut-points.json").read_text())
vision_terms = plan[0]['outputs'][1:-1]
policy_terms = [*plan[1]['outputs'][:-1], *plan[2]['outputs']]
original_inputs = [v.name for v in m.graph.input]
other_inputs = [v for v in original_inputs if v not in ['new_img', 'state_img_q']]
policy = ex.extract_model(['view_39', *other_inputs], policy_terms)
shapes = {v.name: copy.deepcopy(v) for v in [*policy.graph.input, *policy.graph.output, *policy.graph.value_info]}
producer = {s: n for n in policy.graph.node for s in n.output}
users = {}
for n in policy.graph.node:
  for s in n.input:
    users.setdefault(s, []).append(n)
attention = []
for n in policy.graph.node:
  if n.op_type != 'Softmax':
    continue
  where = producer[n.input[0]]
  assert where.op_type == 'Where'
  negative_mask = producer[where.input[0]]
  assert negative_mask.op_type == 'Not'
  score = producer[where.input[2]]
  assert score.op_type == 'Mul'
  qk = producer[score.input[0]]
  assert qk.op_type == 'MatMul'
  kt = producer[qk.input[1]]
  assert kt.op_type == 'Transpose'
  av = [x for x in users[n.output[0]] if x.op_type == 'MatMul']
  assert len(av) == 1 and av[0].input[0] == n.output[0]
  attention.append({'mask': negative_mask.input[0], 'key': kt.input[0], 'value': av[0].input[1], 'key_transpose': kt.name, 'value_product': av[0].name})
assert len(attention) == 8


def prune(model):
  g = model.graph
  producers = {s: n for n in g.node for s in n.output}
  needed = {v.name for v in g.output}
  pending = list(needed)
  nodes = set()
  while pending:
    s = pending.pop()
    if s not in producers:
      continue
    node = producers[s]
    if node.name in nodes:
      continue
    nodes.add(node.name)
    for i in node.input:
      if i and i not in needed:
        needed.add(i)
        pending.append(i)
  kept = [n for n in g.node if n.name in nodes]
  del g.node[:]
  g.node.extend(kept)
  for field in ['initializer', 'input']:
    seq = getattr(g, field)
    retained = [v for v in seq if v.name in needed]
    del seq[:]
    seq.extend(retained)
  del g.value_info[:]
  model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
  onnx.checker.check_model(model, full_check=True)
  return model


def make(role):
  part = copy.deepcopy(policy)
  g = part.graph
  init = {v.name: v for v in g.initializer}
  prod = {s: n for n in g.node for s in n.output}
  length = 256 if role == 'history' else 32
  blocks = 8 if role == 'history' else 1

  def constant(name, arr):
    t = nh.from_array(np.asarray(arr), name)
    if name in init:
      init[name].CopyFrom(t)
    else:
      g.initializer.append(t)
      init[name] = g.initializer[-1]
    return name

  # The eight historical feature blocks are 24...31 of the 32-block
  # historical window. The ninth block is the current view_39 image feature.
  old_idx = nh.to_array(init['val_551'])
  assert old_idx.shape == (9, 1) and np.array_equal(old_idx.reshape(-1), np.arange(-9, 0))
  if role == 'history':
    cat = prod['cat_3']
    assert cat.op_type == 'Concat' and list(cat.input) == ['_to_copy_1', 'unsqueeze_1']
    cat.CopyFrom(h.make_node('Identity', ['_to_copy_1'], ['cat_3'], name='prefill_history_only'))
    constant('val_551', (old_idx + 33)[:8].astype(np.int64))
  else:
    index = prod['index']
    index.CopyFrom(h.make_node('Unsqueeze', ['view_39', 'val_3'], ['index'], name='prefill_current_only'))
  # Desire windows and temporal positions are token-local embedding inputs.
  desire_indices = nh.to_array(init['val_569']).reshape(9, 25)
  constant('val_569', (desire_indices[:8] if role == 'history' else desire_indices[8:]).reshape(-1))
  # Temporal position broadcasts must select the same absolute positions.
  temporal = ['unsqueeze_4', 'unsqueeze_8']
  for name in temporal:
    if name in init:
      a = nh.to_array(init[name])
      assert list(a.shape) == [9, 1, 512], (name, a.shape)
      constant(name + '__' + role, a[:8] if role == 'history' else a[8:])
  nodes = []
  for n in g.node:
    nodes.append(n)
    for name in temporal:
      if name not in n.output:
        continue
      output = name + '__' + role
      start = 0 if role == 'history' else 8
      end = 8 if role == 'history' else 9
      args = [
        constant(output + '__start', np.array([start], np.int64)),
        constant(output + '__end', np.array([end], np.int64)),
        constant(output + '__axis', np.array([0], np.int64)),
      ]
      nodes.append(h.make_node('Slice', [name, *args], [output], name=output))
  # Retarget only consumers of the positions, leaving the Slice source intact.
  for n in nodes:
    if n.op_type == 'Slice' and n.name.endswith('__' + role):
      continue
    for i, s in enumerate(n.input):
      if s in temporal:
        n.input[i] = s + '__' + role
  del g.node[:]
  g.node.extend(nodes)
  # Clone shape constants for each Reshape so no unrelated shape changes.
  for n in g.node:
    if n.op_type != 'Reshape' or n.input[1] not in init:
      continue
    a = nh.to_array(init[n.input[1]])
    if not np.any((a == 288) | (a == 9)):
      continue
    b = np.where(a == 288, length, np.where(a == 9, blocks, a))
    n.input[1] = constant(n.input[1] + '__' + role, b)
  # Causal attention allows past queries to be evaluated without current data.
  for info in attention:
    mask = nh.to_array(init[info['mask']])
    assert mask.shape == (1, 1, 288, 288) and not mask[:, :, :256, 256:].any()
    selected = mask[:, :, :256, :256] if role == 'history' else mask[:, :, 256:, :]
    constant(info['mask'], selected)
  past = []
  insert_before = {}
  renamed = {}
  for i, info in enumerate(attention):
    for kind in ['key', 'value']:
      current = info[kind]
      cached = f'prefill_{i}_{kind}'
      past.append(h.make_tensor_value_info(cached, onnx.TensorProto.FLOAT16, [1, 8, 256, 64]))
      if role == 'history':
        # A named graph output may still feed later layers. QNN GPU
        # rejects output Identity aliases in that case; rename the
        # producer and all consumers instead, with no arithmetic op.
        renamed[current] = cached
      else:
        target = info['key_transpose'] if kind == 'key' else info['value_product']
        joined = f'{cached}__with_current'
        insert_before.setdefault(target, []).append(h.make_node('Concat', [cached, current], [joined], name=joined, axis=2))
        node = next(n for n in g.node if n.name == target)
        node.input[0 if kind == 'key' else 1] = joined
  if role == 'history':
    for n in g.node:
      for field in [n.input, n.output]:
        for at, value in enumerate(field):
          if value in renamed:
            field[at] = renamed[value]
    del g.output[:]
    g.output.extend(past)
  else:
    g.input.extend(past)
    ordered = []
    for n in g.node:
      ordered.extend(insert_before.get(n.name, []))
      ordered.append(n)
      if n.op_type == 'Gather' and n.input[1] in init:
        a = nh.to_array(init[n.input[1]])
        if a.size == 1 and int(a.reshape(-1)[0]) == 287:
          n.input[1] = constant(n.input[1] + '__' + role, np.array(31, dtype=a.dtype))
    del g.node[:]
    g.node.extend(ordered)
  part = prune(part)
  assert ('view_39' not in [v.name for v in part.graph.input]) == (role == 'history')
  if role == 'decode':
    assert 'state_feat_q' not in [v.name for v in part.graph.input]
  return part


records = []
models = [
  ('vision', ex.extract_model(['new_img', 'state_img_q'], plan[0]['outputs']), 'htp'),
  ('history', make('history'), 'gpu'),
  ('decode', make('decode'), 'htp'),
  (
    'join',
    ex.extract_model([*vision_terms, *policy_terms, 'state_feat_q', 'desire', 'state_desire_q'], ['outputs', 'next_state_feat_q', 'next_state_desire_q']),
    'cpu',
  ),
]
for name, part, unit in models:
  part = prune(part)
  path = OUT / (name + '.onnx')
  onnx.save(part, path)
  onnx.checker.check_model(str(path), full_check=True)
  record = {
    'name': name,
    'file': path.name,
    'unit': unit,
    'nodes': len(part.graph.node),
    'weight_bytes': sum(len(t.raw_data) for t in part.graph.initializer),
    'inputs': [v.name for v in part.graph.input],
    'outputs': [v.name for v in part.graph.output],
    'sha256': sha256(path),
  }
  records.append(record)
  print(json.dumps(record), flush=True)
(OUT / 'manifest.json').write_text(
  json.dumps(
    {
      'format': 1,
      'strategy': 'causal-prefill-v1',
      'source_model_sha256': '404a18cfd86d29637d20c697dfde245bb47c666ae016730ab674c65f4d1e1aa4',
      'prepared_model_sha256': '4ef3b6f248bf8d3ae0e1f76cfd0d5fcc4f079b6d910f735bd4cc22172b0d0166',
      'parts': records,
    },
    indent=2,
  )
)
print('Saved exact same-frame (vision || history) -> decode -> join.', flush=True)

assert (OUT / "manifest.json").read_bytes() == Path(__file__).with_name("causal-prefill-manifest.json").read_bytes(), (
  "generated bundle differs from the validated bundle"
)
print("All four generated graph hashes and the manifest exactly match the phone-tested bundle.")
temporary.cleanup()
