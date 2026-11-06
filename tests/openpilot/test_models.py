"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Which large model runs and where its bytes come from: the model index the
comma keeps a copy of, the model manager's pick as the fork's adapter hands it
over, the file on disk, and the index as the model manager's list.
"""
import dataclasses
import json
import os
import time
import unittest
from unittest import mock

from jetlink.openpilot import models
from jetlink.registry.index import DEFAULT_BIG_MODEL_NAME, DEFAULT_BIG_MODEL_REF, RUNTIME, NetworkError, RegistryError, parse_index
from tests.openpilot.fakes import OpenpilotTest


def bundle(ref: str, name: str, index: int = 0, version=19) -> dict:
  """A bundle as the model manager's catalog JSON carries it."""
  return {'ref': ref, 'display_name': name, 'index': index, 'minimum_selector_version': str(version)}


def entry(ref: str, name: str, oid: str, size: int, folder: str = 'Master Models', **extra) -> dict:
  """A listed model as the index carries it."""
  return {'ref': ref, 'name': name, 'oid': oid, 'size': size, 'folder': folder, **extra}


def index_json(*listed, refs=None, onnx=None, default=None) -> dict:
  return {'schema': 1, 'default': default, 'models': list(listed), 'refs': refs or {}, 'onnx': onnx or {}}


REF_A, REF_B, REF_C = 'a' * 40, 'b' * 40, 'c' * 40
SHA_D = 'd' * 64
ALPHA = entry(REF_A, 'Alpha', '1' * 64, 766_000_000)
BETA = entry(REF_B, 'Beta', '2' * 64, 1_757_000_000)


class ModelsTest(OpenpilotTest):
  def setUp(self):
    super().setUp()
    self.models = self.parts.models

  def index(self, *listed, refs=None, onnx=None, default=None):
    """The comma's copy of the index, as a fetch leaves it."""
    path = self.models.index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index_json(*listed, refs=refs, onnx=onnx, default=default)))
    self.models._index_cache = None

  def fetched(self, *listed, refs=None, onnx=None, default=None, **kw):
    """fetch_index serving this index."""
    data = index_json(*listed, refs=refs, onnx=onnx, default=default)
    return mock.patch('jetlink.registry.index.fetch_index', return_value=(data, parse_index(data)), **kw)

  def catalog(self, *bundles):
    """The model manager's cached catalog."""
    self.op.store['ModelManager_ModelsCache_Chestnut'] = {'bundles': list(bundles)}

  def write_slot(self, ago, **slot):
    """A pick in the comma's list `ago` seconds ago: the slot and its file, as Params writes them."""
    key = self.op.keys.big_model
    self.op.store[key] = slot
    path = self.op.store_dir / key
    path.write_text('{}')
    at = time.time() - ago
    os.utime(path, (at, at))
    self.models._slot_cache = None

  def hear_pick(self, ago, **pick):
    """A hello from a server whose pick was made `ago` seconds ago."""
    return self.models.heard({'pick': {**pick, 'age_s': ago, 'by': 'here'}})


class TestIndex(ModelsTest):
  """The comma's copy of the model index: what is listed, in its order, and
  which ONNX every ref and hash is, with no network."""

  def test_the_rows_are_the_index_in_its_order(self):
    self.index(BETA, ALPHA)
    self.assertEqual(self.models.model_index(), [{k: e[k] for k in ('name', 'ref', 'oid', 'size')} for e in (BETA, ALPHA)])

  def test_no_index_yet_is_empty(self):
    self.assertIsNone(self.models.index())
    self.assertEqual(self.models.model_index(), [])

  def test_a_copy_that_does_not_parse_is_none_not_an_error(self):
    # read from the UI's param thread, where an exception takes the panel down
    path = self.models.index_path()
    path.parent.mkdir(parents=True)
    path.write_text('{"schema": 1, "models": [')
    self.assertEqual(self.models.model_index(), [])
    self.assertTrue(self.op.log.has("the comma's model index does not parse", 'exception'))
    # the same copy is told of once, not every INDEX_TTL
    with mock.patch.object(models.time, 'monotonic', return_value=models.time.monotonic() + models.INDEX_TTL):
      self.models.model_index()
    self.assertEqual(len(self.op.log.lines('exception')), 1)

  def test_a_second_read_within_the_ttl_costs_nothing(self):
    # the UI names the active model every frame
    self.index(ALPHA)
    with mock.patch.object(models, 'parse_index', wraps=models.parse_index) as parse:
      first = self.models.model_index()
      self.assertIs(self.models.model_index(), first)
    self.assertEqual(parse.call_count, 1)

  def test_after_the_ttl_only_a_changed_copy_is_parsed_again(self):
    self.index(ALPHA)
    later = mock.patch.object(models.time, 'monotonic', return_value=models.time.monotonic() + models.INDEX_TTL)
    with mock.patch.object(models, 'parse_index', wraps=models.parse_index) as parse:
      self.models.model_index()
      with later:
        self.models.model_index()
      self.assertEqual(parse.call_count, 1)
      self.index(ALPHA, BETA)
      self.assertEqual([m['ref'] for m in self.models.model_index()], [REF_A, REF_B])
    self.assertEqual(parse.call_count, 2)

  def test_a_ref_the_index_resolves_but_does_not_list(self):
    # a recompile, listed under the newest ref, or a pull request since closed
    self.index(BETA, refs={REF_C: {'oid': '3' * 64, 'size': 5}})
    with mock.patch('jetlink.registry.index.fetch_index') as fetch:
      self.assertEqual(self.models.resolve(REF_C, None), ('3' * 64, 5))
    fetch.assert_not_called()

  def test_a_hash_is_sized_from_the_onnx_the_index_knows(self):
    self.index(BETA, onnx={SHA_D: {'size': 792_487_890}})
    self.assertEqual(self.models.resolve(None, SHA_D), (SHA_D, 792_487_890))
    self.assertEqual(self.models.size_for(SHA_D), 792_487_890)
    self.assertIsNone(self.models.size_for('9' * 64))

  def test_a_miss_fetches_the_index_and_keeps_it(self):
    self.index(BETA)
    with self.fetched(BETA, entry(REF_C, 'Gamma', '3' * 64, 5)) as fetch:
      self.assertEqual(self.models.resolve(REF_C, None), ('3' * 64, 5))
    fetch.assert_called_once_with()
    kept = json.loads(self.models.index_path().read_text())
    self.assertEqual([m['ref'] for m in kept['models']], [REF_B, REF_C])

  def test_an_index_that_did_not_change_is_not_written_again(self):
    # every reader would parse it again
    with self.fetched(BETA):
      self.models.refresh_index()
    before = self.models.index_path().stat().st_mtime_ns
    with self.fetched(BETA), mock.patch('jetlink.comma.gadget.write_record') as write:
      self.models.refresh_index()
    write.assert_not_called()
    self.assertEqual(self.models.index_path().stat().st_mtime_ns, before)

  def test_a_miss_fetches_at_most_once_a_while(self):
    self.index(BETA)
    with self.fetched(BETA) as fetch:
      for _ in range(2):
        with self.assertRaises(RegistryError):
          self.models.resolve(REF_C, None)
    fetch.assert_called_once_with()

  def test_a_failed_fetch_raises_and_keeps_the_copy(self):
    self.index(BETA)
    with mock.patch('jetlink.registry.index.fetch_index', side_effect=NetworkError('offline')), \
         self.assertRaises(NetworkError):
      self.models.resolve(REF_C, None)
    self.assertEqual([m['ref'] for m in self.models.model_index()], [REF_B])


class TestSelectedModel(ModelsTest):
  """The pick is the model manager's big-model slot, the same one a chestnut runs
  from, listed or not; the index names its ONNX and supplies the default."""

  INDEX = (entry(REF_A, 'Alpha', 'a' * 64, 10), entry(REF_B, 'Beta', 'b' * 64, 20))

  def select_with(self, slot_ref, default=REF_B, listed=INDEX, refs=None):
    if listed:
      self.index(*listed, refs=refs, default=default)
    pick = {'name': 'Picked', 'ref': slot_ref, 'sha256': None} if slot_ref else None
    with mock.patch.object(self.models, 'selected_slot', return_value=pick):
      return self.models.selected_model()

  def test_an_empty_slot_takes_the_index_s_default(self):
    # a chestnut's default is the model in the tree; the accelerator's is the index's
    assert self.select_with(None)['name'] == 'Beta'

  def test_a_default_not_listed_falls_to_the_first_listed(self):
    assert self.select_with(None, default='f' * 40)['name'] == 'Alpha'
    assert self.select_with(None, default=None)['name'] == 'Alpha'

  def test_the_pick_is_the_slot_with_its_onnx(self):
    assert self.select_with(REF_A) == {'name': 'Picked', 'ref': REF_A, 'oid': 'a' * 64, 'size': 10}
    # listed or not
    assert self.select_with(REF_C, refs={REF_C: {'oid': 'c' * 64, 'size': 30}}) == \
      {'name': 'Picked', 'ref': REF_C, 'oid': 'c' * 64, 'size': 30}

  def test_a_pick_the_index_does_not_know_has_no_oid(self):
    # the provisioning run fetches the index again for it (resolve)
    assert self.select_with(REF_C) == {'name': 'Picked', 'ref': REF_C, 'oid': None, 'size': None}

  def test_with_no_index_yet_the_default_is_named_by_its_ref(self):
    assert self.select_with(None, listed=()) == {'name': DEFAULT_BIG_MODEL_NAME, 'ref': DEFAULT_BIG_MODEL_REF, 'oid': None, 'size': None}

  def test_the_name_is_the_picks_or_none(self):
    with mock.patch.object(self.models, 'selected_model', return_value={'name': 'Picked'}):
      self.assertEqual(self.models.selected_model_name(), 'Picked')
    with mock.patch.object(self.models, 'selected_model', return_value=None):
      self.assertIsNone(self.models.selected_model_name())

  def test_end_to_end_off_the_params(self):
    self.index(ALPHA, BETA)
    self.op.store['ModelManager_ActiveBundleChestnut'] = {'ref': REF_A, 'displayName': 'Alpha'}
    self.assertEqual(self.models.selected_model(), {'name': 'Alpha', 'ref': REF_A, 'oid': '1' * 64, 'size': 766_000_000})


class TestPicksOnBothEnds(ModelsTest):
  """A model picked on the server drives without being picked here too, and
  the newer pick wins wherever it was made: the slot's by when it was
  written, the server's by the age its hello gave."""

  SLOT = 'ModelManager_ActiveBundleChestnut'

  def setUp(self):
    super().setUp()
    self.op.keys = dataclasses.replace(self.op.keys, select='ModelManager_DownloadRef')
    self.index(ALPHA, BETA)
    self.catalog(bundle(REF_A, 'Alpha', 1), bundle(REF_B, 'Beta', 2))

  def pick_here(self, ref, name, ago):
    self.write_slot(ago, ref=ref, displayName=name)

  def hear(self, ref, ago, sha256='2' * 64, nbytes=1_757_000_000, name='Beta on the server'):
    return self.hear_pick(ago, ref=ref, sha256=sha256, nbytes=nbytes, name=name)

  def test_the_hello_carries_our_pick_and_its_age(self):
    self.pick_here(REF_A, 'Alpha', ago=30)
    pick = self.models.hello_pick()
    self.assertEqual({k: pick[k] for k in ('ref', 'sha256', 'nbytes', 'name')},
                     {'ref': REF_A, 'sha256': '1' * 64, 'nbytes': 766_000_000, 'name': 'Alpha'})
    self.assertAlmostEqual(pick['age_s'], 30, delta=2)

  def test_with_no_pick_of_our_own_the_hello_says_none(self):
    self.assertIsNone(self.models.hello_pick())

  def test_a_newer_pick_on_the_server_drives(self):
    self.pick_here(REF_A, 'Alpha', ago=600)
    self.hear(REF_B, ago=60)
    # our name for it, the server's oid and size
    self.assertEqual(self.models.selected_model(), {'name': 'Beta', 'ref': REF_B, 'oid': '2' * 64, 'size': 1_757_000_000})

  def test_an_older_pick_on_the_server_does_not(self):
    self.pick_here(REF_A, 'Alpha', ago=60)
    self.hear(REF_B, ago=600)
    self.assertEqual(self.models.selected_model()['ref'], REF_A)

  def test_with_no_pick_here_the_servers_drives(self):
    self.hear(REF_B, ago=6000)
    self.assertEqual(self.models.selected_model()['ref'], REF_B)

  def test_a_server_too_old_to_say_leaves_our_pick(self):
    self.pick_here(REF_A, 'Alpha', ago=600)
    self.models.heard({'protocol': 3})
    self.assertEqual(self.models.selected_model()['ref'], REF_A)

  def test_another_process_reads_the_owners_record(self):
    # the UI says no hello: it goes by what the owner kept of the last one
    from jetlink.comma import gadget
    self.pick_here(REF_A, 'Alpha', ago=600)
    gadget.write_record(gadget.SERVER, {'pick': {'ref': REF_B, 'sha256': '2' * 64, 'nbytes': 5, 'at': time.time() - 60}})
    self.assertEqual(self.models.selected_model()['ref'], REF_B)

  def test_a_model_the_server_has_and_we_do_not_list_still_drives(self):
    # imported on a Mac, say: named and sized by the server
    self.hear(None, ago=5, sha256='9' * 64, nbytes=42, name='My export')
    self.assertEqual(self.models.selected_model(), {'name': 'My export', 'ref': None, 'oid': '9' * 64, 'size': 42})

  def test_the_servers_pick_is_asked_for_here_too(self):
    self.pick_here(REF_A, 'Alpha', ago=600)
    self.hear(REF_B, ago=60)
    self.models.adopt()
    self.assertEqual(self.op.store['ModelManager_DownloadRef'], REF_B)
    self.assertTrue(self.op.log.has('was picked on the server'))

  def test_one_the_model_managers_list_lacks_is_not_asked_for(self):
    # the model manager would hold a ref it cannot resolve and never let go
    self.pick_here(REF_A, 'Alpha', ago=600)
    self.hear(REF_C, ago=60, sha256='3' * 64, nbytes=7)
    self.models.adopt()
    self.assertNotIn('ModelManager_DownloadRef', self.op.store)
    # it drives all the same, by the server's oid and size
    self.assertEqual(self.models.selected_model()['oid'], '3' * 64)

  def test_nothing_is_asked_for_when_ours_is_newer_or_the_same(self):
    self.pick_here(REF_B, 'Beta', ago=600)
    self.hear(REF_B, ago=60)
    self.models.adopt()
    self.pick_here(REF_A, 'Alpha', ago=6)
    self.models.adopt()
    self.assertNotIn('ModelManager_DownloadRef', self.op.store)

  def test_a_download_the_user_queued_is_not_written_over(self):
    self.pick_here(REF_A, 'Alpha', ago=600)
    self.op.store['ModelManager_DownloadRef'] = 'small-model-ref'
    self.hear(REF_B, ago=60)
    self.models.adopt()
    self.assertEqual(self.op.store['ModelManager_DownloadRef'], 'small-model-ref')
    self.assertEqual(self.models.selected_model()['ref'], REF_B)

  def empty_slot(self):
    """The model manager's Default, chosen: the slot taken out."""
    del self.op.store[self.SLOT]
    (self.op.store_dir / self.SLOT).unlink()
    self.models._slot_cache = None

  def answer(self, ref, sent, ago=0.0):
    """A hello that carried `sent`, answered with a pick of `ref` from a
    server whose clock calls it `ago` seconds old."""
    return self.models.heard({'pick': {'ref': ref, 'sha256': None, 'nbytes': None, 'age_s': ago}}, sent)

  def test_the_hello_dates_our_pick_by_this_clock(self):
    self.pick_here(REF_A, 'Alpha', ago=30)
    self.assertAlmostEqual(self.models.hello_pick()['at'], time.time() - 30, delta=2)

  def test_the_default_chosen_here_is_a_pick_the_server_does_not_undo(self):
    # both ends named Beta; then the user chose the Default in the comma's list
    self.index(entry(DEFAULT_BIG_MODEL_REF, 'Cinque Terre V3', 'c' * 64, 30), ALPHA, BETA, default=DEFAULT_BIG_MODEL_REF)
    self.pick_here(REF_B, 'Beta', ago=600)
    self.answer(REF_B, self.models.hello_pick(), ago=600)
    self.empty_slot()
    self.assertEqual(self.models.selected_model()['ref'], DEFAULT_BIG_MODEL_REF)
    self.models.adopt()
    self.assertNotIn('ModelManager_DownloadRef', self.op.store)
    # and the next hello names it, dated when it was chosen
    pick = self.models.hello_pick()
    self.assertEqual(pick['ref'], DEFAULT_BIG_MODEL_REF)
    self.assertLess(pick['age_s'], 5)
    # which another process knows too: it is kept beside the downloads
    other = models.Models(self.op)
    self.assertEqual(other.hello_pick()['ref'], DEFAULT_BIG_MODEL_REF)

  def test_a_slot_never_filled_is_no_pick(self):
    self.assertIsNone(self.models.hello_pick())
    self.hear(REF_B, ago=6000)
    self.assertEqual(self.models.selected_model()['ref'], REF_B)

  def test_the_servers_answer_stands_for_the_pick_it_was_asked_with(self):
    # a Jetson whose clock started again behind dates its pick oddly; what
    # counts is that it chose it knowing ours
    self.pick_here(REF_A, 'Alpha', ago=60)
    self.answer(REF_B, self.models.hello_pick(), ago=90_000)
    self.assertEqual(self.models.selected_model()['ref'], REF_B)
    self.models.adopt()
    self.assertEqual(self.op.store['ModelManager_DownloadRef'], REF_B)

  def test_a_pick_made_here_since_the_hello_is_ours(self):
    self.pick_here(REF_A, 'Alpha', ago=600)
    self.answer(REF_B, self.models.hello_pick(), ago=0)
    self.pick_here(REF_A, 'Alpha', ago=5)   # picked again after the hello
    self.assertEqual(self.models.selected_model()['ref'], REF_A)
    self.empty_slot()                       # or the default chosen
    self.assertEqual(self.models.selected_model()['ref'], self.models.default_model()['ref'])

  def test_an_answer_to_a_hello_with_no_pick_does_not_beat_one_made_since(self):
    self.answer(REF_B, None)
    self.pick_here(REF_A, 'Alpha', ago=5)
    self.assertEqual(self.models.selected_model()['ref'], REF_A)

  def test_without_a_place_to_ask_it_drives_all_the_same(self):
    self.op.keys = dataclasses.replace(self.op.keys, select=None)
    self.pick_here(REF_A, 'Alpha', ago=600)
    self.hear(REF_B, ago=60)
    self.models.adopt()
    self.assertNotIn('ModelManager_DownloadRef', self.op.store)
    self.assertEqual(self.models.selected_model()['ref'], REF_B)
    # the slot is the model manager's, never written over
    self.assertEqual(self.op.store[self.SLOT]['ref'], REF_A)


class TestPicksByHash(ModelsTest):
  """StarPilot's shape: a picker of its own whose catalog knows a model by its
  ONNX's hash alone, no catalog for jetlink and no model manager to ask. The
  pick names its model by the hash, is sized from the index, and a newer pick
  on the server is written into the slot so the picker shows it."""

  SLOT = 'JetlinkModel'

  def setUp(self):
    super().setUp()
    self.op.keys = dataclasses.replace(self.op.keys, big_model=self.SLOT, catalog=None, select=None)
    self.index(entry(REF_B, 'Beta', '2' * 64, 20), onnx={SHA_D: {'size': 7}})

  def pick_here(self, sha256, name, ago):
    self.write_slot(ago, sha256=sha256, displayName=name)

  def hear(self, ref, sha256, ago, nbytes=None, name='On the server'):
    return self.hear_pick(ago, ref=ref, sha256=sha256, nbytes=nbytes, name=name)

  def test_the_hash_is_its_identity_sized_from_the_index(self):
    self.pick_here(SHA_D, 'Delta', ago=60)
    self.assertEqual(self.models.selected_model(), {'name': 'Delta', 'ref': None, 'oid': SHA_D, 'size': 7})
    self.assertEqual(self.models.resolve(None, SHA_D), (SHA_D, 7))

  def test_a_hash_the_index_does_not_know_has_no_size(self):
    self.pick_here('9' * 64, 'Mine', ago=60)
    self.assertEqual(self.models.selected_model(), {'name': 'Mine', 'ref': None, 'oid': '9' * 64, 'size': None})

  def test_the_hello_carries_it(self):
    self.pick_here(SHA_D, 'Delta', ago=30)
    pick = self.models.hello_pick()
    self.assertEqual({k: pick[k] for k in ('ref', 'sha256', 'nbytes', 'name')},
                     {'ref': None, 'sha256': SHA_D, 'nbytes': 7, 'name': 'Delta'})

  def test_the_same_model_picked_later_on_the_server_is_not_another(self):
    # the server names it by the ref its catalog has; the hash says it is ours
    self.pick_here(SHA_D, 'Delta', ago=600)
    self.hear(REF_B, SHA_D, ago=60, nbytes=7)
    self.models.adopt()
    self.assertEqual(self.op.store[self.SLOT], {'sha256': SHA_D, 'displayName': 'Delta'})
    self.assertEqual(self.models.selected_model(), {'name': 'Delta', 'ref': None, 'oid': SHA_D, 'size': 7})

  def test_a_newer_pick_on_the_server_is_written_into_the_slot(self):
    self.pick_here(SHA_D, 'Delta', ago=600)
    self.hear(REF_B, '2' * 64, ago=60, nbytes=20, name='Beta')
    self.models.adopt()
    self.assertEqual(self.op.store[self.SLOT], {'displayName': 'Beta', 'ref': REF_B, 'sha256': '2' * 64})
    self.assertTrue(self.op.log.has('was picked on the server'))
    # from now on it is our pick, made just now
    (self.op.store_dir / self.SLOT).write_text('{}')
    self.assertIsNone(self.models._newer_server_pick())
    self.assertEqual(self.models.selected_model(), {'name': 'Beta', 'ref': REF_B, 'oid': '2' * 64, 'size': 20})

  def test_with_no_pick_here_the_servers_is_written_too(self):
    self.hear(None, '9' * 64, ago=6000, name='My export')
    self.models.adopt()
    self.assertEqual(self.op.store[self.SLOT], {'displayName': 'My export', 'sha256': '9' * 64})

  def test_an_older_pick_on_the_server_writes_nothing(self):
    self.pick_here(SHA_D, 'Delta', ago=60)
    self.hear(REF_B, '2' * 64, ago=600, nbytes=20)
    self.models.adopt()
    self.assertEqual(self.op.store[self.SLOT], {'sha256': SHA_D, 'displayName': 'Delta'})

  def test_a_server_pick_named_by_its_hash_alone_is_sized_here(self):
    self.hear(None, SHA_D, ago=60, name='Delta')
    self.assertEqual(self.models.selected_model(), {'name': 'Delta', 'ref': None, 'oid': SHA_D, 'size': 7})

  def test_it_is_provisioned_by_its_hash(self):
    from jetlink.openpilot import provision
    self.pick_here(SHA_D, 'Delta', ago=60)
    client = mock.Mock()
    client.hello.return_value = {}
    client.ensure_engine.return_value = mock.Mock(sha256=SHA_D, to_dict=lambda: {'sha256': SHA_D})
    run = provision.ProvisioningRun(self.parts)
    run.client = client
    self.assertTrue(run.provision())
    self.assertEqual(client.ensure_engine.call_args.args[:2], (SHA_D, 7))
    # the server downloads it by the hash and size, with no ref to go by
    self.assertTrue(client.ensure_engine.call_args.kwargs['fetch'])
    self.assertIsNone(client.ensure_engine.call_args.kwargs['ref'])
    self.assertTrue(self.parts.spec.engine_ready_for(SHA_D))


class TestWithoutAModelManager(ModelsTest):
  """A fork without sunnypilot's model manager: no pick, no catalog. The
  large model is jetlink's default, found by its ref like any pick."""

  def setUp(self):
    super().setUp()
    self.op.keys = dataclasses.replace(self.op.keys, catalog=None, big_model=None)

  def test_a_pick_on_the_server_runs_instead(self):
    # a fork with no picker of its own follows the server's
    self.parts.models.heard({'pick': {'ref': REF_B, 'sha256': '2' * 64, 'nbytes': 20, 'name': 'Beta', 'age_s': 10}})
    self.assertEqual(self.models.selected_model(), {'name': 'Beta', 'ref': REF_B, 'oid': '2' * 64, 'size': 20})

  def test_jetlinks_default_runs(self):
    self.assertEqual(self.models.selected_model(),
                     {'name': DEFAULT_BIG_MODEL_NAME, 'ref': DEFAULT_BIG_MODEL_REF, 'oid': None, 'size': None})
    self.assertEqual(self.models.default_model_name(), 'Cinque Terre V3 Model')
    self.assertEqual(self.jl.status().model, 'Cinque Terre V3 Model')

  def test_with_an_index_it_carries_its_identity(self):
    self.index(BETA, entry(DEFAULT_BIG_MODEL_REF, 'Cinque Terre V3', '9' * 64, 766_000_000), default=DEFAULT_BIG_MODEL_REF)
    self.assertEqual(self.models.selected_model()['oid'], '9' * 64)
    self.assertEqual(self.models.selected_model()['size'], 766_000_000)

  def test_it_is_provisioned_like_any_pick(self):
    from jetlink.openpilot import link, provision
    client = mock.Mock()
    client.hello.return_value = {}
    client.ensure_engine.return_value = mock.Mock(sha256='9' * 64, to_dict=lambda: {'sha256': '9' * 64})
    run = provision.ProvisioningRun(self.parts)
    run.client = client
    # the comma has no index yet: the run fetches it
    with self.fetched(entry(DEFAULT_BIG_MODEL_REF, 'Cinque Terre V3', '9' * 64, 766_000_000)) as fetch:
      self.assertTrue(run.provision())
    fetch.assert_called_once_with()
    self.assertEqual(client.ensure_engine.call_args.args[:2], ('9' * 64, 766_000_000))
    self.assertTrue(self.parts.spec.engine_ready_for('9' * 64))
    # neither end has the bytes, so the server is asked to download them, with the time that takes
    self.assertTrue(client.ensure_engine.call_args.kwargs['fetch'])
    self.assertEqual(client.ensure_engine.call_args.kwargs['build_timeout'], link.BUILD_TIMEOUT + link.FETCH_TIMEOUT)

  def test_a_model_manager_with_nothing_picked_runs_the_default_too(self):
    # an empty slot is the default picked, whether or not the index is here yet
    self.op.keys = dataclasses.replace(self.op.keys, catalog='ModelManager_ModelsCache_Chestnut')
    self.assertEqual(self.models.selected_model()['ref'], DEFAULT_BIG_MODEL_REF)


class TestDefaultModelName(ModelsTest):
  """The accelerator's default, named as the chestnut's DEFAULT_BIG_MODEL is: the
  index's name without its build date."""

  def name_with(self, names, default=REF_B):
    if names:
      self.index(*(entry(ref, name, ch * 64, 1) for name, ref, ch in zip(names, (REF_A, REF_B), '12', strict=False)), default=default)
    return self.models.default_model_name()

  def test_the_build_date_is_dropped(self):
    assert self.name_with(['Alpha', 'Cinque Terre V3 Model (September 17, 2026)']) == 'Cinque Terre V3 Model'

  def test_a_parenthesis_that_is_not_a_date_stays(self):
    assert self.name_with(['Alpha', 'Beta (big)']) == 'Beta (big)'

  def test_a_default_not_listed_names_the_first_listed(self):
    assert self.name_with(['Alpha (September 01, 2026)', 'Beta'], default='f' * 40) == 'Alpha'

  def test_no_index_yet_names_jetlinks_default(self):
    assert self.name_with([]) == DEFAULT_BIG_MODEL_NAME


class TestSelectedSlot(ModelsTest):
  def read_with(self, slot):
    self.models._slot_cache = None
    self.op.store['ModelManager_ActiveBundleChestnut'] = slot
    return self.models.selected_slot()

  def test_a_second_read_within_the_ttl_costs_nothing(self):
    # the UI names the active model every frame
    self.op.store['ModelManager_ActiveBundleChestnut'] = {'ref': REF_A}
    with mock.patch.object(self.op, 'get', wraps=self.op.get) as read:
      assert self.models.selected_slot()['ref'] == REF_A
      assert self.models.selected_slot()['ref'] == REF_A
    assert read.call_count == 1

  def test_reads_the_slots_ref_and_name(self):
    assert self.read_with({'ref': REF_A, 'displayName': 'Alpha'}) == {'name': 'Alpha', 'ref': REF_A, 'sha256': None}
    assert self.read_with({'ref': REF_A}) == {'name': REF_A[:10], 'ref': REF_A, 'sha256': None}

  def test_a_pick_named_by_its_hash(self):
    # StarPilot's catalog knows a model by its ONNX's hash alone
    assert self.read_with({'sha256': SHA_D, 'displayName': 'Delta'}) == {'name': 'Delta', 'ref': None, 'sha256': SHA_D}
    assert self.read_with({'sha256': SHA_D}) == {'name': SHA_D[:10], 'ref': None, 'sha256': SHA_D}

  def test_anything_else_is_no_pick(self):
    for slot in (None, {}, {'ref': ''}, {'ref': 7}, {'sha256': 'D' * 64}, {'sha256': 'd' * 40}, 'junk'):
      assert self.read_with(slot) is None, slot

  def test_the_slot_is_the_key_the_adapter_names(self):
    self.op.keys = dataclasses.replace(self.op.keys, big_model='SomeOtherForksSlot')
    self.op.store['SomeOtherForksSlot'] = {'ref': REF_B}
    assert self.read_with({'ref': REF_A}) == {'name': REF_B[:10], 'ref': REF_B, 'sha256': None}


class TestShippedModelPath(ModelsTest):
  """A file counts only when it is the model we mean, at the size we expect.
  Models live one file per oid so switching back does not re-download."""

  MODEL = {'name': 'Alpha', 'ref': REF_A, 'oid': 'a' * 64, 'size': 4096}

  def setUp(self):
    super().setUp()
    self.patch(self.models, 'selected_model', return_value=self.MODEL)

  def fetched(self, size: int):
    path = self.models.model_dir() / self.models.model_file_name(self.MODEL)
    path.parent.mkdir(parents=True)
    path.write_bytes(b'\0' * size)
    return path

  def test_the_downloads_have_a_directory_of_their_own(self):
    # the model manager's cache clear deletes every file in its root that it
    # does not recognise, and leaves directories alone
    self.assertEqual(self.models.model_dir(), self.op.model_root() / 'jetlink')

  def test_a_download_lands_in_it_never_in_the_root(self):
    from jetlink.registry.lfs import Pointer
    with mock.patch('jetlink.registry.lfs.lfs_resolve', return_value='https://x/y'), \
         mock.patch('jetlink.registry.lfs.lfs_download', side_effect=lambda href, pointer, dest, **kw: dest) as download:
      dest = self.models.fetch_shipped_model()
    self.assertEqual(dest.parent, self.op.model_root() / 'jetlink')
    self.assertEqual(download.call_args.args[1], Pointer('a' * 64, 4096))

  def test_a_model_not_resolved_yet_is_no_path(self):
    with mock.patch.object(self.models, 'selected_model', return_value={**self.MODEL, 'oid': None, 'size': None}):
      assert self.models.shipped_model_path() is None

  def test_nothing_fetched_yet(self):
    assert self.models.shipped_model_path() is None

  def test_the_chosen_model_is_accepted(self):
    fetched = self.fetched(4096)
    assert self.models.shipped_model_path() == fetched

  def test_a_truncated_download_is_rejected(self):
    # Half a model is exactly what must not reach TensorRT.
    self.fetched(2048)
    assert self.models.shipped_model_path() is None

  def test_no_model_chosen_is_no_path(self):
    with mock.patch.object(self.models, 'selected_model', return_value=None):
      assert self.models.shipped_model_path() is None


class TestFetchShippedModel(ModelsTest):
  """The bytes come through jetlink's registry downloader; all the checkout adds
  is the LFS server its own .lfsconfig names, asked first."""

  MODEL = {'name': 'Alpha', 'ref': REF_A, 'oid': 'a' * 64, 'size': 4096}

  def setUp(self):
    super().setUp()
    self.patch(self.models, 'selected_model', return_value=self.MODEL)

  def lfsconfig(self, text: str) -> None:
    (self.op.basedir / '.lfsconfig').write_text(text)

  def test_the_checkouts_server_comes_first(self):
    from jetlink.registry.lfs import LFS_ENDPOINTS
    self.lfsconfig('[lfs]\n\turl = https://example.com/info/lfs/\n')
    self.assertEqual(self.models.lfs_endpoints(), ['https://example.com/info/lfs', *LFS_ENDPOINTS])

  def test_without_one_it_is_jetlinks_list(self):
    # a release ships no .lfsconfig
    from jetlink.registry.lfs import LFS_ENDPOINTS
    self.assertEqual(self.models.lfs_endpoints(), list(LFS_ENDPOINTS))

  def test_no_duplicate_when_it_is_already_one_of_jetlinks(self):
    from jetlink.registry.lfs import LFS_ENDPOINTS
    self.lfsconfig(f'[lfs]\n\turl = {LFS_ENDPOINTS[0]}\n')
    self.assertEqual(self.models.lfs_endpoints(), list(LFS_ENDPOINTS))

  def test_already_fetched_is_returned_as_is(self):
    dest = self.models.model_dir() / self.models.model_file_name(self.MODEL)
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b'\0' * 4096)
    with mock.patch('jetlink.registry.lfs.lfs_resolve') as resolve:
      self.assertEqual(self.models.fetch_shipped_model(), dest)
    resolve.assert_not_called()

  def test_falls_through_to_the_next_server(self):
    from jetlink.registry.lfs import LFS_ENDPOINTS, Pointer
    self.lfsconfig('[lfs]\n\turl = https://dead.example/info/lfs\n')
    stop = object()
    with mock.patch('jetlink.registry.lfs.lfs_resolve', side_effect=[None, 'https://x/y']) as resolve, \
         mock.patch('jetlink.registry.lfs.lfs_download', side_effect=lambda href, pointer, dest, **kw: dest) as download:
      dest = self.models.fetch_shipped_model(should_stop=stop)
    self.assertEqual([c.args[0] for c in resolve.call_args_list], ['https://dead.example/info/lfs', LFS_ENDPOINTS[0]])
    self.assertEqual(download.call_args.args[:3], ('https://x/y', Pointer('a' * 64, 4096), dest))
    self.assertIs(download.call_args.kwargs['should_stop'], stop)

  def no_waits(self):
    from jetlink.openpilot import models
    return mock.patch.object(models, 'DOWNLOAD_RETRY_DELAYS', (0.0,))

  def test_nowhere_to_get_it_raises_once_the_attempts_are_spent(self):
    from jetlink.registry.index import NetworkError
    with mock.patch('jetlink.registry.lfs.lfs_resolve', return_value=None) as resolve, self.no_waits(), \
         self.assertRaises(NetworkError):
      self.models.fetch_shipped_model()
    # the one wait (no_waits) between two attempts
    self.assertEqual(resolve.call_count, 2 * len(self.models.lfs_endpoints()))

  def test_a_failed_transfer_is_tried_again_and_told_of(self):
    # a car's connection stalls; each attempt asks the LFS server for a fresh
    # address, which expires, and carries on from the .part (lfs_download)
    from jetlink.registry.index import NetworkError
    waits = []
    transfers = iter([NetworkError('read timed out'), None])

    def download(href, pointer, dest, **kw):
      if (e := next(transfers)) is not None:
        raise e
      return dest

    with mock.patch('jetlink.registry.lfs.lfs_resolve', return_value='https://x/y') as resolve, \
         mock.patch('jetlink.registry.lfs.lfs_download', side_effect=download), self.no_waits():
      dest = self.models.fetch_shipped_model(retrying=lambda e, delay: waits.append((str(e), delay)))
    self.assertEqual(dest.name, self.models.model_file_name(self.MODEL))
    self.assertEqual(resolve.call_count, 2)
    self.assertEqual(waits, [('read timed out', 0.0)])

  def test_a_stopped_run_stops_waiting_to_try_again(self):
    from jetlink.openpilot import models
    from jetlink.registry.index import NetworkError
    with mock.patch('jetlink.registry.lfs.lfs_resolve', return_value=None), \
         mock.patch.object(models, 'DOWNLOAD_RETRY_DELAYS', (60.0,)), self.assertRaises(NetworkError):
      self.models.fetch_shipped_model(should_stop=lambda: True)

  def test_a_model_with_its_file_here_has_it(self):
    self.assertFalse(self.models.has_file(self.MODEL))
    dest = self.models.model_dir() / self.models.model_file_name(self.MODEL)
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b'\0' * 100)
    self.assertFalse(self.models.has_file(self.MODEL), 'a short file is not the model')
    dest.write_bytes(b'\0' * 4096)
    self.assertTrue(self.models.has_file(self.MODEL))
    self.assertFalse(self.models.has_file({**self.MODEL, 'oid': None}))

  def test_nothing_chosen_is_nothing_fetched(self):
    with mock.patch.object(self.models, 'selected_model', return_value={**self.MODEL, 'oid': None}):
      self.assertIsNone(self.models.fetch_shipped_model())


class TestBigCatalog(ModelsTest):
  """Without a chestnut fitted the model manager's big-model list is the
  index's, so what the comma lists is what the Jetson runs. That the model
  manager parses it is the fork's to test."""

  PR = entry(REF_C, '#39235 Light speed 2', '3' * 64, 5, folder='comma Pull Requests', pr=39235)

  def test_the_index_in_its_order_and_folders(self):
    listed = entry(REF_B, 'Beta (October 29, 2026)', '2' * 64, 20, folder='2026 Deep RL Models', short_name='B',
                   build_time='2026-10-29T00:00:00Z', is_20hz=True, generation='13', environment='development', runner='tinygrad')
    with self.fetched(listed, ALPHA, self.PR):
      out = self.jl.big_catalog(19)
    self.assertEqual([(b['ref'], b['index'], b['overrides']['folder']) for b in out['bundles']],
                     [(REF_B, 3, '2026 Deep RL Models'), (REF_A, 2, 'Master Models'), (REF_C, 1, 'comma Pull Requests')])
    first = out['bundles'][0]
    # the index's fields, the fork's selector, and nothing for a chestnut to fetch
    self.assertEqual({k: first[k] for k in ('display_name', 'short_name', 'build_time', 'generation', 'environment', 'runner',
                                            'is_20hz', 'minimum_selector_version', 'is_big', 'models')},
                     {'display_name': 'Beta (October 29, 2026)', 'short_name': 'B', 'build_time': '2026-10-29T00:00:00Z',
                      'generation': '13', 'environment': 'development', 'runner': 'tinygrad', 'is_20hz': True,
                      'minimum_selector_version': '19', 'is_big': True, 'models': []})
    # what the model manager's parser requires of every bundle
    for b in out['bundles']:
      self.assertTrue({'index', 'short_name', 'display_name', 'generation', 'environment', 'minimum_selector_version'} <= set(b))
    # an entry from an index older than its runtime fields gets sunnypilot's
    self.assertEqual({k: out['bundles'][1][k] for k in ('generation', 'environment', 'runner')}, RUNTIME)

  def test_the_selector_is_the_adapters(self):
    with self.fetched(ALPHA):
      self.assertEqual(self.jl.big_catalog(21)['bundles'][0]['minimum_selector_version'], '21')

  def test_the_index_it_lists_is_the_one_the_comma_keeps(self):
    with self.fetched(ALPHA, BETA):
      self.jl.big_catalog(19)
    self.assertEqual([m['ref'] for m in self.models.model_index()], [REF_A, REF_B])

  def test_a_failed_fetch_is_none_and_keeps_the_copy(self):
    self.index(ALPHA)
    with mock.patch('jetlink.registry.index.fetch_index', side_effect=NetworkError('offline')):
      self.assertIsNone(self.jl.big_catalog(19))
    self.assertEqual([m['ref'] for m in self.models.model_index()], [REF_A])
    self.assertTrue(self.op.log.has('could not fetch the model index: offline', 'warning'))
    self.assertEqual(self.op.log.lines('exception'), [])

  def test_whether_it_lists_is_the_hardware(self):
    self.assertTrue(self.jl.lists_big_models())
    self.op.chestnut = True
    self.parts._chestnut = None
    with mock.patch.object(self.parts, 'chestnut_fitted', return_value=True):
      self.assertFalse(self.jl.lists_big_models())

  def test_a_pick_the_list_no_longer_names_carries_over_to_its_onnx(self):
    # a recompile of a listed model is listed once, under the newest ref
    self.index(BETA, refs={REF_C: {'oid': BETA['oid'], 'size': BETA['size']}, REF_A: {'oid': '9' * 64, 'size': 1}})
    self.assertEqual(self.jl.listed_ref(REF_C), REF_B)
    self.assertEqual(self.jl.listed_ref(REF_B), REF_B)
    self.assertIsNone(self.jl.listed_ref(REF_A))
    self.assertIsNone(self.jl.listed_ref('f' * 40))
    self.assertIsNone(self.jl.listed_ref('not a ref'))
    # beside a chestnut the list is sunnypilot's, and jetlink names nothing in it
    with mock.patch.object(self.parts, 'chestnut_fitted', return_value=True):
      self.assertIsNone(self.jl.listed_ref(REF_C))

  def test_api_3_leaves_the_catalog_as_fetched(self):
    # its picks still drive: the index knows every ref it ever saw
    pinned = {'bundles': [bundle(REF_A, 'Alpha', 1)]}
    self.assertFalse(self.jl.should_extend_catalog())
    self.assertIs(self.jl.extend_catalog(pinned), pinned)


if __name__ == '__main__':
  unittest.main()
