import json
import numpy as np
import pytest
from src.common import canonical_hash, file_hash, write_json
from src.embeddings import embedding_identity, load_embeddings, validate_embeddings
from src.semantic_ids import finalize_ids, make_manifest, offset_codes, user_token, validate_maps


def test_collision_suffixes_roundtrip_offsets_and_hashes():
    ids = ["b", "a", "c"]
    forward, inverse = finalize_ids(ids, [[7, 7, 3], [7, 7, 3], [0, 1, 2]])
    assert forward["a"] == [7, 7, 3, 0]
    assert forward["b"] == [7, 7, 3, 1]
    assert forward["c"][-1] == 0
    manifest = make_manifest(sorted(ids), forward, 42)
    validate_maps(forward, inverse, manifest)
    assert offset_codes(forward["a"], manifest) == [9, 265, 517, 770]
    assert manifest["offsets"] == [2, 258, 514, 770]
    assert manifest["user_offset"] == 1026 and manifest["vocabulary_size"] == 3026
    assert 1026 <= user_token("user", manifest) < 3026
    assert canonical_hash(forward) == canonical_hash(json.loads(json.dumps(forward, sort_keys=True)))
    assert finalize_ids(list(reversed(ids)), [[0, 1, 2], [7, 7, 3], [7, 7, 3]]) == (forward, inverse)
    with pytest.raises(ValueError, match="Collision"):
        finalize_ids(["a", "b"], [[0, 0, 0], [0, 0, 0]], suffix_size=1)
    inverse["7,7,3,0"] = "wrong"
    with pytest.raises(ValueError):
        validate_maps(forward, inverse, manifest)


@pytest.mark.parametrize("values", [np.zeros((2, 3), np.float32), np.zeros((2, 768), np.float64), np.full((2, 768), np.nan, np.float32)])
def test_embedding_shape_finiteness_and_dtype(values):
    with pytest.raises(ValueError):
        validate_embeddings(["a", "b"], values)


def test_cache_order_and_fingerprint(tmp_path):
    ids = ["a", "b"]
    np.save(tmp_path / "item_embeddings.npy", np.zeros((2, 768), np.float32))
    write_json(tmp_path / "item_ids.json", ids)
    write_json(tmp_path / "embedding_manifest.json", {"identity": {"item_ids": ids}, "array_sha256": file_hash(tmp_path / "item_embeddings.npy")})
    assert load_embeddings(tmp_path)[0] == ids
    write_json(tmp_path / "item_ids.json", list(reversed(ids)))
    with pytest.raises(ValueError, match="order"):
        load_embeddings(tmp_path)
    config = {"embedding": {"model": "test", "dimension": 768}}
    first = embedding_identity([{"item_id": "a", "text": "A"}], config, "revision1")
    second = embedding_identity([{"item_id": "a", "text": "B"}], config, "revision1")
    assert first != second
