import pytest
from scripts.inspect_example import verify_generation
from src.common import write_jsonl
from src.semantic_ids import finalize_ids, make_manifest, offset_codes


@pytest.mark.parametrize("problem", [None, "wrong_item", "wrong_position", "missing_eos", "missing_user", "duplicate_user", "nonfinite_score"])
def test_raw_generation_trace_checks_actual_beam(tmp_path, problem):
    forward, inverse = finalize_ids(["a", "b"], [[1, 2, 3], [4, 5, 6]])
    manifest = make_manifest(["a", "b"], forward, 42)
    tokens = offset_codes(forward["a"], manifest) + [1]
    if problem == "wrong_item":
        tokens = offset_codes(forward["b"], manifest) + [1]
    elif problem == "wrong_position":
        tokens[0], tokens[1] = tokens[1], tokens[0]
    elif problem == "missing_eos":
        tokens = tokens[:-1]
    row = {"user_id": "other" if problem == "missing_user" else "u",
           "beams": [{"tokens": tokens, "score": "invalid" if problem == "nonfinite_score" else -2.5}]}
    path = tmp_path / "generation.jsonl"
    write_jsonl(path, [row, row] if problem == "duplicate_user" else [row])
    if problem:
        with pytest.raises(ValueError):
            verify_generation(path, "u", "a", forward, inverse, manifest)
    else:
        verified = verify_generation(path, "u", "a", forward, inverse, manifest)
        assert verified["lookup_verified"] and verified["decoded_sid"] == forward["a"]
        assert verified["tokens_without_decoder_start"] == tokens
        with pytest.raises(ValueError, match="score differs"):
            verify_generation(path, "u", "a", forward, inverse, manifest, expected_score=-4)
