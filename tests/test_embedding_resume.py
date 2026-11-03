import numpy as np
import pytest
from src import embeddings


def test_interruption_resumes_completed_chunks_and_rejects_stale_identity(tmp_path, monkeypatch):
    items = [{"item_id": str(i), "text": str(i)} for i in range(260)]
    config = {"embedding": {"model": "fixture", "dimension": 2, "batch_size": 16}}
    identity = embeddings.embedding_identity(items, config, "revision")
    calls = []
    def interrupted(model, texts, batch_size):
        calls.append(texts)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return np.array([[float(text), 1] for text in texts], dtype=np.float32)
    monkeypatch.setattr(embeddings, "encode", interrupted)
    with pytest.raises(KeyboardInterrupt):
        embeddings.encode_checkpointed(None, items, config, identity, tmp_path)
    changed = {**identity, "revision": "different"}
    with pytest.raises(ValueError, match="Stale partial"):
        embeddings.encode_checkpointed(None, items, config, changed, tmp_path)
    resumed = []
    def complete(model, texts, batch_size):
        resumed.extend(texts)
        return np.array([[float(text), 1] for text in texts], dtype=np.float32)
    monkeypatch.setattr(embeddings, "encode", complete)
    values, seconds = embeddings.encode_checkpointed(None, items, config, identity, tmp_path)
    assert resumed == [str(i) for i in range(128, 260)]
    assert values.shape == (260, 2)
    np.testing.assert_array_equal(values[:, 0], np.arange(260))
    assert seconds >= 0
