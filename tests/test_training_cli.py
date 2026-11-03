from pathlib import Path
import sys

from scripts import train_recommender
from src.common import load_config, read_json


def test_checkpoint_selection_uses_ndcg10_when_serving_top5(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/small.yaml")
    config["retrieval"]["top_k"] = 5
    config["recommender"].update(max_steps=1, eval_every=1, val_subset=2)
    monkeypatch.setattr(train_recommender, "load_config", lambda *args: config)
    monkeypatch.setattr(sys, "argv", ["train_recommender", "--data-dir", str(root / "tests/fixtures"),
                                    "--sid-dir", str(root / "tests/fixtures"), "--out", str(tmp_path)])
    train_recommender.main()
    meta = read_json(tmp_path / "best/run_meta.json")
    assert meta["sections"]["retrieval"]["top_k"] == 5
    assert "ndcg@10" in meta["metrics"]
    assert read_json(tmp_path / "summary.json")["best_step"] == 1
