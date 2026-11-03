import pytest
from scripts.evaluate import load_predictions
from src.baselines import popularity_ranking
from src.common import write_jsonl
from src.metrics import evaluate_rankings


@pytest.mark.parametrize("rank,recall,ndcg", [(1, 1, 1), (3, 1, 0.5), (None, 0, 0)])
def test_hand_calculated_metrics(rank, recall, ndcg):
    predictions = ["a", "b", "c"]
    target = predictions[rank - 1] if rank else "absent"
    result = evaluate_rankings({"u": target}, {"u": predictions})
    assert result["Recall@5"] == recall
    assert result["NDCG@5"] == ndcg


def test_all_users_duplicates_short_outputs_and_cutoffs():
    output = evaluate_rankings({"u": "a", "missing": "b"}, {"u": ["x", "x", "a"]})
    assert output["Recall@10"] == 0.5
    assert output["NDCG@5"] == pytest.approx(0.5 / 1.584962500721156)
    assert output["coverage@5"] == 0
    assert output["mean_returned_items"] == 1
    result = evaluate_rankings({"u": "5"}, {"u": list(map(str, range(10)))})
    assert result["Recall@5"] == 0 and result["Recall@10"] == 1
    assert result["coverage@10"] == 1


def test_popularity_counts_original_training_events_only():
    splits = [{"train_items": ["b", "b", "a"], "val_item": "z", "test_item": "z"},
              {"train_items": ["c"], "val_item": "z", "test_item": "z"}]
    assert popularity_ranking(splits, ["z", "c", "a", "b"]) == ["b", "a", "c", "z"]


@pytest.mark.parametrize("rows", [
    [{"user_id": "unknown", "item_ids": []}],
    [{"user_id": "u", "item_ids": ["unknown"]}],
    [{"user_id": "u", "item_ids": []}, {"user_id": "u", "item_ids": []}],
    [{"user_id": "u", "item_ids": "a"}],
])
def test_prediction_validation(tmp_path, rows):
    path = tmp_path / "predictions.jsonl"
    write_jsonl(path, rows)
    with pytest.raises(ValueError):
        load_predictions(path, {"u"}, {"a"})


def test_prediction_deduplication(tmp_path):
    path = tmp_path / "predictions.jsonl"
    write_jsonl(path, [{"user_id": "u", "item_ids": ["b", "a", "b"]}])
    assert load_predictions(path, {"u"}, {"a", "b"}) == {"u": ["b", "a"]}
