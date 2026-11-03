import copy
import random
import signal

import numpy as np
import pytest
import torch

from scripts.train_recommender import parser
from src import training
from src.common import file_hash, load_config, read_jsonl
from src.recommender import load_checkpoint
from src.training_state import StopRequest, atomic_save, load_recovery


def configuration():
    config = copy.deepcopy(load_config())
    config["recommender"].update(max_steps=7, eval_every=3, batch_size=8, val_subset=2,
                                 save_every=2, d_model=32, d_ff=64, d_kv=8,
                                 num_layers=1, num_decoder_layers=1, dropout=0.2)
    return config


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("stop_at", [2, 3, 4])
def test_exact_resume_with_dropout_and_epoch_validation_boundaries(tmp_path, monkeypatch, stop_at):
    config = configuration()
    sections = {key: config[key] for key in ("sequence", "recommender", "retrieval")}
    original_step = training.train_step

    def run(directory, resume=False, interrupt=None):
        args = parser().parse_args(["--data-dir", "tests/fixtures", "--sid-dir", "tests/fixtures",
                                    "--out", str(directory)] +
                                   (["--resume", str(directory / "latest.pt")] if resume else []))
        stop, events = StopRequest(), []
        def step(model, batch, optimizer, device):
            draws = (random.random(), float(np.random.random()))
            loss = original_step(model, batch, optimizer, device)
            events.append((batch["index"].tolist(), draws, loss))
            if len(events) == interrupt:
                stop.request(signal.SIGINT)
            return loss
        monkeypatch.setattr(training, "train_step", step)
        result = training.run_training(config, args, sections, stop)
        return result, events

    uninterrupted, expected_events = run(tmp_path / "full")
    paused, first_events = run(tmp_path / "resumed", interrupt=stop_at)
    assert paused["status"] == "interrupted" and paused["steps"] == stop_at
    # Simulate log lines and a best export written beyond the authoritative snapshot.
    with (tmp_path / "resumed/train_log.jsonl").open("a") as stream:
        stream.write('{"step": 999}\npartial interrupted write')
    best_dir = tmp_path / "resumed/best"
    best_dir.mkdir(exist_ok=True)
    (best_dir / "stale.txt").write_text("stale export")
    resumed, remaining_events = run(tmp_path / "resumed", resume=True)
    assert resumed["status"] == "completed" and resumed["steps"] == 7
    assert expected_events == first_events + remaining_events
    full_state = torch.load(tmp_path / "full/latest.pt", weights_only=True)
    resumed_state = torch.load(tmp_path / "resumed/latest.pt", weights_only=True)
    for key in ("model", "optimizer", "rng", "step", "epoch", "next_batch"):
        assert_equal(full_state[key], resumed_state[key])
    assert uninterrupted["best_step"] == resumed["best_step"]
    full_model, _ = load_checkpoint(tmp_path / "full/best")
    resumed_model, _ = load_checkpoint(tmp_path / "resumed/best")
    assert_equal(full_model.state_dict(), resumed_model.state_dict())
    strip_time = lambda rows: [{k: v for k, v in row.items() if k != "elapsed_s"} for row in rows]
    assert strip_time(read_jsonl(tmp_path / "full/train_log.jsonl")) == strip_time(read_jsonl(tmp_path / "resumed/train_log.jsonl"))
    _, no_events = run(tmp_path / "resumed", resume=True)
    assert no_events == []
    changed = copy.deepcopy(sections)
    changed["recommender"]["batch_size"] = 4
    args = parser().parse_args(["--data-dir", "tests/fixtures", "--sid-dir", "tests/fixtures",
                                "--resume", str(tmp_path / "resumed/latest.pt")])
    with pytest.raises(ValueError, match="configuration, environment, or input"):
        training.run_training(config, args, changed)
    with pytest.raises(ValueError, match="nonempty"):
        run(tmp_path / "full")


def test_atomic_recovery_keeps_prior_snapshot_after_failed_write(tmp_path, monkeypatch):
    path = tmp_path / "latest.pt"
    state = {"version": 1, "identity": {"mapping_hash": "A"}, "step": 2}
    atomic_save(path, state)
    before = file_hash(path)
    def broken_save(state, temporary):
        temporary.write_bytes(b"incomplete checkpoint")
        raise OSError("disk write interrupted")
    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError):
        atomic_save(path, {**state, "step": 3})
    assert file_hash(path) == before
    assert load_recovery(path, state["identity"])["step"] == 2
    with pytest.raises(ValueError, match="hashes differ"):
        load_recovery(path, {"mapping_hash": "B"})


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_cli_signal_saves_after_completed_step(tmp_path, monkeypatch, signum):
    import os
    import sys
    from scripts import train_recommender
    config = configuration()
    monkeypatch.setattr(train_recommender, "load_config", lambda *args: config)
    monkeypatch.setattr(sys, "argv", ["train_recommender", "--data-dir", "tests/fixtures",
                                    "--sid-dir", "tests/fixtures", "--out", str(tmp_path)])
    original = training.train_step
    def interrupted(model, batch, optimizer, device):
        result = original(model, batch, optimizer, device)
        os.kill(os.getpid(), signum)
        return result
    monkeypatch.setattr(training, "train_step", interrupted)
    with pytest.raises(SystemExit) as exit:
        train_recommender.main()
    assert exit.value.code == 128 + signum
    assert torch.load(tmp_path / "latest.pt", weights_only=True)["step"] == 1
