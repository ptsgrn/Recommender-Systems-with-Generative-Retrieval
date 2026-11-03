"""Atomic recovery snapshots. Inference checkpoints keep their existing format."""
import copy
import random
from pathlib import Path

import numpy as np
import torch

from src.recommender import save_checkpoint


class StopRequest:
    def __init__(self):
        self.signal_number = 0

    def request(self, signum, frame=None):
        self.signal_number = signum


def capture_rng():
    state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def cpu_weights(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def atomic_save(path, state):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_recovery(path, identity):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("version") != 1 or state.get("identity") != identity:
        raise ValueError("Resume configuration, environment, or input hashes differ")
    return state


def export_best(model, best, directory):
    """Rebuild best/ from the authoritative recovery state, without consuming RNG."""
    directory = Path(directory)
    if best is None:
        if directory.exists():
            # Only remove files produced by our own inference exporter.
            import shutil
            shutil.rmtree(directory)
        return
    rng = capture_rng()
    try:
        shadow = copy.deepcopy(model).cpu()
        shadow.load_state_dict(best["weights"])
        save_checkpoint(shadow, directory, best["meta"])
    finally:
        restore_rng(rng)
