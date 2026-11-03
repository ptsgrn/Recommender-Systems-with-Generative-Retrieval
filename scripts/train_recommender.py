"""Train or exactly resume the sequence model; select best/ using validation only."""
import argparse
import signal

from src.common import load_config
from src.training import ranking_metrics, run_training
from src.training_state import StopRequest


def b_sections(config):
    defaults = load_config()
    return {key: config.get(key, defaults[key]) for key in ("sequence", "recommender", "retrieval")}


def parser():
    p = argparse.ArgumentParser(description="Train the TIGER sequence model")
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--data-dir")
    p.add_argument("--sid-dir")
    p.add_argument("--out", help="fresh output directory; inferred from --resume when omitted")
    p.add_argument("--overfit", action="store_true", help="dropout 0; select on fixture training examples")
    p.add_argument("--max-steps", type=int, help="total target, including completed steps on resume")
    p.add_argument("--eval-every", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--val-subset", type=int)
    p.add_argument("--benchmark-steps", type=int, default=0)
    p.add_argument("--save-every", type=int, help="recovery snapshot interval; default 100")
    p.add_argument("--resume", help="path to this run's latest.pt")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    config = load_config(args.config)
    stop = StopRequest()
    previous = {sig: signal.signal(sig, stop.request) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        result = run_training(config, args, b_sections(config), stop)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if result["status"] == "interrupted":
        raise SystemExit(128 + stop.signal_number)


if __name__ == "__main__":
    main()
