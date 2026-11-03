import argparse
from src.common import device_for, load_config, seed_everything
from src.embeddings import embed_items


def main():
    parser = argparse.ArgumentParser(description="Cache frozen Sentence-T5 item embeddings")
    parser.add_argument("--config", default="configs/small.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"], config["threads"])
    embed_items(config, device_for(config))


if __name__ == "__main__":
    main()
