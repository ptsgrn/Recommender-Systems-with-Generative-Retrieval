"""Evaluate two baselines and compare with frozen, previously measured TIGER scores."""
import argparse
from pathlib import Path

from src.baselines import ItemCosine, popularity_ranking
from src.common import canonical_hash, file_hash, load_config, read_json, read_jsonl, write_json, write_jsonl
from src.dataset import build_eval_examples
from src.metrics import evaluate_rankings


def run(config, output, predictions_dir):
    output, predictions_dir = Path(output), Path(predictions_dir)
    if output.exists() or predictions_dir.exists():
        raise ValueError('Use fresh baseline report and prediction directories')
    data = Path(config['data_dir'])
    splits = read_jsonl(data / 'splits.jsonl')
    catalog = sorted(row['item_id'] for row in read_jsonl(data / 'items.jsonl'))
    identity = {'catalog_hash': canonical_hash(catalog), 'split_sha256': file_hash(data / 'splits.jsonl')}
    frozen = {split: read_json(Path(config['report_dir']) / f'results_{split}.json') for split in ('val', 'test')}
    for report in frozen.values():
        if any(report[key] != value for key, value in identity.items()):
            raise ValueError('Frozen model results use different data')
    write_json(output / 'protocol.json', {
        **identity, 'method': 'item_cosine', 'fit': 'binary user-item membership in train_items only',
        'similarity': 'shared training users / sqrt(user_count_i * user_count_j); diagonal zero',
        'score': 'sum over distinct items in latest 20 observed history events',
        'neighbors': 'all; no tuned hyperparameters',
        'ties': 'descending original training-event popularity, ascending item ID',
        'candidate_policy': 'full catalog; previously observed items allowed; no sampled negatives',
        'selection': 'fixed before baseline evaluation; no validation/test tuning',
        'model_results': 'copied from frozen poc01 reports; no model generation or retraining',
        'source_sha256': file_hash('src/baselines.py'),
    })
    model = ItemCosine(splits, catalog)
    popular = popularity_ranking(splits, catalog)[:10]
    for split in ('val', 'test'):
        examples = build_eval_examples(splits, split)
        targets = {e.user_id: e.target for e in examples}
        predictions = {e.user_id: model.recommend(e.history) for e in examples}
        path = predictions_dir / f'item_cosine_{split}.jsonl'
        write_jsonl(path, [{'user_id': user, 'item_ids': items} for user, items in predictions.items()])
        popularity = evaluate_rankings(targets, {user: popular for user in targets})
        assert popularity == frozen[split]['popularity']
        result = {**identity, 'split': split, 'catalog_size': len(catalog),
                  'popularity': popularity, 'item_cosine': evaluate_rankings(targets, predictions),
                  'model': frozen[split]['model'], 'model_source': f"{config['report_dir']}/results_{split}.json",
                  'model_source_sha256': file_hash(Path(config['report_dir']) / f'results_{split}.json'),
                  'baseline_predictions_sha256': file_hash(path)}
        write_json(output / f'results_{split}.json', result)
        lines = ['| Method | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |', '|---|---:|---:|---:|---:|']
        for name in ('popularity', 'item_cosine', 'model'):
            lines.append('| ' + name + ' | ' + ' | '.join(f'{result[name][key]:.8f}' for key in
                         ('Recall@5', 'Recall@10', 'NDCG@5', 'NDCG@10')) + ' |')
        (output / f'results_{split}.md').write_text('\n'.join(lines) + '\n')
        print(split, result['item_cosine'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/poc01.yaml')
    parser.add_argument('--out', default='reports/baselines')
    parser.add_argument('--predictions-dir', default='artifacts/baselines')
    args = parser.parse_args()
    run(load_config(args.config), args.out, args.predictions_dir)
