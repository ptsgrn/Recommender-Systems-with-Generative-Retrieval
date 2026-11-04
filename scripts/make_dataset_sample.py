"""Export 100 real retained events and their matching cleaned product metadata."""
from pathlib import Path
from src.common import file_hash, read_jsonl, write_json, write_jsonl


def make_sample():
    source = Path('data/processed')
    output = Path('samples')
    events = []
    for row in read_jsonl(source / 'chronological.jsonl'):
        for index, event in enumerate(row['events']):
            split = 'test' if index == len(row['events']) - 1 else (
                'val' if index == len(row['events']) - 2 else 'train')
            events.append({'user_id': row['user_id'], **event, 'split': split})
            if len(events) == 100:
                break
        if len(events) == 100:
            break
    if len(events) != 100:
        raise ValueError('Need at least 100 retained events')
    ids = {event['item_id'] for event in events}
    items = [row for row in read_jsonl(source / 'items.jsonl') if row['item_id'] in ids]
    if {row['item_id'] for row in items} != ids:
        raise ValueError('Missing sample metadata')
    write_jsonl(output / 'interactions.jsonl', events)
    write_jsonl(output / 'items.jsonl', items)
    write_json(output / 'manifest.json', {
        'interaction_records': len(events), 'item_records': len(items),
        'selection': 'first 100 events in processed chronological.jsonl source order',
        'purpose': 'format inspection only; not the experimental subset',
        'source_sha256': {name: file_hash(source / name) for name in ('chronological.jsonl', 'items.jsonl')},
        'sample_sha256': {name: file_hash(output / name) for name in ('interactions.jsonl', 'items.jsonl')},
    })
    print(f'Exported {len(events)} real events and {len(items)} matching items to {output}')


if __name__ == '__main__':
    make_sample()
