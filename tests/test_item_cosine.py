import math
import pytest
from src.baselines import ItemCosine


def test_cosine_counts_users_not_repeated_events_and_ignores_holdouts():
    rows = [{'train_items': ['a', 'a', 'b'], 'val_item': 'c', 'test_item': 'c'},
            {'train_items': ['a', 'c'], 'val_item': 'b', 'test_item': 'b'}]
    model = ItemCosine(rows, ['a', 'b', 'c', 'z'])
    assert model.similarity['a']['b'] == pytest.approx(1 / math.sqrt(2))
    assert model.similarity['b'].get('c', 0) == 0
    assert model.similarity['a'].get('a', 0) == 0
    assert model.recommend(['b']) == ['a', 'b', 'c', 'z']
    changed = [{**row, 'val_item': 'z', 'test_item': 'z'} for row in rows]
    assert ItemCosine(changed, model.catalog).similarity == model.similarity


def test_history_truncation_deduplication_repeats_and_fallback():
    model = ItemCosine([{'train_items': ['a', 'b']}, {'train_items': ['b', 'c']}], ['z', 'c', 'b', 'a'])
    assert model.recommend(['a', 'a']) == model.recommend(['a'])
    assert model.recommend(['c'] + ['a'] * 20) == model.recommend(['a'])
    assert model.recommend([]) == ['b', 'a', 'c', 'z']
    assert set(model.recommend(['a', 'b'])) == {'a', 'b', 'c', 'z'}
    with pytest.raises(ValueError):
        ItemCosine([{'train_items': ['absent']}], ['a'])
