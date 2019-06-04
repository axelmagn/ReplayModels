from .position_policy import ReplayConverter, FileCachedCalculated
from functools import lru_cache
import tensorflow as tf

EXAMPLE_REPLAY = "D4DE3A894229D5B8A5D0AE9091D3CA6C"

# TODO: delete
@lru_cache()
def _data_manager():
    return FileCachedCalculated()


@lru_cache()
def _example_replay():
    proto = _data_manager().get_proto(EXAMPLE_REPLAY)
    pd = _data_manager().get_pandas(EXAMPLE_REPLAY)
    return (proto, pd)

# TODO: delete


def test_data_manager():
    proto = _data_manager().get_proto(EXAMPLE_REPLAY)
    assert proto is not None
    df = _data_manager().get_pandas(EXAMPLE_REPLAY)
    assert df is not None


def test_ReplayConverter__get_replay_players():
    (proto, _pd) = _example_replay()
    entities = ReplayConverter()._get_replay_players(proto)
    expected = ([
            'CatNip',
            'Vipernite', 
            'Skenderovic_4',
        ], [
            'iNateQ',
            'Cruzz99_mlg',
            'oNoteless'
    ])
    assert entities == expected

def test_ReplayConverter__extract_entity_features():
    (_proto, pd) = _example_replay()
    entity_name = 'CatNip'
    label_prefix = 't0_p0'
    feature_gen = ReplayConverter()._extract_entity_features(entity_name, label_prefix, pd)
    for (label, feature) in feature_gen:
        assert isinstance(label, str)
        assert label[:5] == label_prefix
        assert isinstance(feature, tf.train.Feature)
