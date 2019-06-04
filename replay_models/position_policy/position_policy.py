import logging
import tensorflow as tf
import itertools


# TODO: delete
import glob
import gzip
import io
import os
import pandas as pd
import pickle
import carball
import requests
from carball.analysis.analysis_manager import PandasManager, AnalysisManager
from carball.analysis.utils.proto_manager import ProtobufManager
from carball.generated.api.game_pb2 import Game
BASE_URL = 'https://calculated.gg/api/v1/'


class ReplayGetter:
    """(TODO) Retrieves replays from calculated.gg"""

    def __init__(self, list_page_size=50):
        # list attributes
        self.list_page_size = list_page_size
        self.list_page = 1

# TODO delete


class DataManager:

    def get_replay_list(self, num: int = 50):
        raise NotImplementedError()

    def get_pandas(self, id_: str):
        raise NotImplementedError()

    def get_proto(self, id_: str):
        raise NotImplementedError()

# TODO(delete me)


class Calculated(DataManager):
    PANDAS_MAP = {}
    PROTO_MAP = {}
    BROKEN = []

    def get_replay_list(self, num=50, page=1):
        r = requests.get(
            BASE_URL + 'replays?key=1&minmmr=1300&maxmmr=1400&playlist=13&num={}&page={}'.format(num, page))
        return [replay['hash'] for replay in r.json()['data']]

    def get_pandas(self, id_):
        if id_ in self.BROKEN:
            return None
        if id_ in self.PANDAS_MAP:
            return self.PANDAS_MAP[id_]
        # url = BASE_URL + 'parsed/{}.replay.gzip?key=1'.format(id_)
        url = "https://storage.googleapis.com/calculatedgg-parsed/{GUID}.replay.gzip".format(
            GUID=id_)
        try:
            r = requests.get(url)
            gzip_file = gzip.GzipFile(fileobj=io.BytesIO(r.content), mode='rb')
            pandas_ = PandasManager.safe_read_pandas_to_memory(gzip_file)
        except:
            self.PANDAS_MAP[id_] = None
            self.BROKEN.append(id_)
            return None
        self.PANDAS_MAP[id_] = pandas_
        return pandas_

    def get_proto(self, id_):
        if id_ in self.PROTO_MAP:
            return self.PROTO_MAP[id_]
        url = BASE_URL + 'parsed/{}.replay.pts?key=1'.format(id_)
        r = requests.get(url)
        #     file_obj = io.BytesIO()
        #     for chunk in r.iter_content(chunk_size=1024):
        #         if chunk: # filter out keep-alive new chunks
        #             file_obj.write(chunk)
        proto = ProtobufManager.read_proto_out_from_file(io.BytesIO(r.content))
        self.PROTO_MAP[id_] = proto
        return proto


# TODO(delete me)
class FileCachedCalculated(DataManager):
    """
    Caches replays from a calculated data manager into pickle files.
    """

    DEFAULT_REPLAYS_CACHE_PATH = '.replays_cache'

    def __init__(self, cache_path=None):
        if cache_path is None:
            cache_path = os.path.join(
                os.environ['HOME'], self.DEFAULT_REPLAYS_CACHE_PATH)
        print("Caching replays at: {}".format(cache_path))
        self.cache_path = cache_path
        self.wrapped_manager = Calculated()

    def _setup_cache(self):
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

    def get_replay_list(self, num=50, page=1):
        return self.wrapped_manager.get_replay_list(num, page)

    def get_pandas(self, id_):
        self._setup_cache()
        pickle_path = os.path.join(
            self.cache_path, "{}.pandas.pkl.gz".format(id_))
        if os.path.isfile(pickle_path):
            return pd.read_pickle(pickle_path)
        else:
            result = self.wrapped_manager.get_pandas(id_)
            if result is not None:
                result.to_pickle(pickle_path)
            return result

    def get_proto(self, id_):
        self._setup_cache()
        pickle_path = os.path.join(self.cache_path, "{}.proto".format(id_))
        if os.path.isfile(pickle_path):
            with open(pickle_path, 'rb') as f:
                return ProtobufManager.read_proto_out_from_file(f)
        else:
            result = self.wrapped_manager.get_proto(id_)
            if result is not None:
                with open(pickle_path, 'wb') as f:
                    f.write(result.SerializeToString())
            return result


class ReplayConverter(object):
    """Converts pandas / proto replays into tensorflow-compatible values

    TODO:
    - write examples to tfrecords
    - implement iterable
    
    """
    DEFAULT_POSITION_COLS = ("pos_x", "pos_y", "pos_z")

    def __init__(self, position_cols=DEFAULT_POSITION_COLS, tfrecord_batch_size=10):
        self.position_cols = position_cols

    def convert_replay_to_example(self, replay_pandas, replay_proto):
        """Convert a replay_pandas and replay_proto into a tf.Example"""
        # unpack player names from proto
        (blue_players, orange_players) = self._get_replay_players(replay_proto)
        # extract features
        ball_gen = self._extract_entity_features('ball', 'ball', replay_pandas)
        t0_gen = self._extract_team_features(blue_players, "t0", replay_pandas)
        t1_gen = self._extract_team_features(orange_players, "t1", replay_pandas)
        feature_gen = itertools.chain(ball_gen, t0_gen, t1_gen)
        feature = {label: feature for (label, feature) in feature_gen}
        # pack into example
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def _get_replay_players(self, replay_proto):
        """Return a list of player names to extract from dataframe"""
        blue_players = []
        orange_players = []
        for player in replay_proto.players:
            if player.is_orange == 0:
                blue_players.append(player.name)
            else:
                orange_players.append(player.name)
        return (blue_players, orange_players)

    def _extract_entity_features(self, entity_name, label_prefix, replay_pandas):
        for col in self.position_cols:
            label = "_".join((label_prefix, col))
            value = replay_pandas[entity_name][col]
            float_list = tf.train.FloatList(value=value)
            feature = tf.train.Feature(float_list=float_list)
            yield (label, feature)

    def _extract_team_features(self, team_players, label_prefix, replay_pandas):
        for i, player in enumerate(team_players):
            player_label_prefix = "{}_p{}".format(label_prefix, i)
            feature_gen = self._extract_entity_features(player,
                                                        player_label_prefix,
                                                        replay_pandas)
            for feature_tuple in feature_gen:
                yield feature_tuple
