from itertools import chain
import random
from typing import List, Iterable, Tuple, Dict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import altair as alt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

AVG_WALK_SPEED = 5
AVG_TRAIN_SPEED = 100


@dataclass(frozen=True)
class LabeledFeature:
    features: List[float]
    label: int


@dataclass(frozen=True)
class Sample:
    features: List[LabeledFeature]

    def __len__(self):
        return len(self.features)

    def labeled_features_tuples(self):
        return [(fw.features, fw.label) for fw in self.features]

    def to_numpy(self, scaler) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param scaler: scaler to transform features
        :return: x of shape (seq_length, n_features), y of shape (seq_length, 1)
        """
        features, labels = tuple(zip(*self.labeled_features_tuples()))
        return scaler.transform(np.array(features)), np.reshape(np.array(labels), (-1, 1))

    def to_numpy_split_windows(self, window_size, scaler) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param window_size:
        :param scaler: scale before making windows for speed (2x performance improvement)
        :return: non-overlapping windows of size window_size from original sequence
        """
        x, y = self.to_numpy(scaler)
        x_windows = make_sliding_windows(x, window_size, overlap_size=0, flatten_inside_window=False)
        y_windows = make_sliding_windows(y, window_size, overlap_size=0, flatten_inside_window=False)
        return x_windows, y_windows

    def to_numpy_sliding_windows(self, window_size, scaler) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param window_size:
        :param scaler: scale before making windows for speed (2x performance improvement)
        :return:
        """
        x, y = self.to_numpy(scaler)
        x_windows = make_sliding_windows(x, window_size, overlap_size=window_size - 1, flatten_inside_window=False)
        y_windows = make_sliding_windows(y, window_size, overlap_size=window_size - 1, flatten_inside_window=False)
        # take the last label of the window
        y_windows = np.array([x[-1] for x in y_windows], dtype=int)
        return x_windows, y_windows

    def get_figure(self):
        df = pd.DataFrame(data=({"time step": i, "speed": lf.features[0]} for i, lf in enumerate(self.features)))

        return alt.Chart(df).mark_line().encode(x="time step", y="speed")


def generate_sample(walk_proportion=0.5, outlier_prob=0.0):
    def outlier_replace(lf: LabeledFeature):
        if np.random.rand() <= outlier_prob:
            return LabeledFeature(
                features=[
                    AVG_WALK_SPEED,
                ],
                label=lf.label,
            )
        return lf

    def train_speed_func(train_seg_size):
        time_to_max_speed = 20
        if train_seg_size >= time_to_max_speed * 2:
            return (
                [i * AVG_TRAIN_SPEED / time_to_max_speed for i in range(time_to_max_speed)]
                + [AVG_TRAIN_SPEED for _ in range(train_seg_size - time_to_max_speed * 2)]
                + list(reversed([i * AVG_TRAIN_SPEED / time_to_max_speed for i in range(time_to_max_speed)]))
            )
        else:
            half_way_n = train_seg_size // 2
            return [i * AVG_TRAIN_SPEED / time_to_max_speed for i in range(half_way_n)] + list(
                reversed([i * AVG_TRAIN_SPEED / time_to_max_speed for i in range(half_way_n)])
            )

    def generate_train_segment():
        # always even number
        train_seg_size = random.randrange(20, 101, 2)
        return [outlier_replace(LabeledFeature(features=[f], label=0)) for f in train_speed_func(train_seg_size)]

    def generate_walk_segment(seq_size):
        return [LabeledFeature(features=[s], label=1) for s in [AVG_WALK_SPEED] * seq_size]

    total_train_seg_n = np.random.randint(4, 10)
    # split all train segments into two trips
    train_seg_n1 = np.random.randint(0, total_train_seg_n)
    train_seg_n2 = total_train_seg_n - train_seg_n1
    train_seg1 = list(chain.from_iterable(generate_train_segment() for _ in range(train_seg_n1)))
    train_seg2 = list(chain.from_iterable(generate_train_segment() for _ in range(train_seg_n2)))

    total_walk = int((len(train_seg1) + len(train_seg2)) * walk_proportion * 2)
    walk_seg_n1 = np.random.randint(0, total_walk)
    walk_seg_n2 = total_walk - walk_seg_n1
    segments = [
        train_seg1,
        train_seg2,
        generate_walk_segment(walk_seg_n1),
        generate_walk_segment(walk_seg_n2),
    ]
    random.shuffle(segments)

    return Sample(list(chain.from_iterable(segments)))


@dataclass
class Dataset:
    samples: List[Sample]
    std_scaler: MinMaxScaler

    @staticmethod
    def generate(n_samples=100, train_outlier_prob=0.0):
        samples = []
        for _ in range(n_samples):
            samples.append(generate_sample(outlier_prob=train_outlier_prob))

        return Dataset(samples, MinMaxScaler())

    def _get_flat_features(self) -> List[LabeledFeature]:
        return list(chain.from_iterable(s.features for s in self.samples))

    def get_flat_X_y(self):
        X, y = tuple(zip(*[(f.features, f.label) for f in self._get_flat_features()]))
        p = np.random.permutation(len(X))
        return np.array(X)[p], np.array(y)[p]

    def get_flat_X_y_scaled(self):
        X, y = tuple(zip(*[(f.features, f.label) for f in self._get_flat_features()]))
        X = self.std_scaler.fit_transform(X)
        p = np.random.permutation(len(X))
        return np.array(X)[p], np.array(y)[p]

    def _get_weighted_ndarray(self, weighting: Dict) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return (
            (
                self.std_scaler.transform(np.array(features, copy=True)),
                np.expand_dims(np.array(labels), axis=-1),
                np.array([weighting[l] for l in labels]),
            )
            for features, labels in self._get_sequences()
        )

    def to_weighted_tfds(self, weighting: Dict, batch_size=20):
        X = [f.features for f in self._get_flat_features()]
        self.std_scaler.fit(X)
        feature_n = len(X[0])

        return tf.data.Dataset.from_generator(
            lambda: self._get_weighted_ndarray(weighting),
            output_signature=(
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            ),
        ).padded_batch(
            batch_size,
            padded_shapes=([None, feature_n], [None, 1], [None]),
            padding_values=(-1.0, 0, 0.0),
        )

    def to_tfds(self):
        X = [f.features for f in self._get_flat_features()]
        self.std_scaler.fit(X)
        feature_n = len(X[0])

        # tensorflow can call this method multiple times to get as many samples as needed
        # specifying it as lambda doesn't work since iterator is exhausted
        def scaled_iter():
            return (s.to_numpy(self.std_scaler) for s in self.samples)

        return tf.data.Dataset.from_generator(
            scaled_iter,
            output_signature=(
                tf.TensorSpec(shape=(None, feature_n), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
            ),
        )

    def to_split_window_tfds(self, window_size):
        X = [f.features for f in self._get_flat_features()]
        feature_n = len(X[0])
        self.std_scaler.fit(X)

        # tensorflow can call this method multiple times to get as many samples as needed
        # specifying it as lambda doesn't work since iterator is exhausted
        def scaled_iter():
            for windows_x, windows_y in (
                s.to_numpy_split_windows(window_size, self.std_scaler) for s in self.samples
            ):
                yield from zip(windows_x, windows_y)

        return tf.data.Dataset.from_generator(
            scaled_iter,
            output_signature=(
                tf.TensorSpec(shape=(window_size, feature_n), dtype=tf.float32),
                tf.TensorSpec(shape=(window_size, 1), dtype=tf.int32),
            ),
        )

    def to_window_tfds(self, window_size):
        X = [f.features for f in self._get_flat_features()]
        feature_n = len(X[0])
        self.std_scaler.fit(X)

        # tensorflow can call this method multiple times to get as many samples as needed
        # specifying it as lambda doesn't work since iterator is exhausted
        def scaled_iter():
            for windows_x, windows_y in (
                s.to_numpy_sliding_windows(window_size, self.std_scaler) for s in self.samples
            ):
                yield from zip(windows_x, windows_y)

        return tf.data.Dataset.from_generator(
            scaled_iter,
            output_signature=(
                tf.TensorSpec(shape=(window_size, feature_n), dtype=tf.float32),
                tf.TensorSpec(shape=(1,), dtype=tf.int32),
            ),
        )


def make_sliding_windows(data, window_size, overlap_size=0, flatten_inside_window=True):
    assert data.ndim == 1 or data.ndim == 2
    if data.ndim == 1:
        data = data.reshape((-1, 1))

    # get the number of overlapping windows that fit into the data
    num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
    # ROMAN: skip overhang padding
    # overhang = data.shape[0] - (num_windows * window_size - (num_windows - 1) * overlap_size)

    # if there's overhang, need an extra window and a zero pad on the data
    # (numpy 1.7 has a nice pad function I'm not using here)
    # if overhang != 0:
    #     num_windows += 1
    #     newdata = np.zeros((num_windows * window_size - (num_windows - 1) * overlap_size, data.shape[1]))
    #     newdata[: data.shape[0]] = data
    #     data = newdata

    sz = data.dtype.itemsize
    ret = np.lib.stride_tricks.as_strided(
        data,
        shape=(num_windows, window_size * data.shape[1]),
        strides=((window_size - overlap_size) * data.shape[1] * sz, sz),
    )

    if flatten_inside_window:
        return ret
    else:
        return ret.reshape((num_windows, -1, data.shape[1]))
