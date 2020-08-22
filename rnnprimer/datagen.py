from itertools import chain
import random
from typing import List, Iterable, Tuple, Dict
from dataclasses import dataclass

import numpy as np
import altair as alt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

AVG_WALK_SPEED = 5
AVG_TRAIN_SPEED = 100


@dataclass
class LabeledFeature:
    features: List[float]
    label: int


@dataclass
class Sample:
    features: List[LabeledFeature]

    def __len__(self):
        return len(self.features)

    def labeled_features_tuples(self):
        return [(fw.features, fw.label) for fw in self.features]

    def get_figure(self):
        import pandas as pd

        df = pd.DataFrame(
            data=(
                {"time step": i, "speed": lf.features[0]}
                for i, lf in enumerate(self.features)
            )
        )

        return alt.Chart(df).mark_line().encode(x="time step", y="speed")


def generate_sample(walk_proportion=0.5, outlier_prob=0.0):
    def outlier_replace(lf: LabeledFeature):
        if np.random.rand() <= outlier_prob:
            return LabeledFeature(features=[AVG_WALK_SPEED,], label=lf.label,)
        return lf

    def train_speed_func(train_seg_size):
        time_to_max_speed = 20
        if train_seg_size >= time_to_max_speed * 2:
            return (
                [
                    i * AVG_TRAIN_SPEED / time_to_max_speed
                    for i in range(time_to_max_speed)
                ]
                + [
                    AVG_TRAIN_SPEED
                    for _ in range(train_seg_size - time_to_max_speed * 2)
                ]
                + list(
                    reversed(
                        [
                            i * AVG_TRAIN_SPEED / time_to_max_speed
                            for i in range(time_to_max_speed)
                        ]
                    )
                )
            )
        else:
            half_way_n = train_seg_size // 2
            return [
                i * AVG_TRAIN_SPEED / time_to_max_speed for i in range(half_way_n)
            ] + list(
                reversed(
                    [i * AVG_TRAIN_SPEED / time_to_max_speed for i in range(half_way_n)]
                )
            )

    def generate_train_segment():
        # always even number
        train_seg_size = random.randrange(20, 101, 2)
        return [
            outlier_replace(LabeledFeature(features=[f], label=0))
            for f in train_speed_func(train_seg_size)
        ]

    def generate_walk_segment(seq_size):
        return [
            LabeledFeature(features=[s], label=1) for s in [AVG_WALK_SPEED] * seq_size
        ]

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

    def _get_sequences(self):
        sample_iter = (sample.labeled_features_tuples() for sample in self.samples)
        return (tuple(zip(*windows)) for windows in sample_iter)

    def _get_ndarray(self) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Same output as `get_sequences` but with Numpy ndarrays (features, labels) with
        respective shape ([time, features], [time, 1], [time]).
        This shape is useful for feeding the data into the Keras model.
        """
        return (
            (
                self.std_scaler.transform(np.array(features, copy=True)),
                np.expand_dims(np.array(labels), axis=-1),
            )
            for features, labels in self._get_sequences()
        )

    def _get_weighted_ndarray(
        self, weighting: Dict
    ) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
            (tf.float32, tf.int32, tf.float32),
        ).padded_batch(
            batch_size,
            padded_shapes=([None, feature_n], [None, 1], [None]),
            padding_values=(-1.0, 0, 0.0),
        )

    def to_tfds(self, batch_size=20):
        X = [f.features for f in self._get_flat_features()]
        self.std_scaler.fit(X)
        feature_n = len(X[0])

        return tf.data.Dataset.from_generator(
            lambda: self._get_ndarray(), (tf.float32, tf.int32),
        ).padded_batch(
            batch_size,
            padding_values=(-1.0, 0),
            padded_shapes=([None, feature_n], [None, 1]),
        )

    def _get_strided_ndarray(
        self, window_size
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        for feature_arr, label_arr in self._get_ndarray():
            # features is size (n_timesteps, feature_n=1)
            # labels is size (n_timesteps, 1)
            i = 0
            for stride_features in strided_axis0(feature_arr, window_size):
                yield stride_features, [label_arr[i + window_size - 1]]
                i += 1

    def to_cnn_tfds(self, window_size, batch_size=20):
        X = [f.features for f in self._get_flat_features()]
        self.std_scaler.fit(X)

        return tf.data.Dataset.from_generator(
            lambda: self._get_strided_ndarray(window_size), (tf.float32, tf.int32),
        ).batch(batch_size)


def strided_axis0(a: np.array, L: int):
    """
    https://stackoverflow.com/questions/43413582/selecting-multiple-slices-from-a-numpy-array-at-once/43413801#43413801
    :param a: array
    :param L: length of array along axis=0 to be cut for forming each subarray
    :return:
    """

    # Length of 3D output array along its axis=0
    nd0 = a.shape[0] - L + 1

    # Store shape and strides info
    m, n = a.shape
    s0, s1 = a.strides

    # Finally use strides to get the 3D array view
    return np.lib.stride_tricks.as_strided(a, shape=(nd0, L, n), strides=(s0, s0, s1))
