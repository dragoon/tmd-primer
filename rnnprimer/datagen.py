from itertools import chain
from typing import List, Iterable, Tuple
from dataclasses import dataclass

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

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
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        speeds = [lf.features[0] for lf in self.features]

        fig.add_trace(
            go.Scatter(x=list(range(len(self))), y=speeds, name="speed"),
            secondary_y=True,
        )

        # Set x-axis title
        fig.update_xaxes(title_text="time step")
        # Set y-axes titles
        fig.update_yaxes(title_text="speed")
        return fig


def generate_sample(train_seg_size=100, total_train_seg_n=5, walk_level=0.5, outlier_prob=0.0):
    def outlier_replace(lf: LabeledFeature):
        if np.random.rand() <= outlier_prob:
            return LabeledFeature(features=[AVG_WALK_SPEED,], label=lf.label,)
        return lf

    def train_speed_func(i):
        max_ix = train_seg_size - 1
        accel_n = int(max_ix * 0.2)
        if i < accel_n:
            return (i * AVG_TRAIN_SPEED) / accel_n
        elif i >= max_ix - accel_n:
            return ((max_ix - i) * AVG_TRAIN_SPEED) / accel_n
        else:
            return AVG_TRAIN_SPEED

    def generate_train_segment():
        return [
            LabeledFeature(features=[train_speed_func(i)], label=0)
            for i in range(train_seg_size)
        ]

    def generate_walk_segment(seq_size):
        return [
            LabeledFeature(features=[s], label=1) for s in [AVG_WALK_SPEED] * seq_size
        ]

    total_walk = int(train_seg_size*total_train_seg_n*2*walk_level)
    # generate total_train_seg_n train segments split between a walk
    train_seg_N1 = np.random.randint(0, total_train_seg_n)
    train_seg_N2 = total_train_seg_n - train_seg_N1
    start_walk_size = np.random.randint(0, total_walk)
    end_walk_size = total_walk - start_walk_size

    return Sample(
        generate_walk_segment(start_walk_size)
        + [outlier_replace(lf) for lf in generate_train_segment() * train_seg_N1]
        + generate_walk_segment(end_walk_size)
        + [outlier_replace(lf) for lf in generate_train_segment() * train_seg_N2]
    )


@dataclass
class Dataset:
    samples: List[Sample]
    std_scaler: StandardScaler

    @staticmethod
    def generate(
            n_samples=100, train_seg_size=100, train_outlier_prob=0.0,
            total_train_seg=lambda: 5
    ):
        samples = []
        for _ in range(n_samples):
            samples.append(
                generate_sample(
                    train_seg_size=train_seg_size,
                    outlier_prob=train_outlier_prob,
                    total_train_seg_n=total_train_seg()
                )
            )

        return Dataset(samples, StandardScaler())

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

    def to_tfds(self, batch_size=20):
        X = [f.features for f in self._get_flat_features()]
        self.std_scaler.fit(X)

        return (
            tf.data.Dataset.from_generator(
                lambda: self._get_ndarray(),
                (tf.float32, tf.int32),
                output_shapes=(None, None),
            )
            .shuffle(1000)
            .batch(batch_size)
        )
