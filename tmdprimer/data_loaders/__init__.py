import abc
from typing import Tuple, Callable, List, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf

from tmdprimer.datagen import make_sliding_windows


def identity(x: Any) -> Any:
    return x


class DataFile(abc.ABC):
    df: pd.DataFrame
    transport_mode: str

    @abc.abstractmethod
    def get_figure(self, *args, **kwargs):
        pass

    def to_numpy_sliding_windows(
        self, window_size: int, label_mapping_func: Callable[[Any], int] = identity
    ) -> Tuple[np.ndarray, np.ndarray]:
        linear_accel_norm = self._get_linear_accel_norm()
        df = pd.DataFrame({"linear": linear_accel_norm, "label": self.df["label"]}).dropna()

        # transform label values to integers
        labels = df["label"].apply(label_mapping_func).to_numpy()

        # fmt: off
        windows_x = make_sliding_windows(
            df[["linear", ]].to_numpy(), window_size, overlap_size=window_size - 1, flatten_inside_window=False
        )
        # fmt: on
        windows_y = make_sliding_windows(labels, window_size, overlap_size=window_size - 1, flatten_inside_window=False)
        # now we need to select a single label for a window  -- last label since that's what we will be predicting
        windows_y = np.array([x[-1] for x in windows_y], dtype=int)
        return windows_x, windows_y

    @abc.abstractmethod
    def stop_durations(self) -> List[Dict]:
        """
        collection all durations of stops and non-stops
        :return: dict with labels as keys and durations list
        """
        pass

    def _get_linear_accel_norm(self) -> np.array:
        """
        :return: linear acceleration mean centered and normalized to input for neural nets
        """
        # clip to 0 - 25
        clipped_accel = np.clip(self.df["linear_accel"], 0, 25)
        # zero centering
        clipped_accel -= np.mean(clipped_accel, axis=0)
        # normalize
        linear_accel_norm = clipped_accel / np.std(clipped_accel, axis=0)
        return linear_accel_norm


class Dataset(abc.ABC):
    data_files: List[DataFile]

    def to_window_tfds(self, window_size, label_mapping_func: Callable[[Any], int]) -> tf.data.Dataset:
        def scaled_iter():
            for f in self.data_files:
                windows_x, windows_y = f.to_numpy_sliding_windows(window_size, label_mapping_func)
                yield from zip(windows_x, windows_y)

        return tf.data.Dataset.from_generator(
            scaled_iter,
            output_types=(tf.float32, tf.int32),
            output_shapes=(tf.TensorShape((window_size, 1)), tf.TensorShape((1,))),
        )

    def to_window_numpy(
        self, window_size, label_mapping_func: Callable[[Any], int] = identity
    ) -> Tuple[np.ndarray, np.ndarray]:
        result_x = None
        result_y = None
        for f in self.data_files:
            windows_x, windows_y = f.to_numpy_sliding_windows(window_size, label_mapping_func)
            if result_x is not None:
                result_x = np.append(result_x, windows_x, axis=0)
                result_y = np.append(result_y, windows_y, axis=0)
            else:
                result_x = windows_x
                result_y = windows_y
        return result_x, result_y

    @property
    def stop_durations_df(self) -> pd.DataFrame:
        """
        collection all durations of stops and non-stops
        :return: dict with labels as keys and durations list
        """
        result = []
        for f in self.data_files:
            # ignore first / last durations just in case
            result.extend(f.stop_durations[1:-1])
        return pd.DataFrame(result)
