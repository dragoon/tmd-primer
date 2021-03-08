import abc
from typing import Tuple, Callable, List, Dict

import numpy as np
import pandas as pd
import tensorflow as tf


class DataFile(abc.ABC):
    df: pd.DataFrame
    transport_mode: str

    @abc.abstractmethod
    def get_figure(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def to_numpy_sliding_windows(
        self, window_size: int, label_mapping_func: Callable[[str], int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param window_size: size of the sliding window
        :param label_mapping_func: function mapping raw string label to int type
        :return:
        """
        pass

    @abc.abstractmethod
    def stop_durations(self) -> List[Dict]:
        """
        collection all durations of stops and non-stops
        :return: dict with labels as keys and durations list
        """
        pass


class Dataset(abc.ABC):
    data_files: List[DataFile]

    @abc.abstractmethod
    def to_window_tfds(self, window_size, label_mapping_func: Callable[[str], int]) -> tf.data.Dataset:
        pass

    @abc.abstractmethod
    def to_window_numpy(self, window_size, label_mapping_func: Callable[[str], int]) -> Tuple[np.ndarray, np.ndarray]:
        pass

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
