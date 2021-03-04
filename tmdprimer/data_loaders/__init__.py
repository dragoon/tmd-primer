import abc
from typing import Tuple, Callable

import numpy as np
import tensorflow as tf


class DataFile(abc.ABC):
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


class Dataset(abc.ABC):
    @abc.abstractmethod
    def to_window_tfds(self, window_size, label_mapping_func: Callable[[str], int]) -> tf.data.Dataset:
        pass

    @abc.abstractmethod
    def to_window_numpy(self, window_size, label_mapping_func: Callable[[str], int]) -> Tuple[np.ndarray, np.ndarray]:
        pass
