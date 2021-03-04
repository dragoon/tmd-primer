import abc
from typing import Tuple, Callable

import numpy as np


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
