import os
from unittest import TestCase
import pandas as pd

from tmdprimer.data_loaders.sensorlog_data_loader import SensorLogFile


class TestSensorLogFile(TestCase):
    test_file: SensorLogFile

    def setUp(self):
        super().setUp()
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.test_file = SensorLogFile.from_csv(pd.read_csv(f"{file_path}/sensorlog_data.csv", sep=";"))

    def test_to_numpy_sliding_windows(self):
        window_size = 4
        # need > 1 file to test
        x, y = self.test_file.to_numpy_sliding_windows(window_size=window_size)
        true_windows_len = len(self.test_file.df) - (window_size - 1)
        self.assertEquals(len(x), true_windows_len)
        # each element in X has shape of (window_size, 1)
        self.assertEquals(x[0].shape, (window_size, 1))
        # labels is (1,)
        self.assertEquals(y[0].shape, (1,))
