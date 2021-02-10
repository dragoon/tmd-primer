import os
import json
from unittest import TestCase, main
import numpy as np

from tmdprimer.data_loaders.dvdt_data_loader import DVDTFile, DVDTDataset


class TestDVDTLoader(TestCase):
    test_file: DVDTFile

    def setUp(self):
        super().setUp()
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.test_file = DVDTFile.from_json(json.load(open(f"{file_path}/accel_data.json")))
        self.dataset = DVDTDataset.from_files([self.test_file])

    def test_windows_x_y(self):
        features, labels = self.test_file.get_features_labels(label=1, stop_label=0, median_filter_window=1)
        true_features = np.asarray([[0.04], [0.08], [0.12], [0.16], [0.2], [0.24]])
        true_labels = np.asarray([[1], [0], [0], [0], [1], [1]])
        self.assertTrue(np.array_equal(labels, true_labels))
        self.assertTrue(np.array_equal(features, true_features))

    def test_tfds(self):
        window_size = 5
        tfds = list(self.dataset.to_window_tfds(label=1, window_size=5, stop_label=0).as_numpy_iterator())
        true_windows_size = sum(len(s.df) for s in self.dataset.dvdt_files) - len(self.dataset.dvdt_files) * (
            window_size - 1
        )
        self.assertEquals(len(tfds), true_windows_size)
        # each element has 2 element tuple -- features and labels
        self.assertEquals(len(tfds[0]), 2)
        # features has shape of (window_size, 1)
        self.assertEquals(tfds[0][0].shape, (window_size, 1))
        # labels is (1,)
        self.assertEquals(tfds[0][1].shape, (1,))


if __name__ == "__main__":
    main()
