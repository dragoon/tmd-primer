import json
from unittest import TestCase, main
import numpy as np

from tmdprimer.s3_util.dvdt_data_loader import DVDTFile


class TestDVDTLoader(TestCase):
    test_file: DVDTFile

    def setUp(self):
        super().setUp()
        self.test_file = DVDTFile.from_json(json.load(open("accel_data.json")))

    def test_windows_x_y(self):
        features, labels = self.test_file.get_features_labels(label=1, stop_label=0, median_filter_window=1)
        true_features = np.asarray([[0.04], [0.08], [0.12], [0.16], [0.2], [0.24]])
        true_labels = np.asarray([[1], [0], [0], [0], [1], [1]])
        self.assertTrue(np.array_equal(labels, true_labels))
        self.assertTrue(np.array_equal(features, true_features))


if __name__ == "__main__":
    main()
