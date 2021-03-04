import os
import json
from datetime import timedelta, datetime
from types import SimpleNamespace
from unittest import TestCase, main
import numpy as np

from tmdprimer.data_loaders.dvdt_data_loader import DVDTFile, DVDTDataset, AnnotatedStop


class TestDVDTLoader(TestCase):
    test_file: DVDTFile

    def setUp(self):
        super().setUp()
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.test_file = DVDTFile.from_json(json.load(open(f"{file_path}/accel_data.json")))
        self.dataset = DVDTDataset.from_files([self.test_file])

    def test_dataset_window_tfds(self):
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

    def test_dataset_window_numpy(self):
        window_size = 5
        # need > 1 file to test
        file_path = os.path.dirname(os.path.realpath(__file__))
        test_file1 = DVDTFile.from_json(json.load(open(f"{file_path}/accel_data.json")))
        test_file2 = DVDTFile.from_json(json.load(open(f"{file_path}/accel_data.json")))
        dataset = DVDTDataset.from_files([test_file1, test_file2])
        x, y = dataset.to_window_numpy(label=1, window_size=5, stop_label=0)
        true_windows_len = sum(len(s.df) for s in dataset.dvdt_files) - len(dataset.dvdt_files) * (window_size - 1)
        self.assertEquals(len(x), true_windows_len)
        # each element in X has shape of (window_size, 1)
        self.assertEquals(x[0].shape, (window_size, 1))
        # labels is (1,)
        self.assertEquals(y[0].shape, (1,))

    def test_stop_durations(self):
        durations = self.dataset.stop_durations_df["duration"].to_list()
        self.assertEqual(durations, [timedelta(milliseconds=2)])


class TestModelClassification(TestCase):
    test_file: DVDTFile

    def setUp(self):
        super().setUp()
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.test_file = DVDTFile.from_json(json.load(open(f"{file_path}/accel_data_many_stops.json")))
        self.dataset = DVDTDataset.from_files([self.test_file])

    def test_compute_stops(self):
        model = SimpleNamespace(predict=lambda np_array: np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0]))
        predicted_stops = self.test_file.compute_stops(
            model=model,
            model_window_size=2,
            smoothing_window_size=2,
            threshold_probability=0.5,
            min_stop_duration=timedelta(seconds=3),
            min_interval_between_stops=timedelta(seconds=5),
        )
        self.assertEqual(
            predicted_stops,
            [
                AnnotatedStop(datetime.utcfromtimestamp(10), datetime.utcfromtimestamp(15)),
                AnnotatedStop(datetime.utcfromtimestamp(25), datetime.utcfromtimestamp(30)),
            ],
        )

    def test_compute_stops_with_merging(self):
        model = SimpleNamespace(predict=lambda np_array: np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0]))
        predicted_stops = self.test_file.compute_stops(
            model=model,
            model_window_size=2,
            smoothing_window_size=2,
            threshold_probability=0.5,
            min_stop_duration=timedelta(seconds=3),
            # min interval is > than the time between stops
            min_interval_between_stops=timedelta(seconds=11),
        )
        self.assertEqual(
            predicted_stops,
            [
                AnnotatedStop(datetime.utcfromtimestamp(10), datetime.utcfromtimestamp(30))
            ],
        )

    def test_precision_recall_correct(self):
        predicted_stops = [
            AnnotatedStop(datetime.utcfromtimestamp(10), datetime.utcfromtimestamp(15)),
            AnnotatedStop(datetime.utcfromtimestamp(25), datetime.utcfromtimestamp(30)),
        ]

        precision, recall = self.test_file.get_precision_recall(predicted_stops)
        self.assertEqual(precision, 1.0)
        self.assertEqual(recall, 1.0)

    def test_precision_correct(self):
        predicted_stops = [
            AnnotatedStop(datetime.utcfromtimestamp(10), datetime.utcfromtimestamp(15))
        ]

        precision, recall = self.test_file.get_precision_recall(predicted_stops)
        self.assertEqual(precision, 1.0)
        self.assertEqual(recall, 0.5)

        predicted_stops = [
            AnnotatedStop(datetime.utcfromtimestamp(25), datetime.utcfromtimestamp(30))
        ]

        precision, recall = self.test_file.get_precision_recall(predicted_stops)
        self.assertEqual(precision, 1.0)
        self.assertEqual(recall, 0.5)

    def test_recall_correct(self):
        predicted_stops = [
            AnnotatedStop(datetime.utcfromtimestamp(0), datetime.utcfromtimestamp(5)),
            AnnotatedStop(datetime.utcfromtimestamp(10), datetime.utcfromtimestamp(15)),
            AnnotatedStop(datetime.utcfromtimestamp(25), datetime.utcfromtimestamp(30)),
        ]

        precision, recall = self.test_file.get_precision_recall(predicted_stops)
        self.assertEqual(precision, 2/3)
        self.assertEqual(recall, 1.0)


class TestAnnotatedStop(TestCase):
    def test_max_margin(self):
        as1 = AnnotatedStop(datetime.fromtimestamp(1), datetime.fromtimestamp(10))
        as2 = AnnotatedStop(datetime.fromtimestamp(5), datetime.fromtimestamp(15))

        margin = as1.max_margin(as2)
        self.assertEqual(margin, timedelta(seconds=5))

    def test_max_margin_non_overlap(self):
        as1 = AnnotatedStop(datetime.fromtimestamp(1), datetime.fromtimestamp(10))
        as2 = AnnotatedStop(datetime.fromtimestamp(11), datetime.fromtimestamp(15))

        margin = as1.max_margin(as2)
        self.assertEqual(margin, timedelta(seconds=10))


if __name__ == "__main__":
    main()
