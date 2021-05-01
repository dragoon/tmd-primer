import os
import json
from datetime import timedelta, datetime
from types import SimpleNamespace
from unittest import TestCase, main
import numpy as np

from tmdprimer.stop_classification.datasets.dvdt_dataset import DVDTFile, DVDTDataset, AnnotatedStop
from tmdprimer.stop_classification.domain.models import SlidingWindowModel


class TestSlidingWindowModel(TestCase):
    test_file: DVDTFile

    def setUp(self):
        super().setUp()
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.test_file = DVDTFile.from_json(json.load(open(f"{file_path}/accel_data_many_stops.json")))
        self.dataset = DVDTDataset([self.test_file])

    def test_compute_stops(self):
        model_int = SimpleNamespace(predict=lambda np_array: np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0]))
        model = SlidingWindowModel(model_int, window_size=2)

        predicted_stops = model.compute_stops(
            data_file=self.test_file,
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
        model_int = SimpleNamespace(predict=lambda np_array: np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0]))
        model = SlidingWindowModel(model_int, window_size=2)
        predicted_stops = model.compute_stops(
            data_file=self.test_file,
            smoothing_window_size=2,
            threshold_probability=0.5,
            min_stop_duration=timedelta(seconds=3),
            # min interval is > than the time between stops
            min_interval_between_stops=timedelta(seconds=11),
        )
        self.assertEqual(
            predicted_stops,
            [AnnotatedStop(datetime.utcfromtimestamp(10), datetime.utcfromtimestamp(30))],
        )

if __name__ == "__main__":
    main()
