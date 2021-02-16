from types import SimpleNamespace
from unittest import TestCase, main
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from tmdprimer.datagen import Sample, LabeledFeature, Dataset, make_sliding_windows, generate_sample


class DatagenTest(TestCase):

    def test_generate_sample(self):
        sample = generate_sample()
        self.assertLessEqual(len(sample), 2000)
        self.assertGreaterEqual(len(sample), 160)
        # check walk proportion
        self.assertEqual(sum([f.label for f in sample.features]), len(sample)/2)

    def test_smaple_to_numpy(self):
        sample = Sample([LabeledFeature([x], y) for x, y in zip(range(100), [0] * 100)])
        # dummy scaler
        scaler = SimpleNamespace(transform=lambda x: x)

        np_x, np_y = sample.to_numpy(scaler)
        np_true_x = np.arange(0, 100).reshape(-1, 1)
        np_true_y = np.zeros((100, 1), dtype=int)
        self.assertTrue(np.array_equal(np_x, np_true_x))
        self.assertTrue(np.array_equal(np_y, np_true_y))

    def test_dataset_tfds(self):
        sample1 = Sample([LabeledFeature([x], y) for x, y in zip(range(100), [0] * 100)])
        sample2 = Sample([LabeledFeature([x], y) for x, y in zip(range(100), [1] * 100)])
        dataset = Dataset([sample1, sample2], MinMaxScaler())
        # convert to list to compare
        tfds = list(dataset.to_tfds().as_numpy_iterator())
        # dataset has 2 element
        self.assertEqual(len(tfds), 2)
        # each element has 2 element tuple -- features and labels
        self.assertEqual(len(tfds[0]), 2)
        # both features and labels have shapes of (100, 1)
        self.assertEqual(tfds[0][0].shape, (100, 1))
        self.assertEqual(tfds[0][1].shape, (100, 1))

    def test_dataset_sliding_window_tfds(self):
        sample1 = Sample([LabeledFeature([x], y) for x, y in zip(range(100), [0] * 100)])
        sample2 = Sample([LabeledFeature([x], y) for x, y in zip(range(100), [1] * 100)])
        dataset = Dataset([sample1, sample2], MinMaxScaler())
        # convert to list to compare
        window_size = 5
        tfds = list(dataset.to_window_tfds(window_size).as_numpy_iterator())
        # dataset has 2 element
        # number of widows = total timesteps (100+100) - dropped remainder for each sample (4+4) = 192
        true_windows_size = sum(len(s.features) for s in dataset.samples) - len(dataset.samples) * (window_size - 1)
        self.assertEqual(len(tfds), true_windows_size)
        # each element has 2 element tuple -- features and labels
        self.assertEqual(len(tfds[0]), 2)
        # features has shape of (window_size, 1)
        self.assertEqual(tfds[0][0].shape, (window_size, 1))
        # labels is (1,)
        self.assertEqual(tfds[0][1].shape, (1,))
        # TODO: also check label and feature contents

    def test_dataset_split_window_tfds(self):
        sample1 = Sample([LabeledFeature([x], y) for x, y in zip(range(100), [0] * 100)])
        sample2 = Sample([LabeledFeature([x], y) for x, y in zip(range(100), [1] * 100)])
        dataset = Dataset([sample1, sample2], MinMaxScaler())
        # convert to list to compare
        window_size = 5
        tfds = list(dataset.to_split_window_tfds(window_size).as_numpy_iterator())
        # dataset has 2 element
        # number of widows = sum for each sample == number of timesteps // window_size
        true_windows_size = sum(len(s.features) // window_size for s in dataset.samples)
        self.assertEqual(len(tfds), true_windows_size)
        # each element has 2 element tuple -- features and labels
        self.assertEqual(len(tfds[0]), 2)
        # features has shape of (window_size, 1)
        self.assertEqual(tfds[0][0].shape, (window_size, 1))
        # labels is (5, 1) for split window since we predict sequences
        self.assertEqual(tfds[0][1].shape, (window_size, 1))
        # TODO: also check label and feature contents

    def test_make_sliding_windows_overlap(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7])
        windows = make_sliding_windows(data, window_size=5, overlap_size=4, flatten_inside_window=True)
        true_windows = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])
        self.assertTrue(np.array_equal(true_windows, windows))

    def test_make_sliding_windows_no_overlap(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7])
        windows = make_sliding_windows(data, window_size=5, overlap_size=0, flatten_inside_window=True)
        true_windows = np.array([[1, 2, 3, 4, 5]])
        self.assertTrue(np.array_equal(true_windows, windows))


if __name__ == "__main__":
    main()
