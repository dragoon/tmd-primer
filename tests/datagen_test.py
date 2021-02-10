from types import SimpleNamespace
from unittest import TestCase, main
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from tmdprimer.datagen import Sample, LabeledFeature, Dataset


class DatagenTest(TestCase):
    def test_to_numpy(self):
        sample = Sample([LabeledFeature([x], y) for x, y in zip(range(100), [0] * 100)])
        # dummy scaler
        scaler = SimpleNamespace(transform=lambda x: x)

        np_x, np_y = sample.to_numpy(scaler)
        np_true_x = np.arange(0, 100).reshape(-1, 1)
        np_true_y = np.zeros((100, 1), dtype=int)
        self.assertTrue(np.array_equal(np_x, np_true_x))
        self.assertTrue(np.array_equal(np_y, np_true_y))

    def test_tfds(self):
        sample1 = Sample([LabeledFeature([x], y) for x, y in zip(range(100), [0] * 100)])
        sample2 = Sample([LabeledFeature([x], y) for x, y in zip(range(100), [1] * 100)])
        dataset = Dataset([sample1, sample2], MinMaxScaler())
        # convert to list to compare
        tfds = list(dataset.to_tfds().as_numpy_iterator())
        # dataset has 2 element
        self.assertEquals(len(tfds), 2)
        # each element has 2 element tuple -- features and labels
        self.assertEquals(len(tfds[0]), 2)
        # both features and labels have shapes of (100, 1)
        self.assertEquals(tfds[0][0].shape, (100, 1))
        self.assertEquals(tfds[0][1].shape, (100, 1))

    def test_window_tfds(self):
        sample1 = Sample([LabeledFeature([x], y) for x, y in zip(range(100), [0] * 100)])
        sample2 = Sample([LabeledFeature([x], y) for x, y in zip(range(100), [1] * 100)])
        dataset = Dataset([sample1, sample2], MinMaxScaler())
        # convert to list to compare
        window_size = 5
        tfds = list(dataset.to_window_tfds(window_size).as_numpy_iterator())
        # dataset has 2 element
        # number of widows = total timesteps (100+100) - dropped remainder for each sample (4+4) = 192
        true_windows_size = sum(len(s.features) for s in dataset.samples) - len(dataset.samples) * (window_size - 1)
        self.assertEquals(len(tfds), true_windows_size)
        # each element has 2 element tuple -- features and labels
        self.assertEquals(len(tfds[0]), 2)
        # features has shape of (window_size, 1)
        self.assertEquals(tfds[0][0].shape, (window_size, 1))
        # labels is (1,)
        self.assertEquals(tfds[0][1].shape, (1,))
        # TODO: also check label and feature contents

    def test_split_window_tfds(self):
        sample1 = Sample([LabeledFeature([x], y) for x, y in zip(range(100), [0] * 100)])
        sample2 = Sample([LabeledFeature([x], y) for x, y in zip(range(100), [1] * 100)])
        dataset = Dataset([sample1, sample2], MinMaxScaler())
        # convert to list to compare
        window_size = 5
        tfds = list(dataset.to_split_window_tfds(window_size).as_numpy_iterator())
        # dataset has 2 element
        # number of widows = sum for each sample == number of timesteps // window_size
        true_windows_size = sum(len(s.features) // window_size for s in dataset.samples)
        self.assertEquals(len(tfds), true_windows_size)
        # each element has 2 element tuple -- features and labels
        self.assertEquals(len(tfds[0]), 2)
        # features has shape of (window_size, 1)
        self.assertEquals(tfds[0][0].shape, (window_size, 1))
        # labels is (5, 1) for split window since we predict sequences
        self.assertEquals(tfds[0][1].shape, (window_size, 1))
        # TODO: also check label and feature contents


if __name__ == "__main__":
    main()
