from unittest import TestCase, main
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from tmdprimer.datagen import Sample, LabeledFeature, Dataset


class TestDVDTLoader(TestCase):

    def test_to_numpy(self):
        sample = Sample([LabeledFeature([x], y) for x, y in zip(range(100), [0] * 100)])
        np_x, np_y = sample.to_numpy()
        np_true_x = np.arange(0, 100).reshape(-1, 1)
        np_true_y = np.zeros((100,))
        self.assertTrue(np.array_equal(np_x, np_true_x))
        self.assertTrue(np.array_equal(np_y, np_true_y))

    def test_dataset(self):
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


if __name__ == "__main__":
    main()
