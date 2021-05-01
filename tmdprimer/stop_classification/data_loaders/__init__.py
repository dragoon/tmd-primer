import abc

from tmdprimer.stop_classification.datasets import Dataset, DataFile


class DataLoader(abc.ABC):
    @abc.abstractmethod
    def load_dataset(self, *args, **kwargs) -> Dataset:
        pass

    @abc.abstractmethod
    def load_file(self, *args, **kwargs) -> DataFile:
        pass
