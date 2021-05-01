import json
import os
from pathlib import Path
from typing import Iterable, Union
from zipfile import ZipFile

import pandas as pd

from tmdprimer.stop_classification.data_loaders import DataLoader
from tmdprimer.stop_classification.datasets.dvdt_dataset import DVDTFile, DVDTDataset
from tmdprimer.stop_classification.datasets.sensorlog_dataset import SensorLogFile, SensorLogDataset


class DVDTLocalDataLoader(DataLoader):
    def load_dataset(self, path: str, labels_to_load: Iterable = None) -> DVDTDataset:
        files = []
        path = Path(path)
        for file_name in os.listdir(path):
            if file_name.endswith("high.zip"):
                data_file = self.load_file(file_name=path / file_name)
                if labels_to_load is None or data_file.transport_mode in labels_to_load:
                    files.append(data_file)
        return DVDTDataset(files)

    def load_file(self, file_name: Union[Path, str]) -> DVDTFile:
        print("loading", file_name)
        with open(file_name, 'rb') as fh:
            with ZipFile(fh, mode="r") as zip_file:
                for file in zip_file.namelist():
                    if file.endswith(".json") and "/" not in file:
                        with zip_file.open(file) as accel_json:
                            return DVDTFile.from_json(json.loads(accel_json.read()))


class SensorLogS3DataLoader(DataLoader):
    def load_dataset(self, bucket, path: str, labels_to_load: Iterable = None) -> SensorLogDataset:
        files = []
        path = Path(path)
        for file_name in os.listdir(path):
            if file_name.endswith(".csv"):
                data_file = self.load_file(file_name=path / file_name)
                files.append(data_file)
        return SensorLogDataset(files)

    def load_file(self, file_name: Union[Path, str]) -> SensorLogFile:
        return SensorLogFile.from_csv(pd.read_csv(file_name, sep=";"))
