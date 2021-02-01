import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List

import numpy as np
import tensorflow as tf
import pandas as pd
import io
from zipfile import ZipFile
import boto3

from tmdprimer.datagen import make_sliding_windows


@dataclass
class DVDTFile:
    start_time: datetime
    end_time: datetime
    num_stations: int
    transport_mode: str
    comment: str
    df: pd.DataFrame

    @classmethod
    def from_json(cls, json_dict: Dict):
        metadata = json_dict["metadata"]
        start_time = datetime.fromtimestamp(metadata["timestamp"] / 1000)
        end_time = datetime.fromtimestamp(metadata["endtime"] / 1000)
        num_stations = metadata["numberStations"]
        transport_mode = metadata["transportMode"]
        comment = metadata["comment"]
        df = pd.DataFrame(json_dict["entries"])
        return DVDTFile(start_time, end_time, num_stations, transport_mode, comment, df)

    def _get_linear_accel(self):
        linear_accel_series = np.sqrt(self.df["x"] ** 2 + self.df["y"] ** 2 + self.df["z"] ** 2)
        # clip to 0 - 25
        clipped_accel = np.clip(linear_accel_series, 0, 25)
        # make accel between 0 and 1
        linear_accel_norm = (clipped_accel - np.min(clipped_accel)) / (np.max(clipped_accel) - np.min(clipped_accel))
        return linear_accel_norm

    def to_tfds(self, label, window_size=512) -> tf.data.Dataset:
        time_diff_series = self.df["timestamp"].diff()
        linear_accel_norm = self._get_linear_accel()
        data_x = linear_accel_norm.to_numpy()
        windows_x = make_sliding_windows(data_x, window_size, overlap_size=window_size - 1, flatten_inside_window=False)
        windows_y = np.full((len(windows_x),), [label])
        return tf.data.Dataset.from_tensor_slices((windows_x, windows_y))


class DVDTDataset:
    s3client = None
    bucket: str

    def __init__(self, bucket: str):
        self.s3client = boto3.client("s3")
        self.bucket = bucket

    def get_dataset(self, prefix: str, labels_to_load: Iterable = None) -> List[DVDTFile]:
        file_label_mapping = {}
        for entry in self.s3client.list_objects(Bucket=self.bucket, Prefix=prefix)["Contents"]:
            key = entry["Key"]
            if key.endswith("high.zip"):
                label = key.split("/")[1]
                if label not in file_label_mapping:
                    file_label_mapping[label] = []
                file_label_mapping[label].append(key)

        for label, file_names in file_label_mapping.items():
            print(f"{label}: {len(file_names)} files")

        if labels_to_load is None:
            labels_to_load = file_label_mapping.keys()
        result = []
        for label in labels_to_load:
            for file_name in file_label_mapping[label]:
                try:
                    result.append(self._load_dvdt_file(file_name))
                except:
                    print("could not load", file_name)
        return result

    def _load_dvdt_file(self, file_name) -> DVDTFile:
        print("loading", file_name)
        response = self.s3client.get_object(Bucket=self.bucket, Key=file_name)
        with io.BytesIO(response["Body"].read()) as tf:
            # rewind the file
            tf.seek(0)
            with ZipFile(tf, mode="r") as zip_file:
                for file in zip_file.namelist():
                    if file.endswith(".json") and "/" not in file:
                        with zip_file.open(file) as accel_json:
                            return DVDTFile.from_json(json.loads(accel_json.read()))
