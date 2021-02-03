import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List

import numpy as np
import tensorflow as tf
import pandas as pd
import io
from zipfile import ZipFile
import boto3

from tmdprimer.datagen import make_sliding_windows

STOP_LABEL = "stop"


@dataclass
class AnnotatedStop:
    start_time: datetime
    end_time: datetime

    @classmethod
    def from_json(cls, json_dict: Dict):
        start_time = datetime.fromtimestamp(json_dict["startTime"] / 1000)
        end_time = datetime.fromtimestamp(json_dict["endTime"] / 1000)
        return AnnotatedStop(start_time, end_time)

    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time


@dataclass
class DVDTFile:
    start_time: datetime
    end_time: datetime
    num_stations: int
    transport_mode: str
    comment: str
    annotated_stops: List[AnnotatedStop]
    df: pd.DataFrame

    @classmethod
    def from_json(cls, json_dict: Dict):
        metadata = json_dict["metadata"]
        start_time = datetime.fromtimestamp(metadata["timestamp"] / 1000)
        end_time = datetime.fromtimestamp(metadata["endtime"] / 1000)
        num_stations = metadata["numberStations"]
        transport_mode = metadata["transportMode"]
        comment = metadata["comment"]
        annotated_stops = [AnnotatedStop.from_json(s) for s in json_dict["stops"]]
        df = pd.DataFrame(json_dict["entries"])
        # add labels to the df
        df["label"] = transport_mode
        for st in json_dict["stops"]:
            df.loc[(df["timestamp"] < st["endTime"]) & (df["timestamp"] > st["startTime"]), "label"] = STOP_LABEL
        return DVDTFile(start_time, end_time, num_stations, transport_mode, comment, annotated_stops, df)

    def _get_linear_accel(self):
        linear_accel_series = np.sqrt(self.df["x"] ** 2 + self.df["y"] ** 2 + self.df["z"] ** 2)
        # clip to 0 - 25
        clipped_accel = np.clip(linear_accel_series, 0, 25)
        # make accel between 0 and 1
        linear_accel_norm = (clipped_accel - np.min(clipped_accel)) / (np.max(clipped_accel) - np.min(clipped_accel))
        return linear_accel_norm

    @staticmethod
    def _get_rolling_quantile_accel(window_size, quantile, input_data: pd.Series):
        return input_data.rolling(window_size).quantile(quantile)

    def _windows_x_y(self, label, stop_label, window_size):
        time_diff_series = self.df["timestamp"].diff()
        linear_accel_norm = self._get_linear_accel()
        rolling_accel = self._get_rolling_quantile_accel(window_size, 0.5, linear_accel_norm)
        df = pd.DataFrame({"rolling": rolling_accel, "linear": linear_accel_norm, "label": self.df["label"]}).dropna()

        # transform label values to integers
        labels = df["label"].replace({self.transport_mode: label, STOP_LABEL: stop_label}, inplace=False).to_numpy()

        # fmt: off
        windows_x = make_sliding_windows(
            df[["linear", ]].to_numpy(), window_size, overlap_size=window_size - 1, flatten_inside_window=False
        )
        # fmt: on
        windows_y = make_sliding_windows(labels, window_size, overlap_size=window_size - 1, flatten_inside_window=False)
        # now we need to select a single label for a window based on the mix of labels in it
        windows_y = np.median(windows_y, axis=1).astype(int)
        return windows_x, windows_y

    def to_tfds(self, label, window_size, stop_label=0) -> tf.data.Dataset:
        windows_x, windows_y = self._windows_x_y(label, stop_label, window_size)
        return tf.data.Dataset.from_tensor_slices((windows_x, windows_y))

    def to_cnn_tfds(
        self,
        label,
        window_size,
        n_steps,
        stop_label=0,
    ):
        """
        :param label:
        :param stop_label:
        :param window_size:
        :param n_steps: should be able to divide window_size by n_steps with no remainder
        :return:
        """
        windows_x, windows_y = self._windows_x_y(label, stop_label, window_size)
        if window_size % n_steps != 0:
            raise Exception("Window_size should divide by n_steps without remainder")
        n_length = window_size // n_steps
        windows_x = windows_x.reshape((windows_x.shape[0], n_steps, n_length, windows_x.shape[2]))
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
                result.append(self._load_dvdt_file(file_name))
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
