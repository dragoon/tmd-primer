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
import altair as alt

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
        annotated_stops = [AnnotatedStop.from_json(s) for s in json_dict.get("stops", [])]
        df = pd.DataFrame(json_dict["entries"])
        # add labels to the df
        df["label"] = transport_mode
        for st in json_dict.get("stops", []):
            df.loc[(df["timestamp"] <= st["endTime"]) & (df["timestamp"] >= st["startTime"]), "label"] = STOP_LABEL
        return DVDTFile(start_time, end_time, num_stations, transport_mode, comment, annotated_stops, df)

    def __post_init__(self):
        self.df["linear_accel"] = np.sqrt(self.df["x"] ** 2 + self.df["y"] ** 2 + self.df["z"] ** 2)
        self.df["time"] = pd.to_datetime(self.df["timestamp"], unit="ms")

    def _get_linear_accel_norm(self):
        # clip to 0 - 25
        clipped_accel = np.clip(self.df["linear_accel"], 0, 25)
        # make accel between 0 and 1
        linear_accel_norm = clipped_accel / 25
        return linear_accel_norm

    @staticmethod
    def _get_rolling_quantile_accel(window_size, quantile, input_data: pd.Series):
        return input_data.rolling(window_size).quantile(quantile)

    def _windows_x_y(self, label, stop_label, window_size):
        time_diff_series = self.df["timestamp"].diff()
        linear_accel_norm = self._get_linear_accel_norm()
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
        # now we need to select a single label for a window  -- last label since that's what we will be predicting
        windows_y = np.array([x[-1] for x in windows_y], dtype=int)
        return windows_x, windows_y

    def get_features_labels(self, label, stop_label, median_filter_window=10) -> (np.ndarray, np.ndarray):
        """
        :param label:
        :param stop_label:
        :param median_filter_window: size of the median filter window for pre-processing
        :return: feature and label arrays for the file
        """
        df = self.df[["label", "linear_accel"]].copy()
        df["linear_accel_norm"] = self._get_linear_accel_norm()
        df["median_filter_accel"] = df["linear_accel_norm"].rolling(median_filter_window, center=True).median()
        df = df.dropna()
        df["label"].replace({self.transport_mode: label, STOP_LABEL: stop_label}, inplace=True)
        # fmt: off
        return df[["median_filter_accel", ]].to_numpy(), df[["label", ]].to_numpy()
        # fmt: on

    def get_figure(self, width=800, height=600):
        df = self.df[["label", "linear_accel", "time"]].copy()
        df["label"].replace({self.transport_mode: 1, STOP_LABEL: 0}, inplace=True)
        alt.data_transformers.disable_max_rows()
        base = alt.Chart(df).encode(x="time")

        return alt.layer(
            base.mark_line(color="cornflowerblue").encode(y="linear_accel"),
            base.mark_line(color="orange").encode(y="label"),
        ).properties(width=width, height=height, autosize=alt.AutoSizeParams(type="fit", contains="padding"))


class DVDTDataset:
    s3client = None
    bucket: str
    dvdt_files: List[DVDTFile]

    def __init__(self, bucket: str, prefix: str, labels_to_load: Iterable = None):
        self.s3client = boto3.client("s3")
        self.bucket = bucket
        self.dvdt_files = self._get_dataset(prefix, labels_to_load)

    def _get_dataset(self, prefix: str, labels_to_load: Iterable = None) -> List[DVDTFile]:
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

    def to_tfds(self, label, window_size, stop_label=0) -> tf.data.Dataset:
        x = []
        y = []
        for f in self.dvdt_files:
            windows_x, windows_y = f._windows_x_y(label, stop_label, window_size)
            x.append(windows_x)
            y.append(windows_y)
        return tf.data.Dataset.from_tensor_slices((np.concatenate(x), np.concatenate(y)))

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

    def predict(self, model: tf.keras.Model):
        pass
