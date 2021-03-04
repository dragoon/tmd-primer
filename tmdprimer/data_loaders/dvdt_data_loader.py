import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import groupby
from typing import Dict, Iterable, List, Tuple, Callable

import numpy as np
import tensorflow as tf
import pandas as pd
import io
from zipfile import ZipFile
import boto3
import altair as alt

from tmdprimer.data_loaders import DataFile, Dataset
from tmdprimer.datagen import make_sliding_windows

STOP_LABEL = "stop"


def dvdt_stop_classification_mapping(x):
    if x == STOP_LABEL:
        return STOP_LABEL
    return 1


@dataclass(frozen=True)
class AnnotatedStop:
    start_time: datetime
    end_time: datetime

    @classmethod
    def from_json(cls, json_dict: Dict):
        start_time = datetime.utcfromtimestamp(json_dict["startTime"] / 1000)
        end_time = datetime.utcfromtimestamp(json_dict["endTime"] / 1000)
        return AnnotatedStop(start_time, end_time)

    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time

    def max_margin(self, other: "AnnotatedStop") -> timedelta:
        """
        Calculates maximum time different for another annotated stop on both sides
        Used to compute metrics
        """
        return max(abs(self.start_time - other.start_time), abs(self.end_time - other.end_time))


@dataclass(frozen=True)
class DVDTFile(DataFile):
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
        start_time = datetime.utcfromtimestamp(metadata["timestamp"] / 1000)
        end_time = datetime.utcfromtimestamp(metadata["endtime"] / 1000)
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
        self.df["time_diff"] = self.df["timestamp"].diff()

    def _get_linear_accel_norm(self):
        # clip to 0 - 25
        clipped_accel = np.clip(self.df["linear_accel"], 0, 25)
        # make accel between 0 and 1
        linear_accel_norm = clipped_accel / 25
        return linear_accel_norm

    @staticmethod
    def _get_rolling_quantile_accel(window_size, quantile, input_data: pd.Series):
        return input_data.rolling(window_size).quantile(quantile)

    def to_numpy_sliding_windows(
        self, window_size: int, label_mapping_func: Callable[[str], int] = dvdt_stop_classification_mapping
    ) -> Tuple[np.ndarray, np.ndarray]:
        linear_accel_norm = self._get_linear_accel_norm()
        df = pd.DataFrame({"linear": linear_accel_norm, "label": self.df["label"]}).dropna()

        # transform label values to integers
        labels = df["label"].apply(label_mapping_func).to_numpy()

        # fmt: off
        windows_x = make_sliding_windows(
            df[["linear", ]].to_numpy(), window_size, overlap_size=window_size - 1, flatten_inside_window=False
        )
        # fmt: on
        windows_y = make_sliding_windows(labels, window_size, overlap_size=window_size - 1, flatten_inside_window=False)
        # now we need to select a single label for a window  -- last label since that's what we will be predicting
        windows_y = np.array([x[-1] for x in windows_y], dtype=int)
        return windows_x, windows_y

    def get_figure(self, width=800, height=600):
        df = self.df[["label", "linear_accel", "time"]].copy()
        df["label"].replace({self.transport_mode: 1, STOP_LABEL: 0}, inplace=True)
        alt.data_transformers.disable_max_rows()
        base = alt.Chart(df).encode(x="time")

        return alt.layer(
            base.mark_line(color="cornflowerblue").encode(y="linear_accel"),
            base.mark_line(color="orange").encode(y="label"),
        ).properties(width=width, height=height, autosize=alt.AutoSizeParams(type="fit", contains="padding"))

    def _base_classification_df(self, model: tf.keras.Model, window_size: int) -> pd.DataFrame:
        """
        Computes a base dataframe with model classification results
        """
        df = self.df[["label", "linear_accel", "time"]].copy()

        def label_mapping_func(label):
            if label == STOP_LABEL:
                return 0
            return 1

        df["label"] = df["label"].apply(label_mapping_func)
        x, y = self.to_numpy_sliding_windows(window_size=window_size, label_mapping_func=label_mapping_func)
        pred_y = model.predict(x)
        df.loc[:, "pred_label"] = (
            # reindex to insert NANs in the beginning
            pd.Series(pred_y.flatten())
            .reindex(range(len(pred_y) - len(df), len(pred_y)))
            .reset_index(drop=True)
        )
        # fill the first window with 1 -- no stop
        df.fillna(1, inplace=True)
        return df

    def compute_stops(
        self,
        model: tf.keras.Model,
        model_window_size: int,
        smoothing_window_size: int,
        threshold_probability: float,
        min_stop_duration: timedelta = timedelta(seconds=5),
        min_interval_between_stops: timedelta = timedelta(seconds=10),
    ) -> List[AnnotatedStop]:
        """
        :param model: model to use for classification
        :param model_window_size: window size of the model
        :param smoothing_window_size: size of the window to take the :threshold_probability percentile from
        :param threshold_probability: percentile of reaching to consider window a stop
        :param min_stop_duration: minimum duration of a stop to prune short jumps
        :param min_interval_between_stops: minimum time between two stops
        :return: stop windows

        NOTE:
        smoothing_window_size, probability, min_stop_duration and min_interval_between_stops
        should be optimized to maximize precision/recall.
        """
        df = self._base_classification_df(model, model_window_size)
        # make sure length is the same -- 1 prediction for the first elements
        stop_windows = [{}]
        current_labels = [
            (row[0], row[1]) for row in df[["pred_label", "time"]][:smoothing_window_size].itertuples(index=False)
        ]
        current_sum = sum(x[0] for x in current_labels)
        # smooth with a rolling window -- take 0.8 percentile
        for row in df[["pred_label", "time"]][smoothing_window_size:].itertuples(index=False):
            pred_label = row[0]
            if current_sum / smoothing_window_size < threshold_probability and "start" not in stop_windows[-1]:
                # 0 is a stop => open the stop window
                # find the FIRST stop label
                first_stop_time = next(x[1] for x in current_labels if x[0] < 0.5)
                stop_windows[-1]["start"] = first_stop_time.to_pydatetime()
            elif (
                current_sum / smoothing_window_size >= threshold_probability
                and "start" in stop_windows[-1]
                and "end" not in stop_windows[-1]
            ):
                # find the LAST stop label
                try:
                    last_stop_time = next(x[1] for x in reversed(current_labels) if x[0] < 0.5)
                except StopIteration:
                    last_stop_time = current_labels[-1][1]
                stop_windows[-1]["end"] = last_stop_time.to_pydatetime()
                stop_windows.append({})
            current_sum = current_sum + pred_label - current_labels.pop(0)[0]
            current_labels.append((pred_label, row[1]))
        # check if the last stop window is not closed
        if "start" in stop_windows[-1]:
            stop_windows[-1]["end"] = current_labels[-1][1].to_pydatetime()
            stop_windows.append({})

        annotated_stops = [AnnotatedStop(s["start"], s["end"]) for s in stop_windows[:-1]]

        # 1. merge close or overlapping stops
        def merge_close_stops(stops: List[AnnotatedStop]):
            result = list(stops[:1])
            for s1, s2 in zip(stops, stops[1:]):
                if s2.start_time - s1.end_time < min_interval_between_stops:
                    # merge
                    result[-1] = AnnotatedStop(s1.start_time, s2.end_time)
                else:
                    result.append(s2)
            return result

        while True:
            merged_annotated_stops = merge_close_stops(annotated_stops)
            if len(merged_annotated_stops) < len(annotated_stops):
                annotated_stops = merged_annotated_stops
            else:
                break

        # 2. prune short stops
        annotated_stops = [s for s in annotated_stops if s.duration > min_stop_duration]
        return annotated_stops

    def predict_figure(self, model: tf.keras.Model, window_size: int, width=800, height=600):
        # TODO: try chart with legend differently
        # https://stackoverflow.com/questions/60128774/adding-legend-to-layerd-chart-in-altair
        df = self._base_classification_df(model, window_size)
        alt.data_transformers.disable_max_rows()
        base = alt.Chart(df).encode(x="time")

        return alt.layer(
            base.mark_line(color="cornflowerblue").encode(y="linear_accel"),
            base.mark_line(color="orange").encode(y="label"),
            base.mark_line(color="red").encode(y="pred_label"),
        ).properties(width=width, height=height, autosize=alt.AutoSizeParams(type="fit", contains="padding"))

    def predict_figure_smoothed(
        self,
        model: tf.keras.Model,
        window_size: int,
        smoothing_window_size: int,
        threshold_probability: float = 0.5,
        width=800,
        height=600,
    ):
        df = self.df[["label", "linear_accel", "time"]].copy()
        df["label"].replace({self.transport_mode: 1, STOP_LABEL: 0}, inplace=True)
        df["pred_label"] = 1
        predicted_stops = self.compute_stops(
            model,
            model_window_size=window_size,
            smoothing_window_size=smoothing_window_size,
            threshold_probability=threshold_probability,
        )
        for s in predicted_stops:
            df.loc[(df["time"] <= s.end_time) & (df["time"] >= s.start_time), "pred_label"] = 0

        base = alt.Chart(df).encode(x="time")
        return alt.layer(
            base.mark_line(color="cornflowerblue").encode(y="linear_accel"),
            base.mark_line(color="orange").encode(y="label"),
            base.mark_line(color="red").encode(y="pred_label"),
        ).properties(width=width, height=height, autosize=alt.AutoSizeParams(type="fit", contains="padding"))

    def get_precision_recall(
        self, predicted_stops: List[AnnotatedStop], allowed_margin: timedelta = timedelta(seconds=3)
    ) -> Tuple[float, float]:
        # get stop timespans
        fp = 0
        tp = 0
        fn = 0
        true_stops = self.annotated_stops

        i = 0
        for ts in true_stops:
            if i == len(predicted_stops):
                # make sure to increment false negatives if there are no more predicted stops
                fn += 1
            while i < len(predicted_stops):
                if ts.max_margin(predicted_stops[i]) < allowed_margin:
                    tp += 1
                    i += 1
                    break
                if predicted_stops[i].start_time > ts.start_time:
                    fn += 1
                    break
                else:
                    fp += 1
                    i += 1
        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = 0
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = 0
        return precision, recall

    @property
    def stop_durations(self) -> List[Dict]:
        """
        collection all durations of stops and non-stops
        :return: dict with labels as keys and durations list
        """
        result = []
        for key, group in groupby(self.df[["label", "time"]].values.tolist(), key=lambda x: x[0]):
            g_list = list(group)
            if key == "stop":
                key = f"stop_{self.transport_mode}"
            result.append({"mode": key, "duration": g_list[-1][1] - g_list[0][1]})
        return result


@dataclass(frozen=True)
class DVDTDataset(Dataset):
    dvdt_files: List[DVDTFile]

    @staticmethod
    def load(bucket: str, path: str, labels_to_load: Iterable = None):
        s3client = boto3.client("s3")
        dvdt_files = DVDTDataset._get_dataset(s3client, bucket, path, labels_to_load)
        return DVDTDataset(dvdt_files)

    @staticmethod
    def _get_dataset(s3client, bucket: str, path: str, labels_to_load: Iterable = None) -> List[DVDTFile]:
        result = []
        for entry in s3client.list_objects(Bucket=bucket, Prefix=path)["Contents"]:
            file_name = entry["Key"]
            if file_name.endswith("high.zip"):
                dvdt_file = DVDTDataset._load_dvdt_file(s3client, bucket, file_name)
                if labels_to_load is None or dvdt_file.transport_mode in labels_to_load:
                    result.append(dvdt_file)
        return result

    def to_window_tfds(
        self, window_size, label_mapping_func: Callable[[str], int] = dvdt_stop_classification_mapping
    ) -> tf.data.Dataset:
        def scaled_iter():
            for f in self.dvdt_files:
                windows_x, windows_y = f.to_numpy_sliding_windows(window_size, label_mapping_func)
                yield from zip(windows_x, windows_y)

        return tf.data.Dataset.from_generator(
            scaled_iter,
            output_types=(tf.float32, tf.int32),
            output_shapes=(tf.TensorShape((window_size, 1)), tf.TensorShape((1,))),
        )

    def to_window_numpy(
        self, window_size, label_mapping_func: Callable[[str], int] = dvdt_stop_classification_mapping
    ) -> Tuple[np.ndarray, np.ndarray]:
        result_x = None
        result_y = None
        for f in self.dvdt_files:
            windows_x, windows_y = f.to_numpy_sliding_windows(window_size, label_mapping_func)
            if result_x is not None:
                result_x = np.append(result_x, windows_x, axis=0)
                result_y = np.append(result_y, windows_y, axis=0)
            else:
                result_x = windows_x
                result_y = windows_y
        return result_x, result_y

    @staticmethod
    def _load_dvdt_file(s3client, bucket, file_name) -> DVDTFile:
        print("loading", file_name)
        response = s3client.get_object(Bucket=bucket, Key=file_name)
        with io.BytesIO(response["Body"].read()) as datafile:
            # rewind the file
            datafile.seek(0)
            with ZipFile(datafile, mode="r") as zip_file:
                for file in zip_file.namelist():
                    if file.endswith(".json") and "/" not in file:
                        with zip_file.open(file) as accel_json:
                            return DVDTFile.from_json(json.loads(accel_json.read()))

    @property
    def stop_durations_df(self) -> pd.DataFrame:
        """
        collection all durations of stops and non-stops
        :return: dict with labels as keys and durations list
        """
        result = []
        for f in self.dvdt_files:
            # ignore first / last durations just in case
            result.extend(f.stop_durations[1:-1])
        return pd.DataFrame(result)
