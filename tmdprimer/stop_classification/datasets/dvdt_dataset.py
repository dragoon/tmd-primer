from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import groupby
from typing import Dict, List, Tuple, Callable

import numpy as np
import pandas as pd
import altair as alt

from tmdprimer.stop_classification.datasets import DataFile, Dataset
from tmdprimer.stop_classification.domain.metrics import ClassificationMetric

STOP_LABEL = "stop"


def dvdt_stop_classification_mapping(x):
    if x == STOP_LABEL:
        return 0
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

    def overlap_percent(self, other: "AnnotatedStop") -> float:
        """
        Calculates maximum time different for another annotated stop on both sides
        Used to compute metrics
        """
        overlap = min(self.end_time, other.end_time) - max(self.start_time, other.start_time)
        return overlap / min(self.duration, other.duration)


@dataclass(frozen=True)
class DVDTFile(DataFile):
    start_time: datetime
    end_time: datetime
    num_stations: int
    transport_mode: str
    comment: str
    annotated_stops: List[AnnotatedStop]
    df: pd.DataFrame
    label_mapping_func: Callable[[str], int] = dvdt_stop_classification_mapping

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

    @staticmethod
    def _get_rolling_quantile_accel(window_size, quantile, input_data: pd.Series):
        return input_data.rolling(window_size).quantile(quantile)

    def to_numpy_split_windows(self, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        return super().to_numpy_split_windows(window_size)

    def to_numpy_sliding_windows(self, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        return super().to_numpy_sliding_windows(window_size)

    def get_figure(self, width=800, height=600):
        df = self.df[["label", "linear_accel", "time"]].copy()
        df["label"].replace({self.transport_mode: 1, STOP_LABEL: 0}, inplace=True)
        alt.data_transformers.disable_max_rows()
        base = alt.Chart(df).encode(x="time")

        return alt.layer(
            base.mark_line(color="cornflowerblue").encode(y="linear_accel"),
            base.mark_line(color="orange").encode(y="label"),
        ).properties(width=width, height=height, autosize=alt.AutoSizeParams(type="fit", contains="padding"))

    def get_metrics(
        self, predicted_stops: List[AnnotatedStop], min_allowed_overlap: float = 0.8
    ) -> ClassificationMetric:
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
                if ts.overlap_percent(predicted_stops[i]) > min_allowed_overlap:
                    tp += 1
                    i += 1
                    break
                if predicted_stops[i].start_time > ts.start_time:
                    fn += 1
                    break
                else:
                    fp += 1
                    i += 1
        return ClassificationMetric(tp, fn, fp)

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
    data_files: List[DVDTFile]
