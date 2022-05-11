from dataclasses import dataclass
from datetime import datetime
from itertools import groupby
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np
import pandas as pd
import altair as alt

from tmdprimer.stop_classification.datasets import DataFile, Dataset, AnnotatedStop

STOP_LABEL = "stop"


def dvdt_stop_classification_mapping(x):
    if x == STOP_LABEL:
        return 0
    return 1


@dataclass(frozen=True)
class DVDTFile(DataFile):
    start_time: datetime
    end_time: datetime
    num_stations: int
    transport_mode: str
    comment: str
    annotated_stops: List[AnnotatedStop]
    df: pd.DataFrame
    # df with real-time classification results from the phone
    classification_df: Optional[pd.DataFrame]
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
        classification_df = pd.DataFrame(json_dict.get("stop_classification_entries", []))
        # add labels to the df
        df["label"] = transport_mode
        for st in json_dict.get("stops", []):
            df.loc[(df["timestamp"] <= st["endTime"]) & (df["timestamp"] >= st["startTime"]), "label"] = STOP_LABEL
        return DVDTFile(start_time, end_time, num_stations, transport_mode,
                        comment, annotated_stops, df, classification_df)

    def __post_init__(self):
        if len(self.df) > 0:
            self.df["linear_accel"] = np.sqrt(self.df["x"] ** 2 + self.df["y"] ** 2 + self.df["z"] ** 2)
            self.df["time"] = pd.to_datetime(self.df["timestamp"], unit="ms")
            self.df["time_diff"] = self.df["timestamp"].diff()
        if len(self.classification_df) > 0:
            self.classification_df["time"] = pd.to_datetime(self.df["timestamp"], unit="ms")

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
