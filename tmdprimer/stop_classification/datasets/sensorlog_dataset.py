import io
from dataclasses import dataclass
from itertools import groupby
from typing import List, Dict, Callable

import boto3
import pandas as pd
import numpy as np
import altair as alt

from tmdprimer.stop_classification.datasets import DataFile, Dataset, identity


@dataclass(frozen=True)
class SensorLogFile(DataFile):
    df: pd.DataFrame
    label_mapping_func: Callable[[str], int] = identity

    @classmethod
    def from_csv(cls, csv: pd.DataFrame):
        df = csv[
            ["motionUserAccelerationX(G)", "motionUserAccelerationY(G)", "motionUserAccelerationZ(G)", "label(N)"]
        ].copy()
        df.rename(
            columns={
                "motionUserAccelerationX(G)": "x",
                "motionUserAccelerationY(G)": "y",
                "motionUserAccelerationZ(G)": "z",
                "label(N)": "label",
            },
            inplace=True,
        )
        df["time"] = pd.to_datetime(csv["loggingTime(txt)"], infer_datetime_format=True)
        return SensorLogFile(df)

    def __post_init__(self):
        self.df["linear_accel"] = np.sqrt(self.df["x"] ** 2 + self.df["y"] ** 2 + self.df["z"] ** 2)

    def get_figure(self, width=800, height=600):
        df = self.df[["label", "linear_accel", "time"]].copy()
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
            if key == 1:
                key = "stop"
            else:
                key = "transport"
            result.append({"mode": key, "duration": g_list[-1][1] - g_list[0][1]})
        return result


@dataclass(frozen=True)
class SensorLogDataset(Dataset):
    data_files: List[SensorLogFile]

    @staticmethod
    def load(bucket: str, path: str):
        s3client = boto3.client("s3")
        dvdt_files = SensorLogDataset._get_dataset(s3client, bucket, path)
        return SensorLogDataset(dvdt_files)

    @staticmethod
    def _get_dataset(s3client, bucket: str, path: str) -> List[SensorLogFile]:
        result = []
        for entry in s3client.list_objects(Bucket=bucket, Prefix=path)["Contents"]:
            file_name = entry["Key"]
            if file_name.endswith(".csv"):
                data_file = SensorLogDataset._load_data_file(s3client, bucket, file_name)
                result.append(data_file)
        return result

    @staticmethod
    def _load_data_file(s3client, bucket, file_name) -> SensorLogFile:
        print("loading", file_name)
        response = s3client.get_object(Bucket=bucket, Key=file_name)
        with io.BytesIO(response["Body"].read()) as datafile:
            return SensorLogFile.from_csv(pd.read_csv(datafile, sep=';'))
