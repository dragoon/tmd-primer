from dataclasses import dataclass
import pandas as pd
import numpy as np
import altair as alt

from tmdprimer.datagen import make_sliding_windows


@dataclass(frozen=True)
class SensorLogFile:
    df: pd.DataFrame

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

    def _get_linear_accel_norm(self):
        # clip to 0 - 25
        clipped_accel = np.clip(self.df["linear_accel"], 0, 25)
        # make accel between 0 and 1
        linear_accel_norm = clipped_accel / 25
        return linear_accel_norm

    def to_numpy_sliding_windows(self, window_size: int):
        linear_accel_norm = self._get_linear_accel_norm()
        df = pd.DataFrame({"linear": linear_accel_norm, "label": self.df["label"]}).dropna()

        # fmt: off
        windows_x = make_sliding_windows(
            df[["linear", ]].to_numpy(), window_size, overlap_size=window_size - 1, flatten_inside_window=False
        )
        labels = df["label"]
        # fmt: on
        windows_y = make_sliding_windows(labels, window_size, overlap_size=window_size - 1, flatten_inside_window=False)
        # now we need to select a single label for a window  -- last label since that's what we will be predicting
        windows_y = np.array([x[-1] for x in windows_y], dtype=int)
        return windows_x, windows_y

    def get_figure(self, width=800, height=600):
        df = self.df[["label", "linear_accel", "time"]].copy()
        alt.data_transformers.disable_max_rows()
        base = alt.Chart(df).encode(x="time")

        return alt.layer(
            base.mark_line(color="cornflowerblue").encode(y="linear_accel"),
            base.mark_line(color="orange").encode(y="label"),
        ).properties(width=width, height=height, autosize=alt.AutoSizeParams(type="fit", contains="padding"))
