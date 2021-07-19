import abc
from datetime import timedelta
from typing import List

import tensorflow as tf
import pandas as pd
import altair as alt
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler

from tmdprimer.stop_classification.datasets import DataFile
from tmdprimer.stop_classification.datasets.dvdt_dataset import AnnotatedStop


class StopClassificationModel(abc.ABC):

    @abc.abstractmethod
    def _base_classification_df(self, *args, **kwargs):
        pass

    def compute_stops(
            self,
            data_file: DataFile,
            smoothing_window_size: int,
            threshold_probability: float,
            min_stop_duration: timedelta = timedelta(seconds=5),
            min_interval_between_stops: timedelta = timedelta(seconds=10),
    ) -> List[AnnotatedStop]:
        """
        :param data_file: data file to classify
        :param smoothing_window_size: size of the window to take the :threshold_probability percentile from
        :param threshold_probability: percentile of reaching to consider window a stop
        :param min_stop_duration: minimum duration of a stop to prune short jumps
        :param min_interval_between_stops: minimum time between two stops
        :return: stop windows

        NOTE:
        smoothing_window_size, probability, min_stop_duration and min_interval_between_stops
        should be optimized to maximize precision/recall.
        """
        df = self._base_classification_df(data_file=data_file)
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

    def predict_figure(self, data_file: DataFile, width=800, height=600):
        # TODO: try chart with legend differently
        # https://stackoverflow.com/questions/60128774/adding-legend-to-layerd-chart-in-altair
        df = self._base_classification_df(data_file=data_file)
        alt.data_transformers.disable_max_rows()
        base = alt.Chart(df).encode(x="time")

        return alt.layer(
            base.mark_line(color="cornflowerblue").encode(y="linear_accel"),
            base.mark_line(color="orange").encode(y="label"),
            base.mark_line(color="red").encode(y="pred_label"),
        ).properties(width=width, height=height, autosize=alt.AutoSizeParams(type="fit", contains="padding"))

    def predict_figure_smoothed(
            self,
            data_file: DataFile,
            smoothing_window_size: int,
            threshold_probability: float = 0.5,
            width=800,
            height=600,
    ):
        df = data_file.df[["label", "linear_accel", "time"]].copy()
        df["label"] = df["label"].apply(data_file.label_mapping_func)
        df["pred_label"] = 1
        predicted_stops = self.compute_stops(
            data_file,
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


@dataclass(frozen=True)
class SlidingWindowModel(StopClassificationModel):
    model: tf.keras.Model
    window_size: int

    def _base_classification_df(self, data_file: DataFile) -> pd.DataFrame:
        """
        Computes a base dataframe with model classification results
        """
        df = data_file.df[["label", "linear_accel", "time"]].copy()

        df["label"] = df["label"].apply(data_file.label_mapping_func)
        x, _ = data_file.to_numpy_sliding_windows(window_size=self.window_size)
        pred_y = self.model.predict(x)
        df.loc[:, "pred_label"] = (
            pd.Series(pred_y.flatten())
            # reindex to insert NANs in the beginning, adjust for predicting middle element
            .reindex(range(len(pred_y) - len(df) + self.window_size // 2, len(pred_y))).reset_index(drop=True)
        )
        # fill the first window with 1 -- no stop
        df.fillna(1, inplace=True)
        return df


@dataclass(frozen=True)
class SplitWindowModel(StopClassificationModel):
    model: tf.keras.Model
    window_size: int
    scaler: StandardScaler

    def _base_classification_df(self, data_file: DataFile) -> pd.DataFrame:
        """
        Computes a base dataframe with model classification results
        """
        df = data_file.df[["label", "linear_accel", "time"]].copy()

        df["label"] = df["label"].apply(data_file.label_mapping_func)
        x, _ = data_file.to_numpy_split_windows(window_size=self.window_size)
        # shape[0] is the # of windows, shape[1] is the window size, shape[2] is the # of features
        norm_flat_x = self.scaler.transform(x.reshape(x.shape[0]*x.shape[1], x.shape[2]))
        norm_x = norm_flat_x.reshape(x.shape[0], x.shape[1], x.shape[2])
        pred_y = self.model.predict(norm_x, batch_size=1)
        df.loc[:, "pred_label"] = (
            pd.Series(pred_y.flatten())
        )
        # fill the remainder with 1 -- no stop (will be < size of the window at the end)
        df.fillna(1, inplace=True)
        return df

