import io
import json
from typing import Iterable
from zipfile import ZipFile

import boto3
import pandas as pd

from tmdprimer.stop_classification.data_loaders import DataLoader
from tmdprimer.stop_classification.datasets.dvdt_dataset import DVDTFile, DVDTDataset
from tmdprimer.stop_classification.datasets.sensorlog_dataset import SensorLogFile, SensorLogDataset


class DVDTS3DataLoader(DataLoader):
    def load_dataset(self, bucket, path: str, labels_to_load: Iterable = None) -> DVDTDataset:
        s3client = boto3.client("s3")
        files = []
        for entry in s3client.list_objects(Bucket=bucket, Prefix=path)["Contents"]:
            file_name = entry["Key"]
            if file_name.endswith("high.zip"):
                data_file = self.load_file(bucket=bucket, file_name=file_name, s3client=s3client)
                if labels_to_load is None or data_file.transport_mode in labels_to_load:
                    files.append(data_file)
        return DVDTDataset(files)

    def load_file(self, bucket, file_name, s3client=None) -> DVDTFile:
        if s3client is None:
            s3client = boto3.client("s3")
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


class SensorLogS3DataLoader(DataLoader):
    def load_dataset(self, bucket, path: str, labels_to_load: Iterable = None) -> SensorLogDataset:
        s3client = boto3.client("s3")
        files = []
        for entry in s3client.list_objects(Bucket=bucket, Prefix=path)["Contents"]:
            file_name = entry["Key"]
            if file_name.endswith(".csv"):
                data_file = self.load_file(bucket=bucket, file_name=file_name, s3client=s3client)
                files.append(data_file)
        return SensorLogDataset(files)

    def load_file(self, bucket, file_name, s3client=None) -> SensorLogFile:
        if s3client is None:
            s3client = boto3.client("s3")
        print("loading", file_name)
        response = s3client.get_object(Bucket=bucket, Key=file_name)
        with io.BytesIO(response["Body"].read()) as datafile:
            return SensorLogFile.from_csv(pd.read_csv(datafile, sep=";"))
