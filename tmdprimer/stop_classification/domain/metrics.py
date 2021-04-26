from dataclasses import dataclass


@dataclass
class ClassificationMetric:
    tp: int
    fn: int
    fp: int
