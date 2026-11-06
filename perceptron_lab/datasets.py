"""Два источника двумерных данных и разделение на train/test."""

import csv
import random
from dataclasses import dataclass
from importlib.resources import files


@dataclass
class Dataset:
    X: list[list[float]]
    y: list[int]


def make_synthetic(samples: int = 200, seed: int = 42) -> Dataset:
    if samples < 8:
        raise ValueError("Нужно хотя бы 8 объектов.")
    rng = random.Random(seed)
    X: list[list[float]] = []
    y: list[int] = []
    for i in range(samples):
        label = 0 if i < samples // 2 else 1
        center = -3 if label == 0 else 3
        X.append([center + rng.uniform(-0.75, 0.75) for _ in range(2)])
        y.append(label)
    return Dataset(X, y)


def load_iris() -> Dataset:
    X: list[list[float]] = []
    y: list[int] = []
    with (
        files("perceptron_lab")
        .joinpath("data/iris.csv")
        .open(encoding="utf-8") as stream
    ):
        for row in csv.DictReader(stream):
            if row["species"] in ("setosa", "versicolor"):
                X.append([float(row["petal_length"]), float(row["petal_width"])])
                y.append(0 if row["species"] == "setosa" else 1)
    return Dataset(X, y)


def train_test_split(data: Dataset, seed: int = 42) -> tuple[Dataset, Dataset]:
    rng = random.Random(seed)
    train = Dataset([], [])
    test = Dataset([], [])
    for label in (0, 1):
        indices = [i for i in range(len(data.y)) if data.y[i] == label]
        rng.shuffle(indices)
        count = max(1, round(len(indices) * 0.3))
        for position, i in enumerate(indices):
            part = test if position < count else train
            part.X.append(data.X[i])
            part.y.append(data.y[i])
    return train, test
