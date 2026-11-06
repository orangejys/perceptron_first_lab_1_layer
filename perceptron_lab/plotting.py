"""Два обязательных графика: граница решения и ошибки по эпохам."""

from pathlib import Path

from matplotlib.figure import Figure

from perceptron_lab.datasets import Dataset
from perceptron_lab.model import Perceptron


def save_plots(model: Perceptron, train: Dataset, test: Dataset, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig = Figure()
    ax = fig.subplots()
    for data, marker, name in ((train, "o", "train"), (test, "x", "test")):
        ax.scatter(
            [x[0] for x in data.X],
            [x[1] for x in data.X],
            c=data.y,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            marker=marker,
            label=name,
        )
    points = train.X + test.X
    left = min(x[0] for x in points) - 0.5
    right = max(x[0] for x in points) + 0.5
    if model.w[1] != 0:
        xs = [left, right]
        ys = [-(model.w[0] * x + model.bias) / model.w[1] for x in xs]
        ax.plot(xs, ys, color="black", label="z = 0")
    elif model.w[0] != 0:
        ax.axvline(-model.bias / model.w[0], color="black", label="z = 0")
    ax.set(
        xlabel="Признак 1",
        ylabel="Признак 2",
        title="Граница решения",
        xlim=(left, right),
        ylim=(min(x[1] for x in points) - 0.5, max(x[1] for x in points) + 0.5),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "boundary.png")
    fig = Figure()
    ax = fig.subplots()
    ax.plot(range(1, len(model.errors_history) + 1), model.errors_history, "o-")
    ax.set(xlabel="Эпоха", ylabel="Число ошибок", title="Ошибки во время обучения")
    fig.tight_layout()
    fig.savefig(output / "training.png")
