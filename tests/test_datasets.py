"""Проверка случайных данных и локального Iris."""

from perceptron_lab.datasets import load_iris, make_synthetic, train_test_split
from perceptron_lab.model import Perceptron


def test_synthetic() -> None:
    data = make_synthetic(200)
    assert len(data.X) == len(data.y) == 200
    assert all((x[0] >= 0) == target for x, target in zip(data.X, data.y, strict=True))


def test_iris() -> None:
    train, test = train_test_split(load_iris())
    assert len(train.X) == 70
    assert len(test.X) == 30
    model = Perceptron(2)
    model.train(train.X, train.y)
    assert model.score(test.X, test.y) >= 0.9
