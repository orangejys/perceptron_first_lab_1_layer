"""Проверки обязательных методов перцептрона."""

import pytest

from perceptron_lab.model import Perceptron


def test_activate() -> None:
    model = Perceptron(2)
    assert model.activate(-1) == 0
    assert model.activate(0) == 1
    assert model.activate(1) == 1


def test_weights() -> None:
    model = Perceptron(3)
    assert model.w == [0.0, 0.0, 0.0]
    assert model.bias == 0


def test_predict() -> None:
    model = Perceptron(2)
    model.w = [1.0, 1.0]
    model.bias = -1.5
    assert model.predict([1, 1]) == 1
    assert model.predict([0, 1]) == 0


def test_train() -> None:
    X = [[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]]
    y = [0, 0, 1, 1]
    model = Perceptron(2)
    model.train(X, y)
    assert model.score(X, y) == 1.0
    assert model.errors_history[-1] == 0


def test_score() -> None:
    model = Perceptron(2)
    assert model.score([[0, 0], [1, 1]], [0, 1]) == 0.5


def test_bad_learning_rate() -> None:
    with pytest.raises(ValueError, match="Шаг"):
        Perceptron(2, learning_rate=0)
