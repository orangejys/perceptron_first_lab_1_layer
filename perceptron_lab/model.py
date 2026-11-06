"""Простой перцептрон: сумма, порог и исправление ошибок."""

import math


class Perceptron:
    def __init__(
        self, n_features: int, learning_rate: float = 0.1, epochs: int = 50
    ) -> None:
        if n_features < 1 or epochs < 1:
            raise ValueError("Число признаков и эпох должно быть положительным.")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Шаг обучения должен быть конечным и больше нуля.")
        self.w = [0.0] * n_features
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.errors_history: list[int] = []

    def activate(self, z: float) -> int:
        return 1 if z >= 0 else 0

    def predict(self, x: list[float]) -> int:
        z = self.bias
        for weight, value in zip(self.w, x, strict=True):
            z += weight * value
        return self.activate(z)

    def train(self, X: list[list[float]], y: list[int]) -> None:
        if not X or len(X) != len(y) or any(label not in (0, 1) for label in y):
            raise ValueError("Нужны непустые X и y одинаковой длины и метки 0/1.")
        self.w = [0.0] * len(self.w)
        self.bias = 0.0
        self.errors_history = []
        for _ in range(self.epochs):
            errors = 0
            for x, target in zip(X, y, strict=True):
                error = target - self.predict(x)
                if error != 0:
                    for j in range(len(self.w)):
                        self.w[j] += self.learning_rate * error * x[j]
                    self.bias += self.learning_rate * error
                    errors += 1
            self.errors_history.append(errors)
            if errors == 0:
                break

    def score(self, X: list[list[float]], y: list[int]) -> float:
        if not X or len(X) != len(y):
            raise ValueError("Нужны непустые X и y одинаковой длины.")
        correct = 0
        for x, target in zip(X, y, strict=True):
            if self.predict(x) == target:
                correct += 1
        return correct / len(y)
