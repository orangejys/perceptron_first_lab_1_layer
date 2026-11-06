"""Запуск эксперимента одной командой из терминала."""

import argparse
import sys
from pathlib import Path

from perceptron_lab.datasets import load_iris, make_synthetic, train_test_split
from perceptron_lab.model import Perceptron
from perceptron_lab.plotting import save_plots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Простой перцептрон")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--dataset", choices=["synthetic", "iris"], default="synthetic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    try:
        data = (
            load_iris()
            if args.dataset == "iris"
            else make_synthetic(args.samples, args.seed)
        )
        train, test = train_test_split(data, args.seed)
        model = Perceptron(2, args.lr, args.epochs)
        model.train(train.X, train.y)
        save_plots(model, train, test, args.output / args.dataset)
        print(f"Объектов: train={len(train.X)}, test={len(test.X)}")
        print(f"Точность train: {model.score(train.X, train.y):.2%}")
        print(f"Точность test: {model.score(test.X, test.y):.2%}")
        print(f"Веса: {model.w}; bias: {model.bias:.4f}")
        print(f"Ошибки по эпохам: {model.errors_history}")
        print(f"Графики: {args.output / args.dataset}")
    except (ValueError, OSError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2
    return 0
