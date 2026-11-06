"""Проверка полного запуска и обработки ошибки."""

from pathlib import Path

import pytest

from perceptron_lab.cli import main


def test_cli(tmp_path: Path) -> None:
    assert main(["--samples", "40", "--output", str(tmp_path)]) == 0
    for name in ("boundary.png", "training.png"):
        assert (tmp_path / "synthetic" / name).read_bytes().startswith(b"\x89PNG")


def test_cli_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--lr", "-1"]) == 2
    assert "Ошибка:" in capsys.readouterr().err
