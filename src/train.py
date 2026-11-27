"""Train the model and write the top-10 list (the task asks for `src/train`).

This is a short alias for `src.recommend`: the model is trained on the
train period and the result is written to `ml.recommendations`.
Scoring is a separate step (`--score`).

Usage:
  python -m src.train --model als --run-name als_v1
  python -m src.train --model itemknn --run-name knn_v1 --score
"""
from __future__ import annotations

from .recommend import main as recommend_main


def main() -> None:
    recommend_main()


if __name__ == "__main__":
    main()
