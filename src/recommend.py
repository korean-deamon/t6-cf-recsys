"""One command: compute the recommendations and write them to ml.recommendations.

Examples:
  python -m src.recommend --run-name pop_v1 --model popularity --score
  python -m src.recommend --run-name als_v1 --model als --student-name team_alpha
  python -m src.recommend --run-name test --model als --limit-users 200 --dry-run
"""
from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import run_pipeline
from .store import open_store


def main() -> None:
    ap = argparse.ArgumentParser(description="T6 collaborative filtering")
    ap.add_argument("--run-name", required=True, help="run name (for the log)")
    ap.add_argument("--model", default="als",
                    choices=["popularity", "itemknn", "als"])
    ap.add_argument("--student-name", default="5-guruh T6", help="team name")
    ap.add_argument("--limit-users", type=int, default=None,
                    help="use fewer users, for a quick test")
    ap.add_argument("--dry-run", action="store_true",
                    help="try it without writing to the database")
    ap.add_argument("--score", action="store_true",
                    help="call eval.score_run() after writing")
    ap.add_argument("--source", choices=["db", "local"], default="db",
                    help="db = university database, local = data/raw.npz")
    ap.add_argument("--data-file", default=None, help="path to the local file")
    args = ap.parse_args()

    cfg = load_config()
    store = open_store(args.source, args.data_file)
    try:
        result = run_pipeline(
            store, cfg,
            model_name=args.model,
            run_name=args.run_name,
            student_name=args.student_name,
            limit_users=args.limit_users,
            dry_run=args.dry_run,
        )
        print(f"\nResult: {result}")
        if args.score and result["run_id"]:
            print("\nScore (eval.score_run):")
            for metric, value, info in store.score_run(result["run_id"]):
                shown = f"{value:.5f}" if value is not None else "-"
                print(f"  {metric:<28} {shown:>10}  {info or ''}")
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
