"""Load settings and secrets.

The password is never kept in the code: it is read from the .env file
or from environment variables only.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_ENV = ("PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD")


def load_dotenv(path: str | Path | None = None) -> None:
    """A small .env reader, no extra library needed."""
    path = Path(path or ROOT / ".env")
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(path: str | Path | None = None) -> dict:
    load_dotenv()
    with open(path or ROOT / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def db_params() -> dict:
    load_dotenv()
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise SystemExit(
            "Missing environment variables: "
            + ", ".join(missing)
            + "\nCopy `.env.example` to `.env` and fill in the values from the mentor."
        )
    return {
        "host": os.environ["PGHOST"],
        "port": int(os.environ["PGPORT"]),
        "dbname": os.environ["PGDATABASE"],
        "user": os.environ["PGUSER"],
        "password": os.environ["PGPASSWORD"],
    }
