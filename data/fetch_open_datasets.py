"""Download small, well-known open datasets into data/real/ (idempotent)."""

from __future__ import annotations

import sys
from pathlib import Path

# Project root on path for imports when run as script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx

from src.extractors.csv_extractor import extract_csv

REAL_DIR = Path(__file__).parent / "real"

SOURCES: dict[str, str] = {
    "titanic.csv": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "penguins.csv": "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv",
}


def fetch_one(name: str, url: str, force: bool = False) -> Path:
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    dest = REAL_DIR / name
    if dest.exists() and not force:
        print(f"  Skip (exists): {dest.name}")
        return dest
    print(f"  Downloading {name} …")
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        r = client.get(url)
        r.raise_for_status()
        dest.write_bytes(r.content)
    # Validate as CSV
    extract_csv(dest)
    print(f"  OK → {dest} ({dest.stat().st_size // 1024} KB)")
    return dest


def main() -> None:
    force = "--force" in sys.argv
    print(f"Open datasets → {REAL_DIR}\n")
    for name, url in SOURCES.items():
        fetch_one(name, url, force=force)
    print("\nDone. Run: python3 -m pipelines.demo_open_data_etl")


if __name__ == "__main__":
    main()
