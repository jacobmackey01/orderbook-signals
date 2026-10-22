"""Download Tardis.dev free first-of-month CSV files and record their hashes.

Only the no-key public endpoint is used: datasets for the first day of each month
are downloadable without an API key or account. Raw files are licensed for
internal research use only and must not be committed or redistributed.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

BASE = "https://datasets.tardis.dev/v1"
DATA_TYPES = ("incremental_book_L2", "trades")
MANIFEST = "data/tardis-manifest.json"


def dataset_url(exchange: str, data_type: str, date: str, symbol: str) -> str:
    year, month, day = date.split("-")
    if day != "01":
        raise ValueError("Only first-of-month Tardis files are free without an API key")
    return f"{BASE}/{exchange}/{data_type}/{year}/{month}/{day}/{quote(symbol, safe='')}.csv.gz"


def local_path(root: Path, exchange: str, data_type: str, date: str, symbol: str) -> Path:
    return root/"data/tardis"/exchange/data_type/f"{date}_{symbol}.csv.gz"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_gzip(path: Path) -> int:
    """Tardis recommends size plus successful decompression as the integrity check."""
    lines = 0
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            lines += block.count(b"\n")
    if lines < 2:
        raise ValueError(f"{path} is empty: Tardis returns an empty file when no data exist")
    return lines


def download(root: Path, config: dict, refresh: bool = False) -> list[dict]:
    manifest_path = root/MANIFEST
    known = {r["url"]: r for r in json.loads(manifest_path.read_text())} if manifest_path.exists() else {}
    rows = []
    for symbol in config["symbols"]:
        for date in config["dates"]:
            for data_type in DATA_TYPES:
                url = dataset_url(config["exchange"], data_type, date, symbol)
                path = local_path(root, config["exchange"], data_type, date, symbol)
                if refresh or not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    partial = path.with_suffix(".part")
                    request = Request(url, headers={"User-Agent": "lob-research-replication"})
                    with urlopen(request, timeout=300) as response, partial.open("wb") as out:
                        for block in iter(lambda: response.read(1 << 20), b""):
                            out.write(block)
                    partial.replace(path)
                    acquired = datetime.now(timezone.utc).isoformat(timespec="seconds")
                else:
                    acquired = known.get(url, {}).get("acquired_utc")
                digest = file_sha256(path)
                if url in known and known[url]["sha256"] != digest:
                    raise ValueError(f"Hash mismatch for {url}; inspect before replacing the manifest")
                lines = check_gzip(path)
                rows.append({"exchange": config["exchange"], "symbol": symbol, "date": date, "data_type": data_type,
                             "url": url, "path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                             "sha256": digest, "csv_lines": lines, "acquired_utc": acquired})
                print(f"{symbol} {date} {data_type}: {rows[-1]['bytes']:,} bytes {digest[:12]}", flush=True)
    manifest_path.write_text(json.dumps(rows, indent=2)+"\n")
    return rows
