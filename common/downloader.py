"""
RIPE Atlas Daily Dumps Downloader

Example:
    dl = RipeAtlasDownloader(cache_dir="./ripe_cache")

    # Download files and get paths
    paths = dl.download_range(DataType.PING, "2026-04-29T00:00", "2026-04-29T06:00")

    # Stream records lazily (memory-efficient for multi-GB files)
    for record in dl.iter_records(DataType.TRACEROUTE, "2026-04-29T00:00", "2026-04-29T02:00"):
        print(record)
"""

import bz2
import logging
import polars as pl
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Iterator

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://data-store.ripe.net/datasets/atlas-daily-dumps"


class DataType(str, Enum):
    CONNECTION = "connection"
    DNS = "dns"
    HTTP = "http"
    NTP = "ntp"
    PING = "ping"
    SSLCERT = "sslcert"
    TRACEROUTE = "traceroute"


class RipeAtlasDownloader:
    def __init__(
        self,
        cache_dir: str | Path = "./ripe_cache",
        keep_compressed: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.keep_compressed = keep_compressed
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_range(
        self,
        data_type: DataType,
        start: str | datetime,
        end: str | datetime,
    ) -> Iterator[Path]:
        """Download hourly files for [start, end), yielding each .parquet path as it's ready."""
        for slot in self._hour_slots(self._parse_dt(start), self._parse_dt(end)):
            try:
                yield self._get_file(data_type, slot)
            except Exception as exc:
                log.warning(
                    "Skipping %s @ %s — %s", data_type.value, slot.isoformat(), exc
                )

    def _get_file(self, data_type: DataType, slot: datetime) -> Path:
        """Return local .parquet path, downloading and extracting if not cached."""
        date_dir = self.cache_dir / slot.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{data_type.value}-{slot.strftime('%Y-%m-%dT%H%M')}"
        parquet_path = date_dir / f"{stem}.parquet"

        if parquet_path.exists():
            print(
                f"Cache hit: {parquet_path}",
            )
            return parquet_path

        bz2_path = date_dir / f"{stem}.bz2"
        if not bz2_path.exists():
            url = f"{BASE_URL}/{slot.strftime('%Y-%m-%d')}/{stem}.bz2"

            print(
                f"Downloading {url}",
            )
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(bz2_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    fh.write(chunk)

        print(f"Extracting {bz2_path.name}")
        with bz2.open(bz2_path, "rb") as fin:
            df = pl.scan_ndjson(fin, ignore_errors=True)
            if data_type == DataType.TRACEROUTE:
                df = df.with_columns(
                    hop_count=pl.struct(["result", "dst_addr"]).map_elements(self._hop_count, return_dtype=pl.Int64)
                ).drop("result")
            df.sink_parquet(parquet_path)

        if not self.keep_compressed:
            bz2_path.unlink()

        return parquet_path

    @staticmethod
    def _hour_slots(start: datetime, end: datetime) -> Iterator[datetime]:
        slot = start.replace(minute=0, second=0, microsecond=0)
        while slot < end:
            yield slot
            slot += timedelta(hours=1)

    @staticmethod
    def _parse_dt(value: str | datetime) -> datetime:
        if isinstance(value, datetime):
            return value.replace(tzinfo=value.tzinfo or timezone.utc)
        for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse datetime: {value!r}")

    @staticmethod
    def _hop_count(row) -> int:
        try:
            dst = row["dst_addr"]
            for hop in row["result"]:
                for reply in hop["result"]:
                    if reply.get("from") == dst:
                        return int(hop["hop"])
            # dst never replied — fall back to total hops
            return len(row["result"])
        except:
            return len(row["result"])
