"""Immutable, point-in-time snapshots of the Nasdaq symbol directories."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from alphascreener.data import paths
from alphascreener.data.locking import exclusive_file_lock

_SCHEMA_VERSION = 1
_NEW_YORK = ZoneInfo("America/New_York")
_SOURCE_NAMES = ("nasdaqlisted", "otherlisted")
_FILE_CREATION_RE = re.compile(r"^File Creation Time:\s*(?P<stamp>\d{8}\d{2}:\d{2})(?:\||$)")
_FRAME_SCHEMA = {
    "symbol": pl.String,
    "security_name": pl.String,
    "exchange": pl.String,
    "security_type": pl.String,
    "flags": pl.List(pl.String),
    "source": pl.String,
    "file_creation_time": pl.Datetime(time_unit="us", time_zone="UTC"),
    "available_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}
_NASDAQ_REQUIRED_COLUMNS = {
    "Symbol",
    "Security Name",
    "Market Category",
    "Test Issue",
    "Financial Status",
    "ETF",
    "NextShares",
}
_OTHER_REQUIRED_COLUMNS = {
    "ACT Symbol",
    "Security Name",
    "Exchange",
    "ETF",
    "Test Issue",
    "NASDAQ Symbol",
}
_OTHER_EXCHANGES = {
    "A": "NYSE American",
    "N": "NYSE",
    "P": "NYSE Arca",
    "V": "IEX",
    "Z": "Cboe BZX",
}


class UniverseSnapshotError(RuntimeError):
    """Base error for invalid or unavailable universe snapshots."""


class SnapshotIntegrityError(UniverseSnapshotError):
    """Raised when an immutable snapshot no longer matches its manifest."""


@dataclass(frozen=True)
class UniverseSnapshot:
    """A verified point-in-time universe snapshot and its normalized records."""

    as_of: date
    digest: str
    available_at: datetime
    path: Path
    frame: pl.DataFrame

    @property
    def symbols(self) -> tuple[str, ...]:
        """Return normalized symbols in deterministic order."""
        return tuple(self.frame["symbol"].to_list())


def get_universe_snapshot_dir() -> Path:
    """Return the root used for immutable symbol-directory snapshots."""
    return paths.get_data_home() / "data" / "universe"


def parse_symbol_directories(
    nasdaqlisted: str | bytes,
    otherlisted: str | bytes,
    *,
    available_at: datetime,
) -> pl.DataFrame:
    """Normalize two complete official directories without filtering history.

    ``available_at`` is the observation time, not the timestamp printed in the
    directory footer.  Requiring both prevents a file fetched later from being
    treated as if it had been locally available when Nasdaq generated it.
    """
    observed_at = _aware_utc(available_at, name="available_at")
    records: list[dict[str, object]] = []
    for source, payload in zip(
        _SOURCE_NAMES,
        (_as_bytes(nasdaqlisted), _as_bytes(otherlisted)),
        strict=True,
    ):
        records.extend(_parse_directory(source, payload, observed_at))

    frame = pl.DataFrame(records, schema=_FRAME_SCHEMA)
    duplicates = (
        frame.group_by("symbol").len().filter(pl.col("len") > 1).get_column("symbol").to_list()
    )
    if duplicates:
        raise UniverseSnapshotError(
            f"symbol directories contain duplicate normalized symbols: {sorted(duplicates)}"
        )
    return frame.sort("symbol")


def save_universe_snapshot(
    nasdaqlisted: str | bytes,
    otherlisted: str | bytes,
    *,
    as_of: date,
    available_at: datetime,
    root: Path | None = None,
) -> UniverseSnapshot:
    """Atomically persist raw and normalized data under date/content identity.

    Saving the same bytes again is idempotent and preserves the original
    (earliest) observation time.  Different same-day contents get distinct
    digest directories, so a vendor revision never overwrites prior evidence.
    """
    if isinstance(as_of, datetime) or not isinstance(as_of, date):
        raise TypeError("as_of must be a date")
    observed_at = _aware_utc(available_at, name="available_at")
    raw_payloads = {
        "nasdaqlisted": _as_bytes(nasdaqlisted),
        "otherlisted": _as_bytes(otherlisted),
    }
    digest = _content_digest(raw_payloads)
    base = root or get_universe_snapshot_dir()
    partition = base / f"as_of={as_of.isoformat()}"
    target = partition / f"digest={digest}"

    with exclusive_file_lock(base / ".write.lock"):
        if target.exists():
            return _read_snapshot(target)

        frame = parse_symbol_directories(
            raw_payloads["nasdaqlisted"],
            raw_payloads["otherlisted"],
            available_at=observed_at,
        )
        creation_times = {
            source: _directory_creation_time(payload) for source, payload in raw_payloads.items()
        }
        for source, created_at in creation_times.items():
            if created_at > observed_at:
                raise UniverseSnapshotError(
                    f"{source} creation time {created_at.isoformat()} is after "
                    f"available_at {observed_at.isoformat()}"
                )
            if created_at.astimezone(_NEW_YORK).date() != as_of:
                raise UniverseSnapshotError(
                    f"{source} creation date does not match as_of={as_of.isoformat()}"
                )

        partition.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".digest={digest}.", dir=partition))
        try:
            for source, payload in raw_payloads.items():
                _write_bytes(temporary / f"{source}.txt", payload)
            normalized_path = temporary / "universe.parquet"
            frame.write_parquet(normalized_path)
            _sync_file(normalized_path)
            manifest = {
                "schema_version": _SCHEMA_VERSION,
                "as_of": as_of.isoformat(),
                "digest": digest,
                "available_at": _format_datetime(observed_at),
                "row_count": frame.height,
                "normalized_sha256": _file_sha256(normalized_path),
                "sources": {
                    source: {
                        "filename": f"{source}.txt",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "file_creation_time": _format_datetime(creation_times[source]),
                    }
                    for source, payload in raw_payloads.items()
                },
            }
            manifest_path = temporary / "manifest.json"
            _write_bytes(
                manifest_path,
                (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode(),
            )
            _sync_directory(temporary)
            os.replace(temporary, target)
            _sync_directory(partition)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    return _read_snapshot(target)


def read_universe_as_of(
    as_of: date,
    *,
    known_at: datetime,
    root: Path | None = None,
) -> UniverseSnapshot:
    """Read the newest snapshot no later than ``as_of`` and ``known_at``.

    ``known_at`` is mandatory: silently defaulting to the present would allow
    future directory revisions into historical decisions.
    """
    if isinstance(as_of, datetime) or not isinstance(as_of, date):
        raise TypeError("as_of must be a date")
    cutoff = _aware_utc(known_at, name="known_at")
    base = root or get_universe_snapshot_dir()
    candidates: list[tuple[date, datetime, str, Path]] = []
    if base.exists():
        for manifest_path in base.glob("as_of=*/digest=*/manifest.json"):
            snapshot_path = manifest_path.parent
            manifest = _read_manifest(manifest_path)
            snapshot_date = _manifest_date(manifest, snapshot_path)
            observed_at = _manifest_datetime(manifest, "available_at")
            if snapshot_date <= as_of and observed_at <= cutoff:
                candidates.append(
                    (
                        snapshot_date,
                        observed_at,
                        str(manifest.get("digest", "")),
                        snapshot_path,
                    )
                )
    if not candidates:
        raise FileNotFoundError(
            "no universe snapshot was available by "
            f"{cutoff.isoformat()} for as_of={as_of.isoformat()}"
        )
    return _read_snapshot(max(candidates)[-1])


def _parse_directory(
    source: str,
    payload: bytes,
    available_at: datetime,
) -> list[dict[str, object]]:
    text = _decode(payload)
    creation_time = _directory_creation_time(payload)
    if creation_time > available_at:
        raise UniverseSnapshotError(f"{source} creation time is after its observation time")
    reader = csv.DictReader(io.StringIO(text), delimiter="|")
    fieldnames = set(reader.fieldnames or ())
    required = _NASDAQ_REQUIRED_COLUMNS if source == "nasdaqlisted" else _OTHER_REQUIRED_COLUMNS
    if missing := required - fieldnames:
        raise UniverseSnapshotError(f"{source} directory missing columns: {sorted(missing)}")

    records: list[dict[str, object]] = []
    for row in reader:
        first = next(iter(row.values()), "") or ""
        if first.startswith("File Creation Time:"):
            continue
        if None in row or any(value is None for value in row.values()):
            raise UniverseSnapshotError(f"{source} directory contains a malformed row")
        raw_symbol = (
            row["Symbol"] if source == "nasdaqlisted" else row["NASDAQ Symbol"] or row["ACT Symbol"]
        )
        symbol = raw_symbol.strip().replace(".", "-")
        security_name = row["Security Name"].strip()
        if not symbol or not security_name:
            raise UniverseSnapshotError(
                f"{source} directory contains an empty symbol or security name"
            )
        exchange = (
            "NASDAQ"
            if source == "nasdaqlisted"
            else _OTHER_EXCHANGES.get(row["Exchange"].strip(), row["Exchange"].strip())
        )
        records.append(
            {
                "symbol": symbol,
                "security_name": security_name,
                "exchange": exchange,
                "security_type": _security_type(row, security_name),
                "flags": _flags(source, row),
                "source": source,
                "file_creation_time": creation_time,
                "available_at": available_at,
            }
        )
    if not records:
        raise UniverseSnapshotError(f"{source} directory contains no securities")
    return records


def _security_type(row: dict[str, str], security_name: str) -> str:
    name = f" {security_name.lower()} "
    if row.get("ETF") == "Y":
        return "etf"
    if row.get("NextShares") == "Y":
        return "nextshares"
    if " warrant" in name:
        return "warrant"
    if " right" in name:
        return "right"
    if any(marker in name for marker in (" - unit", " unit ", " units ")):
        return "unit"
    if any(marker in name for marker in (" preferred ", " preference share")):
        return "preferred"
    if any(
        marker in name
        for marker in (
            " american depositary share",
            " american depositary receipt",
            " american depository share",
        )
    ):
        return "adr"
    if any(
        marker in name
        for marker in (
            " common stock",
            " common shares",
            " ordinary share",
        )
    ):
        return "common_stock"
    return "other"


def _flags(source: str, row: dict[str, str]) -> list[str]:
    flags: list[str] = []
    if row.get("Test Issue") == "Y":
        flags.append("test_issue")
    if row.get("ETF") == "Y":
        flags.append("etf")
    if row.get("NextShares") == "Y":
        flags.append("nextshares")
    if source == "nasdaqlisted":
        if value := row.get("Market Category", "").strip():
            flags.append(f"market_category:{value}")
        if value := row.get("Financial Status", "").strip():
            flags.append(f"financial_status:{value}")
    return sorted(flags)


def _directory_creation_time(payload: bytes) -> datetime:
    for line in reversed(_decode(payload).splitlines()):
        if match := _FILE_CREATION_RE.match(line.strip()):
            naive = datetime.strptime(match.group("stamp"), "%m%d%Y%H:%M")
            return naive.replace(tzinfo=_NEW_YORK).astimezone(UTC)
    raise UniverseSnapshotError("symbol directory has no valid File Creation Time footer")


def _content_digest(payloads: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for source in _SOURCE_NAMES:
        payload = payloads[source]
        digest.update(source.encode())
        digest.update(b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _read_snapshot(snapshot_path: Path) -> UniverseSnapshot:
    manifest = _read_manifest(snapshot_path / "manifest.json")
    snapshot_date = _manifest_date(manifest, snapshot_path)
    digest = str(manifest.get("digest", ""))
    if snapshot_path.name != f"digest={digest}":
        raise SnapshotIntegrityError("snapshot path does not match manifest digest")

    raw_payloads: dict[str, bytes] = {}
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(_SOURCE_NAMES):
        raise SnapshotIntegrityError("snapshot manifest has invalid sources")
    for source in _SOURCE_NAMES:
        metadata = sources[source]
        if not isinstance(metadata, dict):
            raise SnapshotIntegrityError(f"invalid manifest metadata for {source}")
        expected_name = f"{source}.txt"
        if metadata.get("filename") != expected_name:
            raise SnapshotIntegrityError(f"unexpected raw filename for {source}")
        try:
            payload = (snapshot_path / expected_name).read_bytes()
        except OSError as exc:
            raise SnapshotIntegrityError(f"could not read raw {source} snapshot") from exc
        if hashlib.sha256(payload).hexdigest() != metadata.get("sha256"):
            raise SnapshotIntegrityError(f"raw {source} snapshot hash mismatch")
        raw_payloads[source] = payload
    if _content_digest(raw_payloads) != digest:
        raise SnapshotIntegrityError("combined snapshot digest mismatch")

    normalized_path = snapshot_path / "universe.parquet"
    if _file_sha256(normalized_path) != manifest.get("normalized_sha256"):
        raise SnapshotIntegrityError("normalized snapshot hash mismatch")
    try:
        frame = pl.read_parquet(normalized_path)
    except Exception as exc:
        raise SnapshotIntegrityError("could not read normalized snapshot") from exc
    if frame.schema != _FRAME_SCHEMA:
        raise SnapshotIntegrityError("normalized snapshot schema mismatch")
    if frame.height != manifest.get("row_count"):
        raise SnapshotIntegrityError("normalized snapshot row count mismatch")

    observed_at = _manifest_datetime(manifest, "available_at")
    if frame.get_column("available_at").unique().to_list() != [observed_at]:
        raise SnapshotIntegrityError("row availability does not match manifest")
    return UniverseSnapshot(
        as_of=snapshot_date,
        digest=digest,
        available_at=observed_at,
        path=snapshot_path,
        frame=frame,
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotIntegrityError(f"invalid snapshot manifest: {path}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != _SCHEMA_VERSION:
        raise SnapshotIntegrityError(f"unsupported snapshot manifest: {path}")
    return manifest


def _manifest_date(manifest: dict[str, Any], snapshot_path: Path) -> date:
    try:
        value = date.fromisoformat(str(manifest["as_of"]))
    except (KeyError, ValueError) as exc:
        raise SnapshotIntegrityError("invalid manifest as_of") from exc
    if snapshot_path.parent.name != f"as_of={value.isoformat()}":
        raise SnapshotIntegrityError("snapshot path does not match manifest as_of")
    return value


def _manifest_datetime(manifest: dict[str, Any], key: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(manifest[key]).replace("Z", "+00:00"))
    except (KeyError, ValueError) as exc:
        raise SnapshotIntegrityError(f"invalid manifest {key}") from exc
    try:
        return _aware_utc(parsed, name=key)
    except (TypeError, ValueError) as exc:
        raise SnapshotIntegrityError(f"invalid manifest {key}") from exc


def _aware_utc(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _as_bytes(contents: str | bytes) -> bytes:
    if isinstance(contents, bytes):
        return contents
    if isinstance(contents, str):
        return contents.encode()
    raise TypeError("directory contents must be str or bytes")


def _decode(payload: bytes) -> str:
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UniverseSnapshotError("symbol directory must be valid UTF-8") from exc


def _format_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise SnapshotIntegrityError(f"could not hash snapshot file: {path}") from exc


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _sync_file(path: Path) -> None:
    with path.open("rb") as saved:
        os.fsync(saved.fileno())


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
