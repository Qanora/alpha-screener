from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime

import polars as pl
import pytest

from alphascreener.data.universe import (
    SnapshotIntegrityError,
    UniverseSnapshotError,
    parse_symbol_directories,
    read_universe_as_of,
    save_universe_snapshot,
)


def _nasdaq(*rows: str, created: str = "0723202607:00") -> str:
    return "\n".join(
        [
            "Symbol|Security Name|Market Category|Test Issue|Financial Status|"
            "Round Lot Size|ETF|NextShares",
            *rows,
            f"File Creation Time: {created}|||||||",
        ]
    )


def _other(*rows: str, created: str = "0723202607:00") -> str:
    return "\n".join(
        [
            "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|"
            "Test Issue|NASDAQ Symbol",
            *rows,
            f"File Creation Time: {created}||||||",
        ]
    )


@pytest.fixture
def directories() -> tuple[str, str]:
    return (
        _nasdaq(
            "GOOD|Good Corporation - Common Stock|Q|N|N|100|N|N",
            "FUND|Example Fund ETF|G|N|N|100|Y|N",
            "ADR|Issuer - American Depositary Shares|S|N|N|100|N|N",
        ),
        _other(
            "BRK.B|Berkshire Hathaway Inc. Common Stock|N|BRK.B|N|100|N|BRK.B",
            "TEST|Test Security Common Stock|P|TEST|N|100|Y|TEST",
        ),
    )


def test_parse_symbol_directories_preserves_records_and_point_in_time_metadata(
    directories,
) -> None:
    observed_at = datetime(2026, 7, 23, 11, 5, tzinfo=UTC)

    frame = parse_symbol_directories(*directories, available_at=observed_at)

    assert frame.columns == [
        "symbol",
        "security_name",
        "exchange",
        "security_type",
        "flags",
        "source",
        "file_creation_time",
        "available_at",
    ]
    assert frame["symbol"].to_list() == ["ADR", "BRK-B", "FUND", "GOOD", "TEST"]
    assert frame.filter(pl.col("symbol") == "ADR").item(0, "security_type") == "adr"
    assert frame.filter(pl.col("symbol") == "FUND")["flags"].to_list()[0] == [
        "etf",
        "financial_status:N",
        "market_category:G",
    ]
    assert frame.filter(pl.col("symbol") == "BRK-B").item(0, "exchange") == "NYSE"
    assert frame.filter(pl.col("symbol") == "TEST")["flags"].to_list()[0] == ["test_issue"]
    assert frame["file_creation_time"].unique().to_list() == [
        datetime(2026, 7, 23, 11, 0, tzinfo=UTC)
    ]
    assert frame["available_at"].unique().to_list() == [observed_at]


def test_save_is_idempotent_and_preserves_raw_bytes(tmp_path, directories) -> None:
    first_seen = datetime(2026, 7, 23, 11, 5, tzinfo=UTC)

    first = save_universe_snapshot(
        *directories,
        as_of=date(2026, 7, 23),
        available_at=first_seen,
        root=tmp_path,
    )
    repeated = save_universe_snapshot(
        *directories,
        as_of=date(2026, 7, 23),
        available_at=datetime(2026, 7, 23, 12, 0, tzinfo=UTC),
        root=tmp_path,
    )

    assert repeated.path == first.path
    assert repeated.available_at == first_seen
    assert (first.path / "nasdaqlisted.txt").read_text() == directories[0]
    assert (first.path / "otherlisted.txt").read_text() == directories[1]
    assert len(list(tmp_path.glob("as_of=*/digest=*"))) == 1


def test_same_day_revision_creates_a_second_immutable_digest(tmp_path, directories) -> None:
    first = save_universe_snapshot(
        *directories,
        as_of=date(2026, 7, 23),
        available_at=datetime(2026, 7, 23, 11, 5, tzinfo=UTC),
        root=tmp_path,
    )
    revised_nasdaq = directories[0].replace("GOOD|Good Corporation", "NEW|New Corporation")
    second = save_universe_snapshot(
        revised_nasdaq,
        directories[1],
        as_of=date(2026, 7, 23),
        available_at=datetime(2026, 7, 23, 12, 0, tzinfo=UTC),
        root=tmp_path,
    )

    assert second.path != first.path
    assert first.path.exists()
    assert second.path.exists()
    assert "GOOD" in first.symbols
    assert "NEW" in second.symbols
    assert len(list(tmp_path.glob("as_of=2026-07-23/digest=*"))) == 2


def test_read_as_of_uses_only_snapshots_known_by_cutoff(tmp_path) -> None:
    day_one_nasdaq = _nasdaq(
        "OLD|Old Corporation Common Stock|Q|N|N|100|N|N",
        created="0722202607:00",
    )
    day_one_other = _other(
        "AAA|A Corporation Common Stock|N|AAA|N|100|N|AAA",
        created="0722202607:00",
    )
    day_two_nasdaq = _nasdaq("NEW|New Corporation Common Stock|Q|N|N|100|N|N")
    day_two_other = _other("AAA|A Corporation Common Stock|N|AAA|N|100|N|AAA")
    save_universe_snapshot(
        day_one_nasdaq,
        day_one_other,
        as_of=date(2026, 7, 22),
        available_at=datetime(2026, 7, 22, 11, 5, tzinfo=UTC),
        root=tmp_path,
    )
    save_universe_snapshot(
        day_two_nasdaq,
        day_two_other,
        as_of=date(2026, 7, 23),
        available_at=datetime(2026, 7, 23, 12, 0, tzinfo=UTC),
        root=tmp_path,
    )

    before_day_two_arrived = read_universe_as_of(
        date(2026, 7, 23),
        known_at=datetime(2026, 7, 23, 11, 30, tzinfo=UTC),
        root=tmp_path,
    )
    after_day_two_arrived = read_universe_as_of(
        date(2026, 7, 23),
        known_at=datetime(2026, 7, 23, 12, 1, tzinfo=UTC),
        root=tmp_path,
    )

    assert before_day_two_arrived.as_of == date(2026, 7, 22)
    assert "OLD" in before_day_two_arrived.symbols
    assert after_day_two_arrived.as_of == date(2026, 7, 23)
    assert "NEW" in after_day_two_arrived.symbols


def test_save_rejects_time_travel_and_naive_timestamps(tmp_path, directories) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        save_universe_snapshot(
            *directories,
            as_of=date(2026, 7, 23),
            available_at=datetime(2026, 7, 23, 11, 5),
            root=tmp_path,
        )
    with pytest.raises(UniverseSnapshotError, match="after its observation time"):
        save_universe_snapshot(
            *directories,
            as_of=date(2026, 7, 23),
            available_at=datetime(2026, 7, 23, 10, 59, tzinfo=UTC),
            root=tmp_path,
        )
    with pytest.raises(UniverseSnapshotError, match="does not match as_of"):
        save_universe_snapshot(
            *directories,
            as_of=date(2026, 7, 22),
            available_at=datetime(2026, 7, 23, 11, 5, tzinfo=UTC),
            root=tmp_path,
        )


def test_concurrent_identical_writes_commit_one_complete_snapshot(tmp_path, directories) -> None:
    def save():
        return save_universe_snapshot(
            *directories,
            as_of=date(2026, 7, 23),
            available_at=datetime(2026, 7, 23, 11, 5, tzinfo=UTC),
            root=tmp_path,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        snapshots = list(executor.map(lambda _: save(), range(8)))

    assert len({snapshot.path for snapshot in snapshots}) == 1
    assert len(list(tmp_path.glob("as_of=*/digest=*"))) == 1
    assert snapshots[0].frame.height == 5


def test_read_detects_tampering(tmp_path, directories) -> None:
    snapshot = save_universe_snapshot(
        *directories,
        as_of=date(2026, 7, 23),
        available_at=datetime(2026, 7, 23, 11, 5, tzinfo=UTC),
        root=tmp_path,
    )
    (snapshot.path / "nasdaqlisted.txt").write_text("tampered")

    with pytest.raises(SnapshotIntegrityError, match="hash mismatch"):
        read_universe_as_of(
            date(2026, 7, 23),
            known_at=datetime(2026, 7, 23, 12, 0, tzinfo=UTC),
            root=tmp_path,
        )
