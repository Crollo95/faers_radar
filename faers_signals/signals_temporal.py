# faers_signals/signals_temporal.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import duckdb

from .config import WAREHOUSE_DB_PATH


@dataclass
class TemporalSignalConfig:
    roles: Optional[Iterable[str]] = ("PS", "SS")  # which roles to include
    min_n11: int = 3
    continuity_correction: float = 0.5
    ci_z: float = 1.96


def _build_role_filter(roles: Optional[Iterable[str]]) -> str:
    if not roles:
        return ""
    roles_list = list(roles)
    quoted = ", ".join(f"'{r}'" for r in roles_list)
    return f" AND role_cod IN ({quoted}) "


def _get_year_bounds(con, year_min: int, year_max: Optional[int]) -> tuple[int, int]:
    # If year_max is None, infer from demo_dedup
    if year_max is None:
        y_min_db, y_max_db = con.execute("""
            SELECT MIN(year), MAX(year) FROM demo_dedup
        """).fetchone()
        if y_min_db is None:
            raise ValueError("demo_dedup is empty or has no year data.")
        if year_min is None:
            year_min = int(y_min_db)
        if year_max is None:
            year_max = int(y_max_db)
    return year_min, year_max


def build_counts_tables(
    cfg: TemporalSignalConfig,
    year_min: int = 2019,
    year_max: Optional[int] = None,
):
    """
    Build global counts tables:

    - demo_counts(year, quarter, total_reports)
    - drug_counts(year, quarter, drugname_norm, total_drug_reports)
    - event_counts(year, quarter, pt, total_event_reports)
    """
    print(f"[info] Connecting to DuckDB at {WAREHOUSE_DB_PATH}")
    con = duckdb.connect(str(WAREHOUSE_DB_PATH))

    year_min, year_max = _get_year_bounds(con, year_min, year_max)
    print(f"[info] Building counts for years {year_min}–{year_max}")

    role_clause = _build_role_filter(cfg.roles)

    print("[info] Creating demo_counts ...")
    con.execute("DROP TABLE IF EXISTS demo_counts")
    con.execute(f"""
        CREATE TABLE demo_counts AS
        SELECT
            year,
            quarter,
            COUNT(DISTINCT primaryid) AS total_reports
        FROM demo_dedup
        WHERE year BETWEEN {year_min} AND {year_max}
        GROUP BY year, quarter
    """)

    print("[info] Creating drug_counts ...")
    con.execute("DROP TABLE IF EXISTS drug_counts")
    con.execute(f"""
        CREATE TABLE drug_counts AS
        SELECT
            year,
            quarter,
            drugname_norm,
            COUNT(DISTINCT primaryid) AS total_drug_reports
        FROM drug_dedup_norm
        WHERE year BETWEEN {year_min} AND {year_max}
        {role_clause}
        GROUP BY year, quarter, drugname_norm
    """)

    print("[info] Creating event_counts ...")
    con.execute("DROP TABLE IF EXISTS event_counts")
    con.execute(f"""
        CREATE TABLE event_counts AS
        SELECT
            year,
            quarter,
            pt,
            COUNT(DISTINCT primaryid) AS total_event_reports
        FROM reac_dedup
        WHERE year BETWEEN {year_min} AND {year_max}
        GROUP BY year, quarter, pt
    """)

    con.close()
    print("[info] demo_counts, drug_counts, event_counts created.")


def build_quarterly_signals_incremental(
    cfg: Optional[TemporalSignalConfig] = None,
    year_min: int = 2019,
    year_max: Optional[int] = None,
):
    """
    Incremental builder for signals_quarterly.

    Strategy:
    1. Build global counts tables once: demo_counts, drug_counts, event_counts.
    2. Create empty signals_quarterly table.
    3. For each year in [year_min, year_max]:
        - Build a temporary joint_y table with N11 for that year only.
        - Join with counts tables and compute ROR/PRR for that year.
        - INSERT results into signals_quarterly.

    This processes ALL data (no subsampling), but limits each heavy join to one year.
    """
    if cfg is None:
        cfg = TemporalSignalConfig()

    # Step 1: build counts
    build_counts_tables(cfg, year_min=year_min, year_max=year_max)

    print(f"[info] Connecting to DuckDB at {WAREHOUSE_DB_PATH}")
    con = duckdb.connect(str(WAREHOUSE_DB_PATH))

    # Get year bounds
    year_min, year_max = _get_year_bounds(con, year_min, year_max)
    print(f"[info] Building signals_quarterly incrementally for {year_min}–{year_max}")

    # Drop existing signals_quarterly
    print("[info] Dropping existing signals_quarterly (if any)")
    con.execute("DROP TABLE IF EXISTS signals_quarterly")

    # Pre-create signals_quarterly with schema but no rows
    print("[info] Creating empty signals_quarterly table with correct schema")
    cc = cfg.continuity_correction
    z = cfg.ci_z

    # Create an empty table by running a query with WHERE FALSE
    con.execute(f"""
        CREATE TABLE signals_quarterly AS
        WITH joint_y AS (
            SELECT
                d.year,
                d.quarter,
                d.drugname_norm,
                r.pt,
                COUNT(DISTINCT d.primaryid) AS n11
            FROM drug_dedup_norm d
            JOIN reac_dedup r
              ON d.primaryid = r.primaryid
            WHERE 1=0
            GROUP BY d.year, d.quarter, d.drugname_norm, r.pt
        ),
        all_counts AS (
            SELECT
                j.year,
                j.quarter,
                j.drugname_norm,
                j.pt,
                j.n11,
                dc.total_reports,
                drc.total_drug_reports,
                ec.total_event_reports
            FROM joint_y j
            JOIN demo_counts dc
              ON j.year = dc.year AND j.quarter = dc.quarter
            JOIN drug_counts drc
              ON j.year = drc.year
             AND j.quarter = drc.quarter
             AND j.drugname_norm = drc.drugname_norm
            JOIN event_counts ec
              ON j.year = ec.year
             AND j.quarter = ec.quarter
             AND j.pt = ec.pt
        ),
        counts_with_n AS (
            SELECT
                *,
                (total_drug_reports - n11) AS n10,
                (total_event_reports - n11) AS n01,
                (total_reports - n11 - (total_drug_reports - n11) - (total_event_reports - n11)) AS n00
            FROM all_counts
        ),
        filtered AS (
            SELECT
                *,
                CASE quarter
                    WHEN 'Q1' THEN 1
                    WHEN 'Q2' THEN 2
                    WHEN 'Q3' THEN 3
                    WHEN 'Q4' THEN 4
                    ELSE NULL
                END AS qnum
            FROM counts_with_n
            WHERE n11 >= {cfg.min_n11}
        ),
        metrics AS (
            SELECT
                drugname_norm,
                pt,
                year,
                quarter,
                (year * 4 + qnum) AS quarter_idx,
                total_reports,
                total_drug_reports,
                total_event_reports,
                n11,
                n10,
                n01,
                n00,
                (n11 + {cc}) AS n11c,
                (n10 + {cc}) AS n10c,
                (n01 + {cc}) AS n01c,
                (n00 + {cc}) AS n00c
            FROM filtered
        ),
        final AS (
            SELECT
                drugname_norm,
                pt,
                year,
                quarter,
                quarter_idx,
                total_reports,
                total_drug_reports,
                total_event_reports,
                n11,
                n10,
                n01,
                n00,
                (n11c * n00c) / (n10c * n01c) AS ror,
                EXP(
                    LN((n11c * n00c) / (n10c * n01c))
                    - {z} * SQRT(1.0 / n11c + 1.0 / n10c + 1.0 / n01c + 1.0 / n00c)
                ) AS ror_ci_low,
                EXP(
                    LN((n11c * n00c) / (n10c * n01c))
                    + {z} * SQRT(1.0 / n11c + 1.0 / n10c + 1.0 / n01c + 1.0 / n00c)
                ) AS ror_ci_high,
                (n11c / (n11c + n10c)) / (n01c / (n01c + n00c)) AS prr
            FROM metrics
        )
        SELECT * FROM final WHERE 1=0
    """)

    role_clause = _build_role_filter(cfg.roles)

    # Step 3: process year by year
    for year in range(year_min, year_max + 1):
        print(f"[info] Processing year {year} ...")

        # Drop temp table if exists
        con.execute("DROP TABLE IF EXISTS joint_y")

        # Build joint_y for this specific year only
        con.execute(f"""
            CREATE TABLE joint_y AS
            SELECT
                d.year,
                d.quarter,
                d.drugname_norm,
                r.pt,
                COUNT(DISTINCT d.primaryid) AS n11
            FROM drug_dedup_norm d
            JOIN reac_dedup r
              ON d.primaryid = r.primaryid
            WHERE d.year = {year}
            {role_clause}
            GROUP BY d.year, d.quarter, d.drugname_norm, r.pt
        """)

        # Insert metrics for this year into signals_quarterly
        con.execute(f"""
            INSERT INTO signals_quarterly
            WITH all_counts AS (
                SELECT
                    j.year,
                    j.quarter,
                    j.drugname_norm,
                    j.pt,
                    j.n11,
                    dc.total_reports,
                    drc.total_drug_reports,
                    ec.total_event_reports
                FROM joint_y j
                JOIN demo_counts dc
                  ON j.year = dc.year AND j.quarter = dc.quarter
                JOIN drug_counts drc
                  ON j.year = drc.year
                 AND j.quarter = drc.quarter
                 AND j.drugname_norm = drc.drugname_norm
                JOIN event_counts ec
                  ON j.year = ec.year
                 AND j.quarter = ec.quarter
                 AND j.pt = ec.pt
            ),
            counts_with_n AS (
                SELECT
                    *,
                    (total_drug_reports - n11) AS n10,
                    (total_event_reports - n11) AS n01,
                    (total_reports - n11 - (total_drug_reports - n11) - (total_event_reports - n11)) AS n00
                FROM all_counts
            ),
            filtered AS (
                SELECT
                    *,
                    CASE quarter
                        WHEN 'Q1' THEN 1
                        WHEN 'Q2' THEN 2
                        WHEN 'Q3' THEN 3
                        WHEN 'Q4' THEN 4
                        ELSE NULL
                    END AS qnum
                FROM counts_with_n
                WHERE n11 >= {cfg.min_n11}
            ),
            metrics AS (
                SELECT
                    drugname_norm,
                    pt,
                    year,
                    quarter,
                    (year * 4 + qnum) AS quarter_idx,
                    total_reports,
                    total_drug_reports,
                    total_event_reports,
                    n11,
                    n10,
                    n01,
                    n00,
                    (n11 + {cc}) AS n11c,
                    (n10 + {cc}) AS n10c,
                    (n01 + {cc}) AS n01c,
                    (n00 + {cc}) AS n00c
                FROM filtered
            ),
            final AS (
                SELECT
                    drugname_norm,
                    pt,
                    year,
                    quarter,
                    quarter_idx,
                    total_reports,
                    total_drug_reports,
                    total_event_reports,
                    n11,
                    n10,
                    n01,
                    n00,
                    (n11c * n00c) / (n10c * n01c) AS ror,
                    EXP(
                        LN((n11c * n00c) / (n10c * n01c))
                        - {z} * SQRT(1.0 / n11c + 1.0 / n10c + 1.0 / n01c + 1.0 / n00c)
                    ) AS ror_ci_low,
                    EXP(
                        LN((n11c * n00c) / (n10c * n01c))
                        + {z} * SQRT(1.0 / n11c + 1.0 / n10c + 1.0 / n01c + 1.0 / n00c)
                    ) AS ror_ci_high,
                    (n11c / (n11c + n10c)) / (n01c / (n01c + n00c)) AS prr
                FROM metrics
            )
            SELECT * FROM final
        """)

        # Quick per-year summary
        n_year = con.execute(f"""
            SELECT COUNT(*) FROM signals_quarterly WHERE year = {year}
        """).fetchone()[0]
        print(f"[info]  -> year {year}: inserted {n_year:,} rows")

    # Final summary
    total_rows = con.execute("SELECT COUNT(*) FROM signals_quarterly").fetchone()[0]
    print(f"[info] signals_quarterly built with total {total_rows:,} rows")

    con.close()
    print("[info] Done.")


if __name__ == "__main__":
    cfg = TemporalSignalConfig()
    build_quarterly_signals_incremental(cfg=cfg, year_min=2019, year_max=None)

