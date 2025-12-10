# faers_signals/emergence.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable

import duckdb
import numpy as np
import pandas as pd

from .config import WAREHOUSE_DB_PATH


@dataclass
class EmergenceConfig:
    """
    Configuration for emergence score computation.
    """
    min_points: int = 4          # minimum number of quarters for a PT
    min_n11_latest: int = 5      # minimum joint counts in latest quarter
    roles: Optional[Iterable[str]] = ("PS", "SS")  # for future use if needed


def get_connection(db_path: Optional[str] = None) -> duckdb.DuckDBPyConnection:
    """
    Open a DuckDB connection to the FAERS warehouse.
    """
    path = db_path or str(WAREHOUSE_DB_PATH)
    return duckdb.connect(path)


def compute_emergence_scores_from_df(
    df: pd.DataFrame,
    cfg: Optional[EmergenceConfig] = None,
) -> pd.DataFrame:
    """
    Compute emergence and trend (slope-based) scores for each PT given a
    signals_quarterly-like DataFrame already filtered to a single drug
    (one drugname_norm).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least the columns:
        - pt
        - quarter_idx
        - n11
        - ror
        - ror_ci_low
    cfg : EmergenceConfig, optional

    Returns
    -------
    pd.DataFrame with columns:
        pt
        emergence_z
        slope_log_ror
        latest_ror
        latest_ror_ci_low
        latest_n11
        n_points
    """
    if cfg is None:
        cfg = EmergenceConfig()

    records = []

    # Group by PT and analyze its time series
    for pt, g in df.groupby("pt"):
        g = g.sort_values("quarter_idx")

        # Require enough time points
        if len(g) < cfg.min_points:
            continue

        # Latest quarter row
        latest = g.iloc[-1]
        if latest["n11"] < cfg.min_n11_latest:
            continue

        # Baseline: all but last
        baseline = g.iloc[:-1]
        # If baseline has essentially no support, skip
        if (baseline["n11"] < 1).all():
            continue

        # Log-ROR time series, guarding against zeros/infs
        log_ror = np.log(g["ror"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        # If we have too many NaNs, skip this PT
        if log_ror.isna().sum() > len(log_ror) - 2:
            continue

        # --- Emergence part (same logic as before) ---
        log_ror_baseline = log_ror.iloc[:-1].dropna()
        log_ror_latest = log_ror.iloc[-1]

        # Need at least 2 baseline points and non-NaN latest
        if len(log_ror_baseline) < 2 or pd.isna(log_ror_latest):
            continue

        mu = log_ror_baseline.mean()
        sigma = log_ror_baseline.std(ddof=1)
        if sigma == 0 or np.isnan(sigma):
            continue

        emergence_z = (log_ror_latest - mu) / sigma

        # --- NEW: slope-based temporal trend on log-ROR ---
        # Use all non-NaN points for the slope.
        mask = ~log_ror.isna()
        if mask.sum() < 2:
            continue

        t = g.loc[mask, "quarter_idx"].to_numpy(dtype=float)
        y = log_ror[mask].to_numpy(dtype=float)

        # Normalize time to [0, 1] to make slopes comparable across different ranges
        t_min, t_max = t.min(), t.max()
        if t_max == t_min:
            continue
        t_norm = (t - t_min) / (t_max - t_min)

        # Simple linear regression: log_ror ~ a + b * t_norm
        # slope_log_ror = b
        slope_log_ror = float(np.polyfit(t_norm, y, 1)[0])

        records.append(
            {
                "pt": pt,
                "emergence_z": float(emergence_z),
                "slope_log_ror": slope_log_ror,
                "latest_ror": float(latest["ror"]),
                "latest_ror_ci_low": float(latest["ror_ci_low"]),
                "latest_n11": int(latest["n11"]),
                "n_points": int(len(g)),
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "pt",
                "emergence_z",
                "slope_log_ror",
                "latest_ror",
                "latest_ror_ci_low",
                "latest_n11",
                "n_points",
            ]
        )

    out = pd.DataFrame(records)
    out = out.sort_values("emergence_z", ascending=False).reset_index(drop=True)
    return out



def compute_emergence_scores_for_drug(
    drugname_norm: str,
    cfg: Optional[EmergenceConfig] = None,
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    """
    Convenience function: pull signals_quarterly for a given normalized drug
    and compute emergence scores for all PTs.

    Parameters
    ----------
    drugname_norm : str
        Normalized drug name, e.g. "ATORVASTATIN".
        (You can safely pass "atorvastatin"; it'll be uppercased and trimmed.)
    cfg : EmergenceConfig, optional
    con : duckdb.DuckDBPyConnection, optional

    Returns
    -------
    pd.DataFrame
        Same as compute_emergence_scores_from_df.
    """
    if cfg is None:
        cfg = EmergenceConfig()

    close_con = False
    if con is None:
        con = get_connection()
        close_con = True

    try:
        dn = drugname_norm.upper().strip()

        signals_q = con.execute(
            """
            SELECT *
            FROM signals_quarterly
            WHERE drugname_norm = ?
            ORDER BY pt, year, quarter_idx
            """,
            [dn],
        ).fetchdf()

        if signals_q.empty:
            return pd.DataFrame(
                columns=[
                    "pt",
                    "emergence_z",
                    "latest_ror",
                    "latest_ror_ci_low",
                    "latest_n11",
                    "n_points",
                ]
            )

        return compute_emergence_scores_from_df(signals_q, cfg=cfg)

    finally:
        if close_con:
            con.close()




def _get_drugs_with_min_reports(
    con: duckdb.DuckDBPyConnection,
    min_total_drug_reports: int = 500,
) -> pd.DataFrame:
    """
    Helper: list normalized drugs with at least min_total_drug_reports
    in drug_dedup_norm.
    """
    df = con.execute(
        """
        SELECT
            drugname_norm,
            COUNT(DISTINCT primaryid) AS n_reports
        FROM drug_dedup_norm
        GROUP BY drugname_norm
        HAVING COUNT(DISTINCT primaryid) >= ?
        ORDER BY n_reports DESC
        """,
        [min_total_drug_reports],
    ).fetchdf()
    return df


def compute_global_emergence_scores(
    cfg: Optional[EmergenceConfig] = None,
    min_total_drug_reports: int = 500,
    max_drugs: Optional[int] = None,
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    """
    Compute emergence + trend scores across ALL eligible drugs.

    For each drugname_norm with at least `min_total_drug_reports` appearances,
    we:
      1. Pull its time series from signals_quarterly,
      2. Compute emergence_z and slope_log_ror per PT,
      3. Attach drugname_norm to each row,
      4. Compute a composite signal_score.

    Composite score (heuristic):
        signal_score = max(slope_log_ror, 0) * emergence_z * log(latest_ror + 1)

    Parameters
    ----------
    cfg : EmergenceConfig, optional
        Configuration for emergence scoring.
    min_total_drug_reports : int
        Only consider drugs with at least this many reports in drug_dedup_norm.
    max_drugs : int, optional
        If provided, only process the top N drugs by report count (for speed).
    con : duckdb connection, optional

    Returns
    -------
    pd.DataFrame with columns:
        drugname_norm
        pt
        signal_score
        emergence_z
        slope_log_ror
        latest_ror
        latest_ror_ci_low
        latest_n11
        n_points
    """
    if cfg is None:
        cfg = EmergenceConfig()

    close_con = False
    if con is None:
        con = get_connection()
        close_con = True

    try:
        # 1) Which drugs are we going to process?
        drugs_df = _get_drugs_with_min_reports(
            con, min_total_drug_reports=min_total_drug_reports
        )
        if max_drugs is not None and len(drugs_df) > max_drugs:
            drugs_df = drugs_df.iloc[:max_drugs].copy()

        print(
            f"[info] Computing global emergence for {len(drugs_df)} drugs "
            f"(min_total_drug_reports={min_total_drug_reports})"
        )

        all_records = []

        for i, row in drugs_df.iterrows():
            dn = row["drugname_norm"]
            n_reports = int(row["n_reports"])
            print(f"[info] [{i+1}/{len(drugs_df)}] {dn} (n_reports={n_reports})")

            # Pull time series for this drug from signals_quarterly
            signals_q = con.execute(
                """
                SELECT *
                FROM signals_quarterly
                WHERE drugname_norm = ?
                ORDER BY pt, year, quarter_idx
                """,
                [dn],
            ).fetchdf()

            if signals_q.empty:
                continue

            # Compute emergence + slope for this drug
            em_df = compute_emergence_scores_from_df(signals_q, cfg=cfg)
            if em_df.empty:
                continue

            em_df.insert(0, "drugname_norm", dn)
            all_records.append(em_df)

        if not all_records:
            return pd.DataFrame(
                columns=[
                    "drugname_norm",
                    "pt",
                    "signal_score",
                    "emergence_z",
                    "slope_log_ror",
                    "latest_ror",
                    "latest_ror_ci_low",
                    "latest_n11",
                    "n_points",
                ]
            )

        out = pd.concat(all_records, ignore_index=True)

        # Composite signal score:
        #   - require upward trend: max(slope_log_ror, 0)
        #   - require surprise: emergence_z
        #   - require strong current signal: log(latest_ror + 1)
        slope_pos = np.maximum(out["slope_log_ror"], 0.0)
        out["signal_score"] = slope_pos * out["emergence_z"] * np.log(
            out["latest_ror"] + 1.0
        )

        # Reorder & sort
        cols_order = [
            "drugname_norm",
            "pt",
            "signal_score",
            "emergence_z",
            "slope_log_ror",
            "latest_ror",
            "latest_ror_ci_low",
            "latest_n11",
            "n_points",
        ]
        out = out[cols_order].sort_values("signal_score", ascending=False).reset_index(
            drop=True
        )

        return out

    finally:
        if close_con:
            con.close()

