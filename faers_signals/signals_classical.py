# faers_signals/signals_classical.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

from .config import WAREHOUSE_DB_PATH


@dataclass
class DisproportionalityConfig:
    roles: Optional[Iterable[str]] = ("PS", "SS")  # roles in DRUG table to consider
    min_n11: int = 3               # minimum joint counts (drug & event)
    min_total_drug_reports: int = 30
    continuity_correction: float = 0.5
    ci_z: float = 1.96             # 95% CI


def get_connection(db_path: Optional[str] = None) -> duckdb.DuckDBPyConnection:
    """
    Open a DuckDB connection to the FAERS warehouse.
    """
    path = db_path or str(WAREHOUSE_DB_PATH)
    return duckdb.connect(path)


def _build_role_filter(roles: Optional[Iterable[str]]) -> str:
    if not roles:
        return ""
    roles_list = list(roles)
    quoted = ", ".join(f"'{r}'" for r in roles_list)
    return f" AND role_cod IN ({quoted}) "


def compute_signals_for_drug(
    drug_name: str,
    cfg: Optional[DisproportionalityConfig] = None,
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    """
    Compute disproportionality metrics (ROR, PRR, etc.) for a given drug
    across all reaction PTs, using *_dedup tables and normalized drug names.

    drug_name is normalized internally (upper, trimmed) and matched against
    drug_dedup_norm.drugname_norm.
    """
    if cfg is None:
        cfg = DisproportionalityConfig()

    close_con = False
    if con is None:
        con = get_connection()
        close_con = True

    try:
        # normalize query same way: upper + trim; we assume drugname_norm was constructed accordingly
        drug_norm = drug_name.upper().strip()
        role_clause = _build_role_filter(cfg.roles)

        # total number of deduplicated reports
        total_reports = con.execute("""
            SELECT COUNT(DISTINCT primaryid) AS n
            FROM demo_dedup
        """).fetchone()[0]

        # number of reports mentioning the drug (with chosen roles)
        total_drug_reports = con.execute(f"""
            SELECT COUNT(DISTINCT primaryid) AS n
            FROM drug_dedup_norm
            WHERE drugname_norm = ?
            {role_clause}
        """, [drug_norm]).fetchone()[0]

        if total_drug_reports < cfg.min_total_drug_reports:
            raise ValueError(
                f"Drug '{drug_name}' appears in only {total_drug_reports} reports "
                f"(min_total_drug_reports={cfg.min_total_drug_reports})."
            )

        # joint counts: N11 per PT (drug & event)
        joint_df = con.execute(f"""
            SELECT
                r.pt AS pt,
                COUNT(DISTINCT r.primaryid) AS n11
            FROM reac_dedup r
            JOIN drug_dedup_norm d
              ON r.primaryid = d.primaryid
            WHERE d.drugname_norm = ?
            {role_clause}
            GROUP BY r.pt
        """, [drug_norm]).fetchdf()

        # total event reports per PT (regardless of drug)
        event_df = con.execute("""
            SELECT
                pt,
                COUNT(DISTINCT primaryid) AS total_event_reports
            FROM reac_dedup
            GROUP BY pt
        """).fetchdf()

        df = joint_df.merge(event_df, on="pt", how="left")

        df["total_drug_reports"] = total_drug_reports
        df["total_reports"] = total_reports

        # N10, N01, N00
        df["n10"] = df["total_drug_reports"] - df["n11"]
        df["n01"] = df["total_event_reports"] - df["n11"]
        df["n00"] = df["total_reports"] - df["n11"] - df["n10"] - df["n01"]

        # filter low counts
        df = df[df["n11"] >= cfg.min_n11].copy()

        cc = cfg.continuity_correction
        for col in ["n11", "n10", "n01", "n00"]:
            df[col] = df[col].astype(float)

        n11c = df["n11"].clip(lower=0) + cc
        n10c = df["n10"].clip(lower=0) + cc
        n01c = df["n01"].clip(lower=0) + cc
        n00c = df["n00"].clip(lower=0) + cc

        # ROR
        ror = (n11c * n00c) / (n10c * n01c)
        df["ror"] = ror

        with np.errstate(divide="ignore", invalid="ignore"):
            log_ror = np.log(ror)
            se_log_ror = np.sqrt(1.0 / n11c + 1.0 / n10c + 1.0 / n01c + 1.0 / n00c)
            z = cfg.ci_z
            df["ror_ci_low"] = np.exp(log_ror - z * se_log_ror)
            df["ror_ci_high"] = np.exp(log_ror + z * se_log_ror)

            risk_drug = n11c / (n11c + n10c)
            risk_other = n01c / (n01c + n00c)
            df["prr"] = risk_drug / risk_other

        df.sort_values(["ror", "n11"], ascending=[False, False], inplace=True)
        return df.reset_index(drop=True)

    finally:
        if close_con:
            con.close()



def compute_signals_for_event(
    pt: str,
    cfg: Optional[DisproportionalityConfig] = None,
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    """
    Symmetric to compute_signals_for_drug, but conditions on a reaction PT
    and returns metrics per drug (using normalized drug names).
    """
    if cfg is None:
        cfg = DisproportionalityConfig()

    close_con = False
    if con is None:
        con = get_connection()
        close_con = True

    try:
        pt_norm = pt.upper().strip()
        role_clause = _build_role_filter(cfg.roles)

        total_reports = con.execute("""
            SELECT COUNT(DISTINCT primaryid) AS n
            FROM demo_dedup
        """).fetchone()[0]

        # Total reports with the event (all drugs)
        total_event_reports = con.execute("""
            SELECT COUNT(DISTINCT primaryid) AS n
            FROM reac_dedup
            WHERE UPPER(TRIM(pt)) = ?
        """, [pt_norm]).fetchone()[0]

        # Joint counts: N11 per normalized drug (drug & event)
        joint_df = con.execute(f"""
            SELECT
                d.drugname_norm AS drugname_norm,
                COUNT(DISTINCT d.primaryid) AS n11
            FROM drug_dedup_norm d
            JOIN reac_dedup r
              ON d.primaryid = r.primaryid
            WHERE UPPER(TRIM(r.pt)) = ?
            {role_clause}
            GROUP BY d.drugname_norm
        """, [pt_norm]).fetchdf()

        # Total drug reports per normalized drug
        if cfg.roles:
            roles_list = list(cfg.roles)
            roles_clause = "WHERE role_cod IN (" + ", ".join(repr(r) for r in roles_list) + ")"
        else:
            roles_clause = ""

        drug_df = con.execute(f"""
            SELECT
                drugname_norm,
                COUNT(DISTINCT primaryid) AS total_drug_reports
            FROM drug_dedup_norm
            {roles_clause}
            GROUP BY drugname_norm
        """).fetchdf()

        df = joint_df.merge(drug_df, on="drugname_norm", how="left")
        df["total_event_reports"] = total_event_reports
        df["total_reports"] = total_reports

        df["n10"] = df["total_drug_reports"] - df["n11"]
        df["n01"] = df["total_event_reports"] - df["n11"]
        df["n00"] = df["total_reports"] - df["n11"] - df["n10"] - df["n01"]

        df = df[df["n11"] >= cfg.min_n11].copy()

        cc = cfg.continuity_correction
        for col in ["n11", "n10", "n01", "n00"]:
            df[col] = df[col].astype(float)

        n11c = df["n11"].clip(lower=0) + cc
        n10c = df["n10"].clip(lower=0) + cc
        n01c = df["n01"].clip(lower=0) + cc
        n00c = df["n00"].clip(lower=0) + cc

        ror = (n11c * n00c) / (n10c * n01c)
        df["ror"] = ror

        with np.errstate(divide="ignore", invalid="ignore"):
            log_ror = np.log(ror)
            se_log_ror = np.sqrt(1.0 / n11c + 1.0 / n10c + 1.0 / n01c + 1.0 / n00c)
            z = cfg.ci_z
            df["ror_ci_low"] = np.exp(log_ror - z * se_log_ror)
            df["ror_ci_high"] = np.exp(log_ror + z * se_log_ror)

            risk_drug = n11c / (n11c + n10c)
            risk_other = n01c / (n01c + n00c)
            df["prr"] = risk_drug / risk_other

        df.sort_values(["ror", "n11"], ascending=[False, False], inplace=True)
        return df.reset_index(drop=True)

    finally:
        if close_con:
            con.close()

