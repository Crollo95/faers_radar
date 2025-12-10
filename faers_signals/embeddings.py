# faers_signals/embeddings.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from .config import WAREHOUSE_DB_PATH


@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding-based structural novelty.
    """
    min_pair_n11: int = 5          # minimum total N11 for (drug, pt) pair
    min_drug_n11: int = 50         # minimum total N11 per drug (across all pts)
    min_event_n11: int = 50        # minimum total N11 per event (across all drugs)
    n_components: int = 50         # latent dimension for embeddings
    random_state: int = 0


def get_connection(db_path: Optional[str] = None) -> duckdb.DuckDBPyConnection:
    path = db_path or str(WAREHOUSE_DB_PATH)
    return duckdb.connect(path)


def build_cooccurrence_df(
    cfg: Optional[EmbeddingConfig] = None,
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    """
    Build a drug-event co-occurrence DataFrame from signals_quarterly.

    We aggregate N11 over all quarters:

        n11_total = SUM(n11)

    Then we filter:
      - pairs with n11_total >= min_pair_n11
      - drugs with total_n11_drug >= min_drug_n11
      - events with total_n11_event >= min_event_n11

    Returns
    -------
    pd.DataFrame with columns:
        drugname_norm
        pt
        n11_total
    """
    if cfg is None:
        cfg = EmbeddingConfig()

    close_con = False
    if con is None:
        con = get_connection()
        close_con = True

    try:
        # 1) aggregate N11 over all quarters
        cooc = con.execute("""
            SELECT
                drugname_norm,
                pt,
                SUM(n11) AS n11_total
            FROM signals_quarterly
            GROUP BY drugname_norm, pt
        """).fetchdf()

        # filter by pair strength
        cooc = cooc[cooc["n11_total"] >= cfg.min_pair_n11].copy()

        # 2) filter by per-drug total support
        drug_totals = (
            cooc.groupby("drugname_norm")["n11_total"]
            .sum()
            .reset_index(name="drug_n11_total")
        )
        strong_drugs = drug_totals[
            drug_totals["drug_n11_total"] >= cfg.min_drug_n11
        ]["drugname_norm"]

        cooc = cooc[cooc["drugname_norm"].isin(strong_drugs)].copy()

        # 3) filter by per-event total support
        event_totals = (
            cooc.groupby("pt")["n11_total"]
            .sum()
            .reset_index(name="event_n11_total")
        )
        strong_events = event_totals[
            event_totals["event_n11_total"] >= cfg.min_event_n11
        ]["pt"]

        cooc = cooc[cooc["pt"].isin(strong_events)].copy()

        cooc.reset_index(drop=True, inplace=True)
        return cooc

    finally:
        if close_con:
            con.close()


def build_sparse_matrix(
    cooc: pd.DataFrame,
) -> Tuple[sparse.csr_matrix, Dict[str, int], Dict[str, int]]:
    """
    Build a sparse matrix X from a co-occurrence DataFrame.

    Rows    = drugs (drugname_norm)
    Columns = events (pt)
    Values  = log(1 + n11_total)  (to stabilize scale)

    Returns
    -------
    X : csr_matrix (n_drugs x n_events)
    drug_index : dict mapping drugname_norm -> row index
    event_index : dict mapping pt -> col index
    """
    drugs = sorted(cooc["drugname_norm"].unique())
    events = sorted(cooc["pt"].unique())

    drug_index = {d: i for i, d in enumerate(drugs)}
    event_index = {e: j for j, e in enumerate(events)}

    rows = cooc["drugname_norm"].map(drug_index).to_numpy()
    cols = cooc["pt"].map(event_index).to_numpy()
    data = np.log1p(cooc["n11_total"].to_numpy().astype(float))  # log(1 + n11_total)

    n_drugs = len(drugs)
    n_events = len(events)

    X = sparse.csr_matrix((data, (rows, cols)), shape=(n_drugs, n_events))
    return X, drug_index, event_index


def fit_svd_embeddings(
    X: sparse.csr_matrix,
    cfg: Optional[EmbeddingConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, TruncatedSVD]:
    """
    Fit a TruncatedSVD (low-rank) model to the log-count matrix X.

    Parameters
    ----------
    X : csr_matrix
        n_drugs x n_events matrix of log(1 + n11_total).
    cfg : EmbeddingConfig, optional

    Returns
    -------
    drug_embeddings : pd.DataFrame
        shape (n_drugs, n_components), columns like "dim_0", "dim_1", ...
    event_embeddings : pd.DataFrame
        shape (n_events, n_components)
    svd_model : TruncatedSVD
    """
    if cfg is None:
        cfg = EmbeddingConfig()

    n_components = min(cfg.n_components, min(X.shape) - 1)
    if n_components <= 1:
        raise ValueError(
            f"n_components too large for matrix shape {X.shape}; "
            f"got {cfg.n_components}, effective {n_components}"
        )

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=cfg.random_state,
    )
    # U (drug factors) and V (event factors) in the latent space
    U = svd.fit_transform(X)              # shape: (n_drugs, k)
    Vt = svd.components_                  # shape: (k, n_events)
    V = Vt.T                              # shape: (n_events, k)

    dim_cols = [f"dim_{i}" for i in range(n_components)]
    drug_embeddings = pd.DataFrame(U, columns=dim_cols)
    event_embeddings = pd.DataFrame(V, columns=dim_cols)

    return drug_embeddings, event_embeddings, svd


def compute_structural_novelty(
    cooc: pd.DataFrame,
    drug_embeddings: pd.DataFrame,
    event_embeddings: pd.DataFrame,
    drug_index: Dict[str, int],
    event_index: Dict[str, int],
) -> pd.DataFrame:
    """
    Given:
      - cooc          : DataFrame with (drugname_norm, pt, n11_total)
      - drug_emb      : DataFrame with row-wise drug embeddings (same order as drug_index mapping)
      - event_emb     : DataFrame with row-wise event embeddings (same order as event_index mapping)
      - index maps    : mapping names to row indices

    Compute, for each (drug, pt) pair:
      - observed_log   = log(1 + n11_total)
      - predicted_log  = dot(embedding_drug, embedding_event)
      - residual       = observed_log - predicted_log
      - structural_z   = z-score of residual across all pairs

    Returns
    -------
    pd.DataFrame with columns:
        drugname_norm
        pt
        n11_total
        observed_log
        predicted_log
        residual
        structural_z
    """
    # Ensure alignment
    dim_cols = drug_embeddings.columns.tolist()

    obs_log = np.log1p(cooc["n11_total"].to_numpy().astype(float))

    pred_log = []
    for _, row in cooc.iterrows():
        d = row["drugname_norm"]
        e = row["pt"]
        di = drug_index[d]
        ei = event_index[e]

        dv = drug_embeddings.loc[di, dim_cols].to_numpy()
        ev = event_embeddings.loc[ei, dim_cols].to_numpy()
        pred_log.append(np.dot(dv, ev))

    pred_log = np.array(pred_log, dtype=float)
    residual = obs_log - pred_log

    # Standardize residuals across all pairs
    mu = residual.mean()
    sigma = residual.std(ddof=1) if residual.size > 1 else 0.0
    if sigma == 0 or np.isnan(sigma):
        structural_z = np.zeros_like(residual)
    else:
        structural_z = (residual - mu) / sigma

    out = cooc.copy()
    out["observed_log"] = obs_log
    out["predicted_log"] = pred_log
    out["residual"] = residual
    out["structural_z"] = structural_z

    # Sort descending by structural_z: most "unexpected" high co-occurrence first
    out = out.sort_values("structural_z", ascending=False).reset_index(drop=True)
    return out


def compute_embedding_novelty(
    cfg: Optional[EmbeddingConfig] = None,
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    High-level convenience function:

    1. Build co-occurrence DataFrame from signals_quarterly (aggregated N11)
    2. Build sparse matrix X = log(1 + n11_total)
    3. Fit truncated SVD embeddings for drugs & events
    4. Compute structural novelty scores (residual z-scores)

    Returns
    -------
    cooc : pd.DataFrame
        (drugname_norm, pt, n11_total)
    embeddings : pd.DataFrame
        drug & event embeddings separately (returned as two dfs)
    structural_novelty : pd.DataFrame
        (drugname_norm, pt, n11_total, observed_log, predicted_log, residual, structural_z)
    """
    if cfg is None:
        cfg = EmbeddingConfig()

    close_con = False
    if con is None:
        con = get_connection()
        close_con = True

    try:
        # 1) co-occurrence
        cooc = build_cooccurrence_df(cfg=cfg, con=con)

        # 2) sparse matrix
        X, drug_index, event_index = build_sparse_matrix(cooc)

        # 3) embeddings
        drug_emb, event_emb, svd = fit_svd_embeddings(X, cfg=cfg)

        # 4) structural novelty
        novelty_df = compute_structural_novelty(
            cooc=cooc,
            drug_embeddings=drug_emb,
            event_embeddings=event_emb,
            drug_index=drug_index,
            event_index=event_index,
        )

        # attach identifiers to embedding tables
        # (so you know which row corresponds to which drug / event)
        drug_names_by_idx = {i: d for d, i in drug_index.items()}
        event_names_by_idx = {j: e for e, j in event_index.items()}

        drug_emb = drug_emb.copy()
        drug_emb.insert(0, "drugname_norm", pd.Series(
            [drug_names_by_idx[i] for i in range(len(drug_emb))],
            index=drug_emb.index
        ))

        event_emb = event_emb.copy()
        event_emb.insert(0, "pt", pd.Series(
            [event_names_by_idx[j] for j in range(len(event_emb))],
            index=event_emb.index
        ))

        return cooc, drug_emb, event_emb, novelty_df

    finally:
        if close_con:
            con.close()
