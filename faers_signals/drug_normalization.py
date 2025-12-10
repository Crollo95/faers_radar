# faers_signals/drug_normalization.py

from __future__ import annotations

import duckdb

from .config import WAREHOUSE_DB_PATH


def normalize_drugnames():
    """
    Create a normalized version of drug_dedup called drug_dedup_norm,
    with an extra column `drugname_norm`.

    Normalization steps (in SQL, all uppercased):
    - Trim whitespace
    - Remove trademark symbols (®, ™, ©)
    - Collapse multiple spaces -> single space
    - Remove trailing punctuation (.,;: etc.)
    - Remove common strength patterns like "500 MG", "10MG", "0.5 MG/ML"
    - Final trim of spaces

    This is a deliberately conservative but useful canonicalization.
    """
    con = duckdb.connect(str(WAREHOUSE_DB_PATH))

    # We’ll build this in stages using CTEs so it stays readable.
    con.execute("""
        CREATE OR REPLACE TABLE drug_dedup_norm AS
        WITH base AS (
            SELECT
                *,
                UPPER(TRIM(COALESCE(drugname, ''))) AS dn
            FROM drug_dedup
        ),
        step1 AS (
            -- remove trademark-like symbols
            SELECT
                *,
                REGEXP_REPLACE(dn, '[®™©]', '') AS dn1
            FROM base
        ),
        step2 AS (
            -- collapse multiple whitespace
            SELECT
                *,
                REGEXP_REPLACE(dn1, '[[:space:]]+', ' ') AS dn2
            FROM step1
        ),
        step3 AS (
            -- remove trailing punctuation (one or more punctuation chars at end)
            SELECT
                *,
                REGEXP_REPLACE(dn2, '[[:punct:]]+$', '') AS dn3
            FROM step2
        ),
        step4 AS (
            -- remove common strength patterns like:
            -- "500 MG", "10MG", "0.5 MG/ML", "40 MCG", "2 G", "1000 IU", "10 UNITS", "100 ML"
            SELECT
                *,
                REGEXP_REPLACE(
                    dn3,
                    '\\b[0-9]+(\\.[0-9]+)?\\s*(MG|MCG|G|IU|UNITS|ML)\\b',
                    ''
                ) AS dn4
            FROM step3
        ),
        norm AS (
            SELECT
                *,
                -- final pass: collapse spaces again and trim
                TRIM(REGEXP_REPLACE(dn4, '[[:space:]]+', ' ')) AS drugname_norm
            FROM step4
        )
        SELECT
            * EXCLUDE (dn, dn1, dn2, dn3, dn4)
        FROM norm
    """)

    con.close()
    print("Created/updated table drug_dedup_norm with normalized drugname_norm.")
    

if __name__ == "__main__":
    normalize_drugnames()
