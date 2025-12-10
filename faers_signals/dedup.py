import duckdb

from .config import WAREHOUSE_DB_PATH


def deduplicate_demo():
    con = duckdb.connect(str(WAREHOUSE_DB_PATH))

    # Optionally, if you’ve loaded "deletedCases" from the quarterly deleted file,
    # you can store it as a table (e.g., deleted_cases(primaryid)) and filter those out.

    # Remove duplicates with FDA recommended logic
    # We'll create a new table demo_dedup and a mapping of valid primaryids.
    con.execute("""
        CREATE OR REPLACE TABLE demo_dedup AS
        WITH ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY caseid
                    ORDER BY
                        fda_dt DESC NULLS LAST,
                        primaryid::BIGINT DESC NULLS LAST
                ) AS rn
            FROM demo
        )
        SELECT * EXCLUDE (rn)
        FROM ranked
        WHERE rn = 1
    """)

    # Create a helper table of kept primary IDs
    con.execute("""
        CREATE OR REPLACE TABLE primaryid_kept AS
        SELECT DISTINCT primaryid FROM demo_dedup
    """)

    # Filter other tables to only those primaryids present in demo_dedup
    for table in ["drug", "reac", "outc", "rpsr", "ther", "indi"]:
        # Only run if table exists
        exists = con.execute(
            f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name='{table}'"
        ).fetchone()[0]
        if not exists:
            continue

        con.execute(f"""
            CREATE OR REPLACE TABLE {table}_dedup AS
            SELECT t.*
            FROM {table} t
            JOIN primaryid_kept k
            ON t.primaryid = k.primaryid
        """)

    con.close()


if __name__ == "__main__":
    deduplicate_demo()
