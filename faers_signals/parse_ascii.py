import zipfile
from pathlib import Path
from typing import Dict, List

import duckdb
import pandas as pd
from tqdm import tqdm

from .config import RAW_DIR, EXTRACTED_DIR, WAREHOUSE_DB_PATH, START_YEAR, END_YEAR


ASCII_TABLE_FILES = {
    "demo": "DEMO",
    "drug": "DRUG",
    "reac": "REAC",
    "outc": "OUTC",
    "rpsr": "RPSR",
    "ther": "THER",
    "indi": "INDI",
}


def extract_zip(zip_path: Path, dest_dir: Path) -> List[Path]:
    """
    Extracts a FAERS quarterly ASCII zip into a directory and
    returns a list of extracted file paths.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    return [dest_dir / name for name in zf.namelist()]


def find_table_files(root_dir: Path) -> Dict[str, Path]:
    """
    Recursively locate DEMO/DRUG/REAC/... ASCII files beneath root_dir.

    Returns a dict { "demo": Path(...), ... }.

    Uses robust naming heuristics to handle different FAERS layouts:
    - Nested 'ascii/' or 'ASCII/' folders
    - Filenames like DEMO19Q1.TXT, DEMO2020Q1.TXT, DEMOGRAPHIC.TXT, etc.
    """
    table_paths: Dict[str, Path] = {}

    # Keywords that identify each table (case-insensitive)
    patterns = {
        "demo": ["DEMO", "DEMOGRAPH", "DEMO_"],  # demo / demographics
        "drug": ["DRUG"],
        "reac": ["REAC", "REACTION"],
        "outc": ["OUTC", "OUTCOME"],
        "rpsr": ["RPSR", "REPORT", "RPSR_"],
        "ther": ["THER", "THERAP"],
        "indi": ["INDI", "INDICATION"],
    }

    # Scan all .txt / .TXT under root_dir
    all_txt_files = list(root_dir.rglob("*.txt")) + list(root_dir.rglob("*.TXT"))

    for logical_name, keywords in patterns.items():
        candidates = []
        for path in all_txt_files:
            name_u = path.name.upper()
            # Skip obvious non-data files if they sneak in (e.g. ASC_NTS.TXT, READMEs)
            if "ASC_NTS" in name_u or "READ" in name_u or "FAQ" in name_u:
                continue

            if any(kw in name_u for kw in keywords):
                candidates.append(path)

        if candidates:
            # Choose the largest candidate (data files >> docs)
            best = max(candidates, key=lambda p: p.stat().st_size)
            table_paths[logical_name] = best

    return table_paths


def read_ascii_table(path: Path, sep: str = "$") -> pd.DataFrame:
    """
    Read a FAERS ASCII table into a pandas DataFrame.

    - Files are '$'-delimited.
    - Encoding is often Latin-1 / ISO-8859-1 rather than UTF-8.
    - We read everything as string to avoid dtype surprises.
    """
    df = pd.read_csv(
        path,
        sep=sep,
        header=0,
        dtype=str,
        engine="python",        # more forgiving parser
        encoding="latin-1",     # key change: handle non-UTF8 bytes
        on_bad_lines="warn",    # or "skip" if you want to drop malformed lines
    )
    return df




def init_duckdb() -> duckdb.DuckDBPyConnection:
    WAREHOUSE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(WAREHOUSE_DB_PATH))
    return con


def append_quarter_to_warehouse(year: int, quarter: str):
    """
    Parse a single year-quarter ASCII zip and append its tables to DuckDB.

    quarter is 'Q1','Q2','Q3','Q4'.
    """
    zip_name = f"faers_ascii_{year}{quarter.lower()}.zip"
    zip_path = RAW_DIR / zip_name
    if not zip_path.exists():
        print(f"[warn] {zip_path} not found, skipping")
        return

    print(f"[info] Processing {zip_name}")
    extracted_dir = EXTRACTED_DIR / f"{year}{quarter.lower()}"
    extract_zip(zip_path, extracted_dir)

    # NEW: robust recursive search
    table_files = find_table_files(extracted_dir)
    if not table_files:
        print(f"[warn] No table files found in {extracted_dir}")
        return

    con = init_duckdb()

    for table_name, file_path in table_files.items():
        print(f"  [table] {table_name}: {file_path.name}")
        df = read_ascii_table(file_path)
        df["year"] = year
        df["quarter"] = quarter

        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} AS
            SELECT * FROM df LIMIT 0
            """
        )
        con.execute(f"INSERT INTO {table_name} SELECT * FROM df")



def build_warehouse(start_year: int = START_YEAR, end_year: int = END_YEAR):
    for year in range(start_year, end_year + 1):
        for quarter in ["Q1", "Q2", "Q3", "Q4"]:
            append_quarter_to_warehouse(year, quarter)


if __name__ == "__main__":
    build_warehouse()
