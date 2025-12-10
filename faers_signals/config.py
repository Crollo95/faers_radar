from pathlib import Path

# Base project directory (you can also drive this from env variables)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTRACTED_DIR = DATA_DIR / "extracted"
WAREHOUSE_DIR = DATA_DIR / "warehouse"

WAREHOUSE_DB_PATH = WAREHOUSE_DIR / "faers_2019_present.duckdb"

# Years we target
START_YEAR = 2019
END_YEAR = 2025  # adjust as new data comes out

# Quarters (labeling matches FDA site: 'January - March', etc., but we’ll index by Q1..Q4)
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]

