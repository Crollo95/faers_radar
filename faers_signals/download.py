import time
from pathlib import Path

import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError
from urllib3.exceptions import ProtocolError

from .config import RAW_DIR, START_YEAR, END_YEAR

HEADERS = {
    "User-Agent": "faers-signals-research-tool/0.1 (contact: your_email@example.com)"
}

BASE_URL = "https://fis.fda.gov/content/Exports/faers_ascii_{year}Q{q}.zip"


def download_file(
    url: str,
    dest: Path,
    chunk_size: int = 1 << 20,
    max_retries: int = 5,
    backoff_sec: float = 2.0,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest.name} already exists")
        return

    for attempt in range(1, max_retries + 1):
        print(f"[download] {url} -> {dest} (attempt {attempt}/{max_retries})")
        try:
            with requests.get(
                url, stream=True, headers=HEADERS, timeout=60
            ) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            # success: return
            return

        except (ChunkedEncodingError, ConnectionError, ProtocolError) as e:
            print(f"[warn] Transient download error for {url}: {e}")
            # delete partial file if present
            if dest.exists():
                try:
                    dest.unlink()
                except OSError:
                    pass

            if attempt == max_retries:
                print(f"[error] Giving up on {url} after {max_retries} attempts.")
                return
            # simple backoff
            time.sleep(backoff_sec * attempt)

        except requests.HTTPError as e:
            print(f"[error] HTTP error for {url}: {e}")
            # If it’s a 404 or other permanent error, don’t retry
            if dest.exists():
                try:
                    dest.unlink()
                except OSError:
                    pass
            return


def download_faers_quarters(start_year: int = START_YEAR, end_year: int = END_YEAR):
    print(f"[info] Downloading FAERS ASCII zips {start_year}–{end_year} using URL pattern…")
    for year in range(start_year, end_year + 1):
        for q in range(1, 5):
            url = BASE_URL.format(year=year, q=q)
            filename = f"faers_ascii_{year}q{q}.zip"
            dest = RAW_DIR / filename
            download_file(url, dest)
            time.sleep(1)


if __name__ == "__main__":
    download_faers_quarters()
