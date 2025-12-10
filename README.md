# 🌠 **FAERS Radar**

*A complete, open-source pipeline for pharmacovigilance signal detection on the FDA FAERS database (2019–present)*

FAERS Radar is a research-oriented Python toolkit for working with **FAERS** (FDA Adverse Event Reporting System) data:

* 📥 Automated download & parsing (2019 → present)
* 🧹 FDA-style deduplication of cases
* 💊 Drug name normalization
* 📊 Classical disproportionality analysis (ROR/PRR + CIs)
* 🕒 Quarterly temporal signal computation (drug–event × time)
* 🔥 Novel *emergence* and *trend* detection algorithms for **emerging safety signals**
* 📓 Jupyter notebooks for exploration

This project aims to make real-world pharmacovigilance research accessible, transparent, and fully reproducible.

---

# 🔎 **What is FAERS and why does it matter?**

The **FDA Adverse Event Reporting System (FAERS)** collects millions of reports from:

* clinicians
* patients
* manufacturers

Each report describes a **suspected adverse drug reaction**, including:

* the **drug(s)** involved
* the **reaction(s)** (MedDRA terms)
* patient demographics
* outcome information

FAERS is a **spontaneous reporting system** — it contains valuable signals but also noise, biases, and duplicates.

Pharmacovigilance analysts and researchers use FAERS to detect:

### → **Potential safety signals**

associations between *a drug* and *an adverse event* that occur more often than expected.

These signals are **hypothesis-generating**, not proof of causality
—but they often precede formal safety actions.

---

# 🎯 **Project Goals**

FAERS Radar provides:

### 1. A *complete data engineering pipeline*

from raw FDA zips → structured, deduplicated database using DuckDB.

### 2. A *robust analytical core*

implementing standard pharmacovigilance statistics:

* **ROR** — Reporting Odds Ratio
* **PRR** — Proportional Reporting Ratio
* With confidence intervals and contingency tables (N11, N10, …)

### 3. A *temporal modeling layer*

computing ROR per **drug × event × quarter** to detect changes over time.

### 4. A *novel research layer*

based on:

* **emergence score** — how surprising the latest ROR is vs history
* **trend score** — slope of log-ROR over time
* **composite signal score** — innovation: signals that are strong + trending up

This architecture supports advanced pharmacovigilance, signal prioritization, ML research, and exploratory data science.

---

# 🏗️ **Project Architecture**

```
faers_radar/
│
├── faers_signals/
│   ├── download.py          # Download FAERS zips from FDA
│   ├── parse_ascii.py       # Extract + parse ASCII FAERS files
│   ├── dedup.py             # FDA-style deduplication
│   ├── drug_normalization.py# Normalize drug names (brand/generic variants)
│   ├── signals_classical.py # ROR/PRR disproportionality
│   ├── signals_temporal.py  # Quarterly time-series aggregation
│   └── emergence.py         # Emergence + slope-based novelty
│
├── notebooks/
│   ├── 01_inspect.ipynb          # Inspect warehouse
│   ├── 02_drug_level_signals.ipynb
│   ├── 03_temporal_signals.ipynb
│   └── 04_temporal_novelty.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🧬 **Methodology Overview (Didactic Explanation)**

This section explains *in simple terms* the analytical pipeline.

---

## 1️⃣ Download & Parse FAERS Data (2019 → present)

FAERS releases quarterly ASCII zip files such as:

```
2020Q1_ascii.zip
2020Q2_ascii.zip
...
```

Each contains tables like:

* **DEMO** — patient demographics
* **DRUG** — drugs involved
* **REAC** — reported reactions
* **OUTC** — outcomes
* **RPSR**, **THER**, **INDI**

We use a robust parser that handles:

* mixed encodings (latin-1)
* changing directory structures
* variations in filenames
* malformed rows (skipped safely)

Parsed data is stored in **DuckDB**, a fast embedded analytical database.

---

## 2️⃣ FDA-Style Deduplication

The same case can be reported multiple times (updates, follow-ups).
The FDA recommends keeping **the most recent version** per `caseid`.

We:

1. Sort by `fda_dt` (report date)
2. Keep the latest record per `caseid`
3. Apply same dedup logic to `DRUG`, `REAC`, etc. based on consistent `primaryid` mapping

This produces:

* `demo_dedup`
* `drug_dedup`
* `reac_dedup`
* …

These are the clean base tables for all analyses.

---

## 3️⃣ Drug Name Normalization

FAERS drug names vary:

```
"ATORVASTATIN"
"Atorvastatin 40 MG"
"ATORVASTATIN®"
"ATORVASTATINE" (misspelling)
"ATORVASTATIN TAB"
```

We normalize by:

* uppercasing
* removing dosage (e.g., “20 mg”, “0.5 mL”)
* removing punctuation
* removing trademark symbols
* collapsing whitespace

This produces a canonical `drugname_norm`.

---

## 4️⃣ Classical Disproportionality (Global Analysis)

For each (drug, adverse event) pair:

We build the contingency table:

|                  | event present | event absent |
| ---------------- | ------------- | ------------ |
| **drug present** | N11           | N10          |
| **drug absent**  | N01           | N00          |

From this we calculate:

* **ROR** = (N11 / N10) / (N01 / N00)
* **PRR** = risk_drug / risk_other
* CIs via log-ROR

The notebook `02_drug_level_signals.ipynb` covers this.

---

## 5️⃣ Temporal Signal Engine (Quarterly ROR)

We compute ROR **per quarter**:

```
(drug, PT, year, quarter) → ROR_t
```

This gives a time series per pair, e.g.:

```
Q1 2019 → 1.2
Q2 2019 → 1.3
Q3 2019 → 2.1
...
Q2 2024 → 5.6
```

Stored in:

```
signals_quarterly
```

This is the basis of all temporal novelty analysis.

---

## 6️⃣ Novelty Layer: Emergence & Trend Scores

### 🔥 **Emergence score (emergence_z)**

“How surprising is the **latest** ROR compared with the drug’s historical baseline?”

* Compute log(ROR)
* Compare last point vs baseline mean/std
* Z-score = (latest - baseline_mean) / sd

High emergence_z = sudden recent increase.

---

### 📈 **Trend score (slope_log_ror)**

“Is ROR increasing over time overall?”

* Fit a simple linear regression:
  `log(ROR) ~ time`
* Use the slope as an indicator of trend

Positive slope = upward trend.

---

### ⭐ **Composite Signal Score**

We combine the two (plus current ROR strength):

```
signal_score = max(slope_log_ror, 0) 
               * emergence_z
               * log(latest_ror + 1)
```

> High score =
> **currently strong** + **increasing over time** + **recent jump**.

This is conceptually similar to early-warning indicators used in surveillance systems.

The notebook `04_temporal_novelty.ipynb` explores this.

---

# 📦 **Installation**

```bash
git clone https://github.com/<your-username>/faers_radar.git
cd faers_radar
pip install -r requirements.txt
```

---

# 🚀 **Quickstart**

### Step 1: Download FAERS data

```bash
python -m faers_signals.download
```

### Step 2: Parse & load into DuckDB

```bash
python -m faers_signals.parse_ascii
```

### Step 3: Deduplicate

```bash
python -m faers_signals.dedup
```

### Step 4: Normalize drug names

```bash
python -m faers_signals.drug_normalization
```

### Step 5: Build temporal signals

```bash
python -m faers_signals.signals_temporal
```

Open the notebooks in **notebooks/** for exploration.

---

# 📊 Example: Compute temporal emergence for a drug

```python
from faers_signals.emergence import compute_emergence_scores_for_drug
from faers_signals.config import WAREHOUSE_DB_PATH
import duckdb

con = duckdb.connect(str(WAREHOUSE_DB_PATH))

em = compute_emergence_scores_for_drug("ATORVASTATIN")
em.head()
```

---

# 📈 Example: Global emerging signals

```python
from faers_signals.emergence import compute_global_emergence_scores

global_em = compute_global_emergence_scores(min_total_drug_reports=500)
global_em.head(20)
```

---

# 📚 **Disclaimer**

FAERS is a **spontaneous reporting system**. Signals detected here:

* **do not establish causality**
* may reflect reporting biases
* require clinical and epidemiological investigation

This software is intended for research and educational purposes only.

---

# 🤝 Contributing

Pull requests are welcome!
Useful contributions include:

* additional signal detection models
* embedding-based novelty
* ATC/RxNorm integration
* UI dashboards (Streamlit)

---

# ⭐ License

MIT License.

---
