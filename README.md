# 🌠 **FAERS Radar**

*A complete, open-source pipeline for pharmacovigilance signal detection on the FDA FAERS database (2019–present)*

FAERS Radar is a research-oriented Python toolkit for working with **FAERS** (FDA Adverse Event Reporting System) data:

* 📥 Automated download & parsing (2019 → present)
* 🧹 FDA-style deduplication of cases
* 💊 Drug name normalization
* 📊 Classical disproportionality analysis (ROR/PRR + CIs)
* 🕒 Quarterly temporal signal computation (drug–event × time)
* 🔥 Novel *emergence* and *trend* detection algorithms for **emerging safety signals**
* 🧭 NEW: **Embedding-based structural novelty detection**
  (unexpected drug–event associations given global co-occurrence structure)
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

These signals are **hypothesis-generating**, not proof of causality —
but they often precede formal safety actions, labeling changes, or epidemiological studies.

---

# 🎯 **Project Goals**

FAERS Radar provides:

### 1. A *complete data engineering pipeline*

From raw FDA zips → structured, deduplicated database using DuckDB.

### 2. A *robust analytical core*

Implementing standard pharmacovigilance statistics:

* **ROR** — Reporting Odds Ratio
* **PRR** — Proportional Reporting Ratio
* Confidence intervals and full N11/N10/N01/N00 contingency tables

### 3. A *temporal modeling layer*

Computing ROR per **drug × event × quarter** to detect changes over time.

### 4. A *novel temporal novelty layer*

Based on:

* **emergence score** — how surprising the latest ROR is vs history
* **trend score** — slope of log-ROR over time
* **composite temporal score** — signals that are strong + trending upward

### 5. A *structural novelty layer* (NEW)

Using low-rank **drug–event embeddings**:

* Builds a global co-occurrence matrix
* Learns latent factors via truncated SVD
* Detects **unexpected drug–event associations**
  (e.g., unusually high reporting relative to model expectations)

Together, these layers support signal prioritization, safety hypothesis generation, machine learning research, and exploratory drug safety analytics.

---

# 🏗️ **Project Architecture**

```
faers_radar/
│
├── faers_signals/
│   ├── download.py           # Download FAERS zips
│   ├── parse_ascii.py        # Extract + parse ASCII FAERS files
│   ├── dedup.py              # FDA-style deduplication
│   ├── drug_normalization.py # Normalize drug names
│   ├── signals_classical.py  # ROR/PRR disproportionality
│   ├── signals_temporal.py   # Quarterly time-series aggregation
│   ├── emergence.py          # Emergence + trend novelty
│   └── embeddings.py         # Embedding-based structural novelty (NEW)
│
├── notebooks/
│   ├── 01_inspect.ipynb             # Inspect warehouse
│   ├── 02_drug_level_signals.ipynb  # Classical signals
│   ├── 03_temporal_signals.ipynb    # Quarterly ROR
│   ├── 04_temporal_novelty.ipynb    # Emergence + trend
│   └── 05_embedding_novelty.ipynb   # Structural novelty (NEW)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🧬 **Methodology Overview (Didactic Explanation)**

This section explains *in simple terms* how the entire analysis works.

---

## 1️⃣ Download & Parse FAERS Data

FAERS releases quarterly ASCII zip files (e.g., `2021Q4_ascii.zip`).
Each contains tables:

* **DEMO**, **DRUG**, **REAC**, **OUTC**, **RPSR**, **THER**, **INDI**

Challenges we solve:

* mixed encodings (Latin-1 vs UTF-8)
* inconsistent folder names
* changing file naming conventions
* malformed rows

Everything is loaded into a unified **DuckDB warehouse**.

---

## 2️⃣ FDA-Style Deduplication

A single adverse event case may be updated over time.

We:

1. Sort reports by FDA receipt date
2. Keep only the **latest version** per `caseid`
3. Use consistent `primaryid` keys across DEMO/DRUG/REAC tables

This produces clean deduplicated tables, e.g. `demo_dedup`, `drug_dedup`, `reac_dedup`.

---

## 3️⃣ Drug Name Normalization

Drug names are messy in FAERS:

```
"Atorvastatin"
"ATORVASTATINA"
"Atorvastatin Calcium 40MG"
"ATORVASTATIN®"
```

We normalize using heuristics that remove:

* dosages
* formulations
* trademarks
* punctuation
* casing differences

This yields a canonical `drugname_norm`.

---

## 4️⃣ Classical Disproportionality (ROR / PRR)

For each drug–event pair, we compute:

|                  | event present | event absent |
| ---------------- | ------------- | ------------ |
| **drug present** | N11           | N10          |
| **drug absent**  | N01           | N00          |

Then:

* **ROR** = (N11/N10) / (N01/N00)
* **PRR**
* Confidence intervals

These form the backbone of many safety surveillance systems.

---

## 5️⃣ Temporal Signal Engine (Quarterly)

We compute ROR per quarter:

```
(drug, event, year, quarter) → ROR_t
```

This allows us to see *trends* rather than only snapshot associations.

The table is stored as:

```
signals_quarterly
```

---

## 6️⃣ Novelty Layer I — Temporal Novelty

*(Emergence & Trend)*

### 🔥 Emergence score

“How surprising is the latest ROR vs historical baseline?”

* compute log(ROR)
* baseline = all past quarters
* Z-score of last value vs baseline variance

### 📈 Trend score

“Is the association increasing over time?”

* fit a linear regression: `log(ROR) ~ time`
* slope > 0 ⇒ upward trend

### ⭐ Composite temporal score

```
signal_score =
    max(slope_log_ror, 0)
  * emergence_z
  * log(latest_ror + 1)
```

This prioritizes:

* strong associations
* that are trending upward
* with a recent jump

---

## 7️⃣ Novelty Layer II — **Structural Novelty (NEW)**

*(Embedding-based detection of unexpected associations)*

FAERS can be viewed as a **drug × event matrix**:

```
          Event1  Event2  Event3 ...
DrugA       12      0       4
DrugB        2      5      11
DrugC        0      8       1
...
```

We aggregate total co-occurrence counts across years:

```
(drug, event) → N11_total
```

Then we:

1. Build a sparse matrix of **log(1 + N11_total)**
2. Fit a **low-rank model** using Truncated SVD (latent factors)
3. Predict expected log-co-occurrence from embeddings
4. Compute **structural novelty** via residuals:

```
structural_z =
    (observed_log - predicted_log)
    standardized across all pairs
```

### Interpretation:

*High structural_z ⇒ the drug–event pair occurs **much more often** than predicted by the global co-occurrence structure.*

This is **orthogonal** to temporal novelty:

* Temporal novelty asks: *Is it increasing suddenly?*
* Structural novelty asks: *Is it unusual in general compared to similar drugs and events?*

This combination is extremely powerful for signal prioritization.

The notebook `05_embedding_novelty.ipynb` demonstrates this.

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

### Step 2: Parse into DuckDB

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

### Step 6 (optional): Compute structural novelty

Using notebooks or the Python API.

---

# 📊 Example: Temporal emergence for a drug

```python
from faers_signals.emergence import compute_emergence_scores_for_drug
import duckdb
from faers_signals.config import WAREHOUSE_DB_PATH

con = duckdb.connect(str(WAREHOUSE_DB_PATH))
em = compute_emergence_scores_for_drug("ATORVASTATIN")
em.head()
```

---

# 🔍 Example: Global emerging signals

```python
from faers_signals.emergence import compute_global_emergence_scores

global_em = compute_global_emergence_scores(min_total_drug_reports=500)
global_em.head(20)
```

---

# 🔮 Example: Structural novelty (embedding-based)

```python
from faers_signals.embeddings import compute_embedding_novelty

cooc, drug_emb, event_emb, novelty = compute_embedding_novelty()
novelty.head(20)
```

This ranks drug–event pairs by **unexpectedness** relative to the global FAERS structure.

---

# 📚 **Disclaimer**

FAERS is a **spontaneous reporting system**.
Signals detected here:

* **do not establish causality**
* may reflect reporting biases
* require clinical and epidemiological investigation

FAERS Radar is intended for research and educational purposes only.

---

# 🤝 Contributing

Pull requests are welcome!
Useful contributions include:

* additional signal detection models
* improved time-series analysis
* embedding-based enhancements
* ATC/RxNorm integration
* UI dashboards (Streamlit)

---

# ⭐ License

MIT License.

---