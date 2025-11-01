# Solar Flare Data Processing Pipeline

This directory contains all scripts used to **download, preprocess, and transform** the solar flare datasets into a structured, model-ready format.  
The pipeline integrates both **magnetic field time-series data** and **SDO image data**, aligning them for multimodal flare prediction tasks.

---

## Overview

The pipeline manages the complete data preparation lifecycle — from raw downloads to engineered features.

**Scripts in this directory:**
```

loader_2020.py  
loader_stealth.py  
loader_sdo.py  
preprocess_2020.py  
preprocess_sdo.py  
process_magnetic_data.py

```

---

## Workflow Summary

1. **Download Phase (Loaders)**  
   Downloads raw datasets from their official or Kaggle sources and organizes them under `data/raw/`.
   - `loader_2020.py`: Downloads the 2020 magnetic field dataset (IEEE Big Data Cup 2020).  
   - `loader_stealth.py`: Downloads additional stealth event magnetic field data.  
   - `loader_sdo.py`: Downloads the SDO image dataset and associated metadata (SDOBenchmark).

2. **Preprocessing Phase**  
   Converts raw JSON/CSV data and image metadata into clean, Dask-compatible Parquet and CSV formats.
   - `preprocess_2020.py`: Processes magnetic field data.
   - `preprocess_sdo.py`: Builds structured metadata for SDO image data.

3. **Feature Engineering Phase**  
   - `process_magnetic_data.py`: Performs log transforms, time indexing, and feature selection to create a model-ready dataset.

---

## Dataset Sources

| Dataset | Description | Source |
|----------|--------------|---------|
| **Solar Flare Prediction from Time Series Data (2020)** | Time-series magnetic field parameters from SDO/HMI vector magnetograms, labeled with flare intensity. | [GSU Dataset (IEEE Big Data Cup 2020)](https://dmlab.cs.gsu.edu/solar/data/data-comp-2020/) |
| **SDOBenchmark** | Image-sequence dataset of active solar regions for flare prediction. | [Kaggle – SDOBenchmark](https://www.kaggle.com/datasets/fhnw-i4ds/sdobenchmark) |
| **Stealth Dataset** *(optional)* | Supplemental magnetic field observations for stealth CMEs. | [Kaggle – Stealth Technologies](https://www.kaggle.com/datasets/stealthtechnologies/solar-flares-dataset) |

---

## Download Scripts

### `loader_2020.py`, `loader_stealth.py`, `loader_sdo.py`

**Purpose:**  
Automate the dataset downloads and ensure consistent directory structures.

**Details:**

- `loader_2020.py`:  
  Downloads the **IEEE Big Data Cup 2020** magnetic field dataset, consisting of **four JSON files**:
```

train_partition1_data.json  
train_partition2_data.json  
train_partition3_data.json  
test_4_5_data.json

```
Collectively, these files total **≈17 GB** of raw data.

- `loader_stealth.py`:  
Handles downloading of auxiliary stealth datasets.

- `loader_sdo.py`:  
Downloads the **SDO image dataset** (SDOBenchmark).  

⚠️ **Important Note on SDO Downloads:**  
The SDO dataset has a **case-sensitive duplication issue** — the same files appear under four differently cased folder names (`AR1234/`, `ar1234/`, `Ar1234/`, `aR1234/`).  
As a result, you will **download four copies** of the dataset, leading to a total of:
- **2.8 GB × 4 = ~11.2 GB**

**Output:**  
All raw data will be saved under:
```

data/raw/  
├── solar_flare_2020/  
├── sdo/  
└── stealth/

```

---

## Preprocessing Scripts

### `preprocess_2020.py`

**Purpose:**  
Converts the raw 17 GB magnetic field JSON data into efficient, structured Parquet files for Dask-based processing.

**Input:**  
Four JSON files from the 2020 dataset (`train_partition1_data.json`, …, `test_4_5_data.json`).

**Process:**
1. Reads and normalizes the JSON structure into tabular format.  
2. Cleans, validates, and splits data into training and testing partitions.  
3. Saves results as **chunked Parquet files**.

**Output:**  
```

data/preprocessed/train/  
data/preprocessed/test/

```
Each split contains multiple Parquet files (≈95 total, ~5 GB combined).

---

### `preprocess_sdo.py`

**Purpose:**  
Processes the SDO (Solar Dynamics Observatory) **image metadata** and builds a structured CSV index mapping each image sequence to its corresponding label and disk path.

**Input:**  
`meta_data.csv` and the image folder hierarchy downloaded by `loader_sdo.py`.

**Process:**
1. Reads the raw `meta_data.csv`.  
2. Parses unique IDs to extract **Active Region (AR)** numbers and timestamps.  
3. Derives the **flare label** (`X`, `M`, `C`, `Q`) based on `peak_flux`.  
4. Constructs the **absolute file paths** to each image sequence.  
5. Verifies dataset integrity to ensure image and metadata consistency.  
6. Outputs cleaned metadata CSVs.

**Output:**  
```

data/processed/sdo/  
├── train_metadata.csv  
└── test_metadata.csv

```

---

## Feature Engineering Script

### `process_magnetic_data.py`

**Purpose:**  
Transforms the cleaned magnetic field Parquet data into a **final feature-rich dataset** suitable for training machine learning models.

**Workflow Summary:**
1. **Index Stabilization** – Ensures data is grouped and sorted by `record_id`.  
2. **Time-Step Generation** – Adds `seq_id` columns for sequence modeling.  
3. **Log Transformations** – Applies `log(1+x)` normalization to skewed features.  
4. **Feature Selection** – Removes redundant, highly correlated variables.  
5. **Persist & Save** – Exports the final model-ready dataset.

**Output:**  
`data/processed/magnetic_data/` — the dataset consumed by training scripts.

---

## Pipeline Design Principles

- **Separation of Concerns:**  
  - Loaders → Download  
  - Preprocessors → Clean  
  - Processor → Feature engineer  
- **No Data Leakage:** Scaling and encoding are deferred to the modeling phase.  
- **Scalable:** Uses Dask for handling multi-GB data efficiently.  
- **Traceable:** Each transformation outputs reproducible and auditable data.

---

## Expected Data Volume

| Dataset | Approx. Size | Notes |
|----------|---------------|-------|
| 2020 Magnetic Field Data | **17 GB (JSON)** → **~5 GB (Parquet)** | Four raw JSON partitions. |
| SDOBenchmark Image Dataset | **2.8 GB × 4 = ~11.2 GB** | Four duplicate copies due to case-sensitive folder naming. |
| Stealth Dataset *(optional)* | **≈1–2 GB** | Supplemental magnetic field data. |

---

## Output Structure

```

data/  
├── raw/  
│ ├── solar_flare_2020/  
│ ├── sdo/  
│ └── stealth/  
├── preprocessed/  
│ ├── solar_flare_2020/  
│ └── sdo/   
└── processed/  
	└── magnetic_data/
		├── train/
		└── test/
```

---

**Author:** - [Zen_Nightshade](https://github.com/Zen-Nightshade)
**See Also:** `docs/Data_Processing_Report.md`