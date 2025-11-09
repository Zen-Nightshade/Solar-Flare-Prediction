# Solar Flare Data Processing Pipeline

This directory contains all scripts used to **download, preprocess, augment, and transform** the solar flare datasets into a structured, model-ready format.
The pipeline integrates both **magnetic field time-series data** and **SDO image data**, aligning them for multimodal flare prediction tasks.

---

## Overview

The pipeline manages the complete data preparation lifecycle — from raw downloads to engineered features.

**Scripts in this directory:**
```
# Data Acquisition
loader_2020.py
loader_stealth.py
loader_sdo.py

# Initial Conversion & Cleaning
preprocess_2020.py
preprocess_sdo.py

# Feature Engineering
process_magnetic_data.py

# Augmentation & Balancing
magnetic_augmentor.py

# Diagnostic & Utility Scripts
check_corrupted_distribution.py
check_magnetic_record_counts.py
temp.py
```

---

## Workflow Summary

1.  **Download Phase (Loaders)**
    Downloads raw datasets and organizes them under `data/raw/`.
    - `loader_2020.py`: Downloads the 2020 magnetic field dataset.
    - `loader_stealth.py`: Downloads additional stealth event data.
    - `loader_sdo.py`: Downloads the SDO image dataset.

2.  **Preprocessing Phase (Initial Conversion)**
    Converts raw source files into a clean, Dask-compatible Parquet format.
    - `preprocess_2020.py`: Processes raw magnetic field JSON data, **resolving record_id collisions by renumbering duplicates.**
    - `preprocess_sdo.py`: Builds structured metadata for SDO image data.

3.  **Feature Engineering Phase**
    - `process_magnetic_data.py`: Performs log transforms, time indexing, feature selection, and **advanced time-series feature engineering** to create the final model-ready dataset.

4.  **Data Augmentation Phase (for Magnetic Data)**
    Addresses severe class imbalance in the training set.
    - `magnetic_augmentor.py`: Applies a multi-step strategy of **synthetic data generation** (scaling, warping), **oversampling** (duplication), and **undersampling** to create a balanced training dataset.

---

## Dataset Sources

| Dataset | Description | Source |
| :--- | :--- | :--- |
| **Solar Flare Prediction from Time Series Data (2020)** | Time-series magnetic field parameters from SDO/HMI vector magnetograms. | [GSU Dataset (IEEE Big Data Cup 2020)](https://dmlab.cs.gsu.edu/solar/data/data-comp-2020/) |
| **SDOBenchmark** | Image-sequence dataset of active solar regions for flare prediction. | [Kaggle – SDOBenchmark](https://www.kaggle.com/datasets/fhnw-i4ds/sdobenchmark) |
| **Stealth Dataset** *(optional)* | Supplemental magnetic field observations for stealth CMEs. | [Kaggle – Stealth Technologies](https://www.kaggle.com/datasets/stealthtechnologies/solar-flares-dataset) |

---

## Download Scripts

### `loader_2020.py`, `loader_stealth.py`, `loader_sdo.py`

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
Converts the raw magnetic field JSON data into efficient, structured Parquet files, **while correcting a critical data integrity issue in the source data.**

**Input:**
Four JSON files from the 2020 dataset (`train_partition1_data.json`, etc.).

**Process:**
1.  Reads each JSON file line-by-line.
2.  **Handles `record_id` Collisions:** The source data contains overlapping `record_id`s across the training partition files. This script identifies these collisions.
    - The first time a `record_id` is encountered, it is processed and saved normally.
    - On subsequent encounters, the script assumes it's a new, independent observation with an incorrect ID. It assigns a **new, unique, sequential integer ID** (starting from 171038) to preserve the data without creating duplicates.
3.  **Validates and Cleans:** Each record is validated to ensure it contains exactly 60 time steps.
4.  **Adds Identifiers:** Creates a globally unique `safe_index`.
5.  Saves results as **chunked Parquet files**.

**Output:**
```
data/preprocessed/train/
data/preprocessed/test/
```
The output is a clean dataset where every `record_id` is guaranteed to be unique.

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
Transforms the augmented magnetic field data into a final, feature-rich dataset for model training. Optimized for memory efficiency.

**Workflow Summary:**
1.  **Data Splitting:** The augmented training data is split by `record_id` into `train`, `dev`, and `test` sets (e.g., 98%/1%/1%).
2.  **Sorting & Indexing:** Data is sorted by `record_id` and `seq_id` to prepare for time-series operations. The `seq_id` column is relied upon for temporal ordering.
3.  **Log Transformations:** Applies `log(1+x)` normalization to highly skewed features.
4.  **Advanced Time-Series Feature Engineering:** Calculates:
    *   **Rolling Means/Stdevs:** (`USFLUX_log_roll_mean5`, etc.) to capture trends.
    *   **Rate of Change:** (`TOTPOT_log_diff1`, etc.) to measure energy changes.
    *   **Lag Features:** (`MEANSHR_lag3`, etc.) to provide short-term memory.
5.  **Feature Selection & Saving:** Prunes the dataset to a curated list of features and saves the final `train`, `dev`, `test`, and `holdout_test` sets.

**Output:**
Final, model-ready datasets located in:
```
data/processed/magnetic_data/
├── train/
├── dev/
├── test/
└── holdout_test/
```

---

## Data Augmentation Script

### `magnetic_augmentor.py`

**Purpose:**
Takes the cleaned, preprocessed training data and applies a sophisticated augmentation and resampling strategy to combat severe class imbalance before feature engineering.

**Workflow:**
1.  **Calculates Distribution:** Determines the initial counts of each flare class (X, M, B, C, Q).
2.  **Generates Synthetic Data:** For the rarest classes ('X', 'M'), it creates new, physically plausible time-series samples using:
    - **Magnitude Warping:** Distorts the time series with a smooth, random curve.
    - **Scaling:** Multiplies the entire time series by a random scalar.
3.  **Oversamples:** For minority classes ('X', 'B', 'C'), it duplicates a fraction of the original samples to increase their weight.
4.  **Undersamples:** For the majority class ('Q'), it randomly removes a large portion of samples to reduce its dominance.
5.  **Replaces Data:** The script saves the new, balanced dataset to a temporary location and then replaces the original `preprocessed/train` directory.

**Output:**
The original `data/preprocessed/train/` directory is overwritten with a larger, more balanced dataset containing a mix of original, synthetic, and duplicated samples.


---

## Pipeline Design Principles

- **Separation of Concerns:**
  - Loaders → Download
  - Preprocessors → Clean
  - Processor → Feature engineer
- **No Data Leakage:** The train/dev/test/holdout sets are all processed independently. Scaling and encoding are deferred to the modeling phase.
- **Scalable & Memory-Aware:** Uses Dask for handling multi-GB data. The `process_magnetic_data.py` script is specifically optimized with repartitioning, sequential processing of splits, and aggressive garbage collection to run on memory-constrained systems.
- **Traceable:** Each transformation outputs reproducible and auditable data.

---

## Expected Data Volume

| Dataset                      | Approx. Size                           | Notes                                                      |
| :--------------------------- | :------------------------------------- | :--------------------------------------------------------- |
| 2020 Magnetic Field Data     | **17 GB (JSON)** → **~5 GB (Parquet)** | Four raw JSON partitions.                                  |
| SDOBenchmark Image Dataset   | **2.8 GB × 4 = ~11.2 GB**              | Four duplicate copies due to case-sensitive folder naming. |
| Stealth Dataset *(optional)* | **≈36 KB**                             | Supplemental magnetic field data.                          |

---

## Output Structure

```
data/
├── raw/
│   ├── solar_flare_2020/
│   ├── sdo/
│   └── stealth/
├── preprocessed/
│   ├── solar_flare_2020/
│   │   ├── train/
│   │   └── test/
│   └── sdo/
└── processed/
    ├── magnetic_data/
    │   ├── train/
    │   ├── dev/
    │   ├── test/
    │   └── holdout_test/
    └── sdo/(yet to work on this)
```

---

**Author:** - [Zen_Nightshade](https://github.com/Zen-Nightshade)
**See Also:** `docs/Data_Processing_Report.md`