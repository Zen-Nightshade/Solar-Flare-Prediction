# Data Processing, Feature Engineering and Data Augmentation Report

## 1. Executive Summary

This report details the complete data pipeline designed for the **Solar Flare Prediction Model**, covering **preprocessing, data integrity correction, augmentation, and feature engineering**. The pipeline operates on the **magnetic parameter time-series data** from the **IEEE Big Data Cup 2020**.

The initial dataset, comprising **≈17 GB** across three training partitions, presented critical structural and statistical challenges that required a multi-stage solution.

*   **Data Integrity Crisis:** Exploratory analysis and initial processing runs revealed a severe data integrity issue: **over 40,000 `record_id`s were duplicated across the source partition files**. These were not simple duplicates but distinct time-series observations incorrectly assigned the same ID, causing records to have 120 time steps instead of the expected 60. This issue severely impacted over 35% of the rare 'X' and 'M' class flares.
*   **Severe Class Imbalance:** The dataset is dominated by 'Flare Quiet' (Q) samples, with the critical 'X' class flares representing less than 0.2% of the data.
*   **Statistical Challenges:** Key flux and energy parameters (`USFLUX`, `TOTPOT`) exhibited severe right-skew and strong multicollinearity.

The final pipeline, implemented with **Dask** for memory-efficient processing, systematically addresses these challenges. It first corrects the `record_id` collision bug, then applies a sophisticated augmentation strategy to rebalance the classes, and finally engineers a **14-feature refined dataset**, ready for downstream modeling.

---

## 2. Dataset Overview

### 2.1 Magnetic Field Data (IEEE Big Data Cup 2020)

The dataset consists of four large JSON files:

- `train_partition1_data.json`  
- `train_partition2_data.json`  
- `train_partition3_data.json`  
- `test_4_5_data.json`

Collectively, these amount to **≈17 GB** of time-series magnetic parameter data.  
Each record corresponds to a unique solar active region observed by the **SDO/HMI instrument**, with associated physical parameters and a labeled flare class. Note for this project I have not used `test_4_5_data.json` as it doesn't have any labels that i can use to test against.

**Source:** [IEEE Big Data Cup 2020 – Solar Flare Forecasting Data](https://dmlab.cs.gsu.edu/solar/data/data-comp-2020/)  
**Kaggle Mirror:** [Stealth Technologies – Solar Flares Dataset](https://www.kaggle.com/datasets/stealthtechnologies/solar-flares-dataset)

### 2.2 SDO/HMI Image Dataset

The image dataset complements the magnetic data with raw solar imagery used for visual pattern extraction.  
It can be obtained from:

- [SDO Benchmark (FHNW I4DS)](https://www.kaggle.com/datasets/fhnw-i4ds/sdobenchmark)

**Important Note on Download Duplication:**

When downloading the SDO image dataset, the pipeline retrieves the same files **four times**, due to case variations in filenames (e.g., `image_001.FITS`, `IMAGE_001.FITS`, etc.).  
As a result, users should **expect ~2.8 GB × 4 = ≈11.2 GB** of image data after full download.

### 2.3 Loader and Preprocessing Scripts

* **`loaders/`**  
  - Responsible for **downloading and verifying** the magnetic and SDO datasets.  
  - Handles network retries, file checksums, and extraction.  

* **`sdo_preprocess.py`**  
  - Generates a **metadata index** for the image dataset, linking each image to its corresponding magnetic record (`record_id`) and flare label.  
  - Produces structured metadata files used for efficient downstream model loading and alignment between image and magnetic data.


---

### 3. Data Integrity Investigation and Correction

Initial attempts to augment the data using `magnetic_augmentor.py` failed with shape mismatch errors (`ValueError: operands could not be broadcast together...`). This indicated that some records had 120 time steps instead of the expected 60, triggering a critical debugging and data validation phase.

### 3.1 The Debugging Journey

1.  **Problem Identification:** The `magnetic_augmentor.py` script, which groups data by `record_id` to create synthetic samples, was the first to crash. This proved that the data integrity issue existed *before* any augmentation was attempted.
2.  **Diagnostic Analysis:** A series of diagnostic scripts were developed to validate the preprocessed data:
    *   `check_magnetic_record_counts.py`: Confirmed that **41,680 `record_id`s** in the preprocessed data had exactly 120 rows.
    *   `check_corrupted_distribution.py`: Revealed that the corruption severely impacted the rare 'X' and 'M' classes. For instance, **87 'X'-class flares**—over a third of the total—were part of this corrupted set, making simple deletion an unacceptable option.
3.  **Root Cause Discovery:** Analysis of the source JSON files revealed two key facts:
    *   Any single line in a JSON file correctly contained exactly 60 time steps.
    *   The `record_id` ranges across the training partition files **were not disjoint**. Specifically:
        *   `partition1`: 1 to 77270
        *   `partition2`: 77271 to 171037
        *   `partition3`: **93768 to 136753** (This range significantly overlaps with `partition2`).
    This confirmed that the `preprocess_2020.py` script, by processing each file independently, was correctly creating 60-row entries for the same `record_id` from different files. These were then combined by downstream Dask operations, leading to the 120-row corrupted records.

### 3.2 The Solution: "Rename on Collision"

The `preprocess_2020.py` script was fundamentally rewritten to solve this problem at the source. It now implements a **stateful "Rename on Collision"** strategy:
1.  The script maintains a global set of all `record_id`s it has processed.
2.  The first time an ID is encountered (e.g., from `partition1` or `partition2`), it is used as-is.
3.  On any subsequent encounter of the same ID (e.g., from `partition3`), the script assumes it is a new, independent observation with a logging error. It assigns a **new, unique, sequential integer ID** (starting from 171038) to preserve the data while guaranteeing uniqueness.

This correction ensures the preprocessed dataset is clean, with every record having a unique ID and exactly 60 time steps, forming a reliable foundation for all future steps.

---

## 4. Data Augmentation for Class Imbalance

With a clean dataset, the severe class imbalance was addressed using the `magnetic_augmentor.py` script. A multi-step strategy was applied *only to the training set* before feature engineering.

1.  **Synthetic Data Generation:** For the rarest classes ('X', 'M'), new, physically plausible samples were generated using:
    *   **Magnitude Warping:** Multiplying the time series by a smooth, random curve to simulate variations in energy evolution.
    *   **Scaling:** Multiplying the entire time series by a random scalar to simulate slightly stronger or weaker events.
2.  **Naive Oversampling:** For minority classes ('X', 'B', 'C'), a fraction of the original samples were duplicated to increase their weight.
3.  **Naive Undersampling:** For the majority 'Flare Quiet' (Q) class, a large portion of samples were randomly removed to reduce their dominance and decrease training time.

This strategy drastically improved the class balance, reducing the ratio of the most common to rarest class from over **450-to-1** to approximately **9-to-1**, enabling the model to learn meaningful patterns from the minority classes.

---

## 5. Feature Engineering Pipeline (`process_magnetic_data.py`)

The feature engineering pipeline operates on the **clean, augmented, and balanced** training data. It follows a modular, multi-phase approach:

### **Phase 1 — Data Structuring**
1.  **Data Splitting:** The augmented training data is first split by `record_id` into `train` (98%), `dev` (1%), and `test` (1%) sets for local validation.
2.  **Sorting:** Data is sorted by `record_id` and the pre-existing `seq_id` to ensure correct temporal order for grouped operations.

### **Phase 2 — Foundational Transformations**
- **Logarithmic Transformations** (`log(1+x)`) are applied to right-skewed variables such as `TOTPOT` and `USFLUX`.

### **Phase 3 — Advanced Time-Series Feature Engineering**
Group-wise operations on each `record_id` yield:
- **Rolling Features:** 5-step moving averages and standard deviations capturing local trends and volatility.  
- **Difference Features:** 1-step temporal derivatives measuring immediate rate-of-change.  
- **Lag Features:** 3-step lags introducing short-term temporal memory.

### **Phase 4 — Feature Selection and Cleanup**
- Redundant features removed using correlation thresholds and domain-driven selection.  
- Missing values introduced by rolling/lag operations filled with zero.

---

## 6. Final Feature Set

| Original Column(s) | Final Feature | Status | Description |
|--------------------|---------------|---------|-------------|
| `record_id` | `record_id` | Kept | Unique time-series group identifier |
| `label` | `label` | Kept | Flare class (C, M, X, etc.) |
| `USFLUX` | `USFLUX_log` | Transformed | Log-scaled magnetic flux |
| `TOTPOT` | `TOTPOT_log` | Transformed | Log-scaled magnetic energy |
| `PIL_LEN` | `PIL_LEN_log` | Transformed | Log-scaled polarity inversion line length |
| `MEANSHR` | `MEANSHR` | Kept | Magnetic shear magnitude |
| `TOTFZ` | `TOTFZ` | Kept | Magnetic flux imbalance |
| `EPSZ` | `EPSZ` | Kept | Normalized flux imbalance |
| `R_VALUE` | `R_VALUE` | Kept | Zero-inflated rare-activity indicator |
| Grouped by `record_id` | `seq_id` | Created | Temporal step index |
| `USFLUX_log` | `USFLUX_log_roll_mean5` | Created | 5-step rolling mean (trend) |
| `USFLUX_log` | `USFLUX_log_roll_std5` | Created | 5-step rolling standard deviation (volatility) |
| `TOTPOT_log` | `TOTPOT_log_diff1` | Created | 1-step difference (instantaneous change) |
| `MEANSHR` | `MEANSHR_lag3` | Created | 3-step lag (temporal memory) |

All other 28 raw parameters were excluded to minimize redundancy and ensure interpretability.

---

## 7. Deferred Operations: Scaling and Encoding
Two key preprocessing steps — **Feature Scaling** and **Categorical Encoding** — are **intentionally deferred** to the modeling phase (`train.py`).  
This prevents **data leakage**, ensuring the model’s performance metrics remain valid.  
Scaling and encoding must be **fitted on training data only** and reused consistently across validation and test splits.

---

## 8. Storage and Split Strategy

The processed data is stored under:
```
data/processed/magnetic_data/
```
with partitions:
- `train`
- `dev`
- `test`

Each partition is independently processed to preserve isolation between data subsets.

---

## 9. Conclusion

The data processing pipeline successfully transforms a raw dataset with **critical integrity issues and severe class imbalance** into a compact, interpretable, and model-ready format. The debugging and correction of the source data's `record_id` collision bug was a critical, non-trivial step that salvaged over 35% of the rare flare samples. The subsequent augmentation strategy further enhanced the dataset, enabling robust model training.

The final result is a 14-feature dataset enriched with meaningful temporal descriptors, forming a trustworthy foundation for solar flare prediction.

The loader and SDO preprocessing components extend this ecosystem by integrating metadata and imagery, enabling multimodal analysis.  
Researchers should **prepare for approximately 17 GB (magnetic data) + 11.2 GB (image data)** during full replication of the workflow.

This modular and scalable design ensures reproducibility, computational efficiency, and scientific rigor for both time-series and image-based solar flare modeling.