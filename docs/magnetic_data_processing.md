# Data Processing and Feature Engineering Report

## 1. Executive Summary

This report details the complete preprocessing and feature engineering pipeline designed for the **Solar Flare Prediction Model**.  
The pipeline operates on two primary datasets — the **magnetic parameter time-series data** from the **IEEE Big Data Cup 2020** and the **SDO/HMI image dataset** from complementary Kaggle sources.

The magnetic dataset comprises **≈95 Parquet files (~5 GB)**, each containing solar active region (AR) magnetic field measurements over time, while the SDO image dataset provides multi-channel images of the solar surface. Together, these datasets form the basis for building predictive models for solar flare activity.

Exploratory Data Analysis (EDA) revealed critical structural and statistical issues in the raw magnetic data:

* **Structural Challenges:**  
  - Non-unique DataFrame indices and a **panel (multi-sequence) structure**, where each `record_id` represents an independent time-series sequence.  
  - Missing or inconsistent temporal ordering within certain groups.  

* **Statistical Challenges:**  
  - Severe **right-skew** in key flux and energy parameters (`USFLUX`, `TOTPOT`, etc.).  
  - Strong **multicollinearity**, with correlation coefficients > 0.9 among clusters of related variables.

The preprocessing pipeline — implemented with **Dask** for distributed and memory-efficient processing — addresses these challenges systematically and produces a **14-feature refined dataset**, ready for downstream modeling.

---

## 2. Dataset Overview

### 2.1 Magnetic Field Data (IEEE Big Data Cup 2020)

The dataset consists of four large JSON files:

- `train_partition1_data.json`  
- `train_partition2_data.json`  
- `train_partition3_data.json`  
- `test_4_5_data.json`

Collectively, these amount to **≈17 GB** of time-series magnetic parameter data.  
Each record corresponds to a unique solar active region observed by the **SDO/HMI instrument**, with associated physical parameters and a labeled flare class.

**Source:** [IEEE Big Data Cup 2020 – Solar Flare Forecasting Data](https://dmlab.cs.gsu.edu/solar/data/data-comp-2020/)  
**Kaggle Mirror:** [Stealth Technologies – Solar Flares Dataset](https://www.kaggle.com/datasets/stealthtechnologies/solar-flares-dataset)

---

### 2.2 SDO/HMI Image Dataset

The image dataset complements the magnetic data with raw solar imagery used for visual pattern extraction.  
It can be obtained from:

- [SDO Benchmark (FHNW I4DS)](https://www.kaggle.com/datasets/fhnw-i4ds/sdobenchmark)

**Important Note on Download Duplication:**

When downloading the SDO image dataset, the pipeline retrieves the same files **four times**, due to case variations in filenames (e.g., `image_001.FITS`, `IMAGE_001.FITS`, etc.).  
As a result, users should **expect ~2.8 GB × 4 = ≈11.2 GB** of image data after full download.

---

### 2.3 Loader and Preprocessing Scripts

* **`loaders/`**  
  - Responsible for **downloading and verifying** the magnetic and SDO datasets.  
  - Handles network retries, file checksums, and extraction.  

* **`sdo_preprocess.py`**  
  - Generates a **metadata index** for the image dataset, linking each image to its corresponding magnetic record (`record_id`) and flare label.  
  - Produces structured metadata files used for efficient downstream model loading and alignment between image and magnetic data.

---

## 3. Exploratory Data Analysis (EDA)

EDA was conducted using a subset of the magnetic dataset and visualized via correlation matrices and feature distributions.

- **Raw Correlation Matrix:**  
 ![Raw Correlation Matrix](https://raw.githubusercontent.com/Zen-Nightshade/Solar-Flare-Prediction/main/figures/magnetic_data_EDA/raw_plots/correlation_matrix.png)


  The raw data exhibits heavy feature redundancy, with large clusters of features being highly correlated (e.g., `TOTUSJH`, `TOTBSQ`, and `USFLUX`).

- **Processed Correlation Matrix:**  
  ![Processed Correlation Matrix](https://raw.githubusercontent.com/Zen-Nightshade/Solar-Flare-Prediction/main/figures/magnetic_data_EDA/transformed_plots/correlation_matrix.png)

  After transformation and selection, multicollinearity is substantially reduced, resulting in a more orthogonal feature space suitable for modeling.

Additional figures and distribution plots (available in [this directory](https://github.com/Zen-Nightshade/Solar-Flare-Prediction/blob/main/figures/magnetic_data_EDA/)) further illustrate data normalization and feature distribution changes after processing.

---

## 4. Processing Pipeline Overview

Implemented in `process_magnetic_data.py`, the preprocessing pipeline follows a modular, multi-phase approach:

### **Phase 1 — Data Stabilization and Structuring**
1. **Index Resetting** to remove duplicate indices.  
2. **Sorting by `record_id`** to preserve time continuity.  
3. **Sequence ID Creation (`seq_id`)** to explicitly index time steps within each sequence.

### **Phase 2 — Foundational Transformations**
- **Logarithmic Transformations** (`log(1+x)`) applied to right-skewed variables such as `TOTPOT`, `USFLUX`, and `PIL_LEN`.

### **Phase 3 — Advanced Time-Series Feature Engineering**
Group-wise operations on each `record_id` yield:
- **Rolling Features:** 5-step moving averages and standard deviations capturing local trends and volatility.  
- **Difference Features:** 1-step temporal derivatives measuring immediate rate-of-change.  
- **Lag Features:** 3-step lags introducing short-term temporal memory.

### **Phase 4 — Feature Selection and Cleanup**
- Redundant features removed using correlation thresholds and domain-driven selection.  
- Missing values introduced by rolling/lag operations filled with zero.

---

## 5. Final Feature Set

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

## 6. Deferred Operations: Scaling and Encoding

Two key preprocessing steps — **Feature Scaling** and **Categorical Encoding** — are **intentionally deferred** to the modeling phase (`train.py`).  
This prevents **data leakage**, ensuring the model’s performance metrics remain valid.  
Scaling and encoding must be **fitted on training data only** and reused consistently across validation and test splits.

---

## 7. Storage and Split Strategy

The processed data is stored under:
```
data/processed/magnetic_data/
```
with partitions:
- `train`
- `dev`
- `test`
- `holdout_test`

Each partition is independently processed to preserve isolation between data subsets.

---

## 8. Conclusion

The magnetic data processing pipeline successfully transforms complex, high-dimensional, and correlated raw solar data into a **compact, interpretable, and model-ready dataset**.  
The result is a 14-feature dataset enriched with meaningful temporal descriptors — trend, volatility, rate-of-change, and short-term memory — forming a robust foundation for solar flare prediction.

The loader and SDO preprocessing components extend this ecosystem by integrating metadata and imagery, enabling multimodal analysis.  
Researchers should **prepare for approximately 17 GB (magnetic data) + 11.2 GB (image data)** during full replication of the workflow.

This modular and scalable design ensures reproducibility, computational efficiency, and scientific rigor for both time-series and image-based solar flare modeling.