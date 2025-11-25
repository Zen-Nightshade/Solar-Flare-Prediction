# Solar Flare Prediction using Attention-Based Deep Learning

This repository contains the complete implementation of a **Solar Flare Prediction Framework** using large-scale magnetic field time-series data from the *IEEE Big Data 2020 Solar Flare Dataset*.

The project moves beyond standard baselines to implement **Bi-Directional LSTMs integrated with Attention Mechanisms** (Dot-Product, Concatenation, and Multi-Head). It features a robust **Dask-based Data Engineering pipeline** capable of handling ~17 GB of data, correcting critical integrity issues, and mitigating extreme class imbalance through synthetic augmentation.

---

## Project Overview

**Goal:** Predict the occurrence and intensity class (Q, B, C, M, X) of solar flares within a 24-hour forecast window.

**Key Features:**
*   **Big Data Handling:** Utilized `Dask` and `Parquet` to process 17 GB of JSON data on limited RAM.
*   **Data Integrity Fix:** Identified and resolved a critical "Duplicate Record ID" bug that corrupted >35% of minority class samples.
*   **Advanced Augmentation:** Implemented **Magnitude Warping** and **Scaling** to synthetically generate physically plausible X and M-class flares.
*   **Deep Learning:** Comparative analysis of RNN vs. LSTM, Uni- vs. Bi-directional architectures, and various Attention mechanisms.

> **Note on Scope:** While SDO/HMI images were initially acquired, this project focuses exclusively on the **Time-Series Magnetic Parameters** due to a lack of absolute timestamps in the magnetic dataset required for temporal alignment.

---

## Data Processing Pipeline

The pipeline is engineered for scalability and data quality.

### 1. Ingestion & Optimization (`dask`)
*   **Conversion:** Transformed raw JSONL streams into optimized **Parquet** shards.
*   **Optimization:** Reduced memory footprint and accelerated I/O operations by ~4x compared to standard Pandas.

### 2. Data Cleaning & Integrity
*   **Collision Fix:** Implemented a "Rename-on-Collision" strategy to resolve duplicate `record_id`s across disjoint partition files.
*   **Restoration:** Recovered ~40,000 time-series sequences that were previously merged incorrectly.

### 3. Feature Engineering
*   **Transformations:** Applied `log(1+x)` to heavy-tailed features (`USFLUX`, `TOTPOT`).
*   **Temporal Features:** Generated Rolling Means (Trend), Standard Deviations (Volatility), and Lag/Difference features to capture magnetic flux evolution.
*   **Selection:** Reduced dimensionality to 14 core features based on Correlation Matrix analysis.

### 4. Data Augmentation
Addressed the severe **450:1 Class Imbalance** (Quiet vs. X-Class):
*   **Minority Classes (X, M):** Generated synthetic samples using **Magnitude Warping** (simulating non-linear flux changes) and **Random Scaling**.
*   **Majority Class (Q):** Preserved or Undersampled depending on the experiment configuration.
*   **Result:** Reduced imbalance ratio to **9:1**.

---

## Model Architectures

We conducted an extensive ablation study comparing Baseline ML models against Deep Learning architectures.

### 1. Baselines
*   **Logistic / Softmax Regression**
*   **Random Forest Classifier** (Best Baseline)
*   **Support Vector Machines (SVM)** *(Failed to converge due to dataset size)*

### 2. Deep Learning Models
*   **Recurrent Units:** Comparison of Simple RNN vs. LSTM.
*   **Directionality:** Uni-directional vs. Bi-directional (Bi-LSTM).
*   **Attention Mechanisms:**
    *   **Dot-Product Attention:** Measures global similarity between hidden states.
    *   **Concatenation (Additive) Attention:** Learns alignment via a feed-forward network.
    *   **Multi-Head Attention:** Adapted from Transformers to capture diverse temporal dependencies.

---

## Experimental Results

Primary evaluation metrics were **Matthews Correlation Coefficient (MCC)** and **F1-Macro** due to class imbalance.

| Model Architecture | Attention Type | MCC Score | Status |
|:--|:--|:--|:--|
| **Bi-LSTM** | **Dot-Product** | **0.708** | **Best Model** |
| Bi-LSTM | Concatenation | 0.663 | Runner Up |
| Multi-Head Bi-LSTM | Self-Attention | 0.643 | Complex / Overfit |
| Random Forest | N/A | 0.619 | Baseline |
| Simple RNN | None | 0.402 | Poor |

**Key Finding:** The **Bi-Directional LSTM with Dot-Product Attention** effectively isolates critical precursor signals (e.g., sharp flux changes) from the background noise, significantly outperforming baselines.

---

## Challenges Faced

1.  **Memory Constraints:** The 17 GB dataset required `Dask` for out-of-core processing; standard Pandas workflows crashed immediately.
2.  **Data Integrity:** Discovering that 40,000 records had 120 time-steps (instead of 60) due to ID collisions was a critical debugging milestone.
3.  **Extreme Imbalance:** The raw dataset had only ~380 X-class flares vs. ~175,000 Quiet samples. Augmentation was mandatory.
4.  **Temporal Misalignment:** The inability to map specific magnetic time-steps to absolute dates prevented the integration of the SDO Image dataset.

---

## Future Work

*   **Multimodal Alignment:** Recover timestamps to fuse SDO Images (CNN) with Magnetic Data (LSTM).
*   **Transformers:** Implement pure Time-Series Transformer models (e.g., Informer, Autoformer).
*   **Operational Deployment:** Optimize the best model for real-time inference on streaming solar data.

---

## Author

[**Zen Nightshade**](https://github.com/Zen-Nightshade)