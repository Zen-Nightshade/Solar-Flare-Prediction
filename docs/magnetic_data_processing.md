## Data Processing Report for Solar Flare Prediction Model

### 1. Executive Summary

This report presents the complete preprocessing pipeline developed for the **Solar Flare Prediction Model**.  
The pipeline was applied to the raw solar magnetic field dataset, consisting of **35 features across 95 Parquet files (≈5 GB total: train + test)**.  
Exploratory Data Analysis (EDA) identified several structural and statistical challenges: a **non-unique index**, **panel data organization**, **highly skewed feature distributions**, and **strong multicollinearity** among variables.

The goal of this preprocessing stage is to convert the raw input into a **clean, feature-engineered, and model-ready dataset** suitable for machine learning applications.  
The process, implemented using **Dask** for scalability, draws directly from EDA insights. The final output is a compact, 10-column dataset optimized for the modeling phase.

---

### 2. Processing Pipeline: Methodology and Justification

All transformations were carried out by the `process_magnetic_data.py` script on both the training and testing datasets.  
The pipeline is organized into two major phases: **Data Stabilization and Structuring**, followed by **Feature Transformation and Selection**.

#### **Phase 1: Data Stabilization and Structuring**

These steps ensure data consistency, integrity, and computational efficiency across distributed processing.

- **Step 1.1 — Index Resetting**  
  The dataset was reindexed to remove duplicate labels caused by merging multiple Parquet files. This resolved issues with non-unique indices that previously led to computation errors, ensuring a stable DataFrame structure.

- **Step 1.2 — Sorting by Group Identifier**  
  The data was sorted by `record_id`, where each `record_id` represents a unique time-series segment. Sorting improved performance by minimizing data shuffling during group operations, a crucial optimization for distributed computation.

- **Step 1.3 — Sequence ID Creation (`seq_id`)**  
  A new column `seq_id` was added to assign a sequential time-step index (0, 1, 2, …) to each measurement within its respective `record_id`.  
  This transformation explicitly encodes the temporal order necessary for time-series analysis and feature engineering.

---

#### **Phase 2: Feature Transformation and Selection**

This phase modifies and refines features based on their statistical characteristics and relationships identified during EDA.

- **Step 2.1 — Logarithmic Transformation of Skewed Features**  
  Several features exhibited strong right-skewed distributions. Applying a logarithmic transformation reduced skewness and stabilized variance, improving the behavior of downstream models that assume normally distributed inputs.

- **Step 2.2 — Representative Feature Selection**  
  High correlations (often > 0.9) were observed among many features, forming distinct multicollinearity clusters.  
  To avoid redundancy and improve model stability, representative features were selected from each cluster, resulting in a focused and interpretable subset of 10 columns.

---

### 3. Feature Decisions Summary

| Original Column | Final Status | Justification |
|-----------------|--------------|----------------|
| `record_id` | **Kept** | Serves as the unique sequence identifier for each time-series instance. |
| `label` | **Kept** | Retained as the target variable for classification. |
| `USFLUX` | **Transformed & Kept** as `USFLUX_log` | Core measure of magnetic flux; log-transformed due to high skew. |
| `TOTPOT` | **Transformed & Kept** as `TOTPOT_log` | Represents total magnetic energy; log-transformed and selected as the second key representative of the energy cluster. |
| `PIL_LEN` | **Transformed & Kept** as `PIL_LEN_log` | Captures the polarity inversion line length; physically distinct and log-transformed. |
| `MEANSHR` | **Kept** | Represents magnetic shear and field complexity; exhibits moderate distribution and strong interpretability. |
| `TOTFZ` | **Kept** | Left-skewed feature providing complementary distributional information. |
| `EPSZ` | **Kept** | Normalized parameter capturing unique information not present in other features. |
| `R_VALUE` | **Kept** | Zero-inflated feature; retained due to potential predictive value of rare non-zero events. |
| `seq_id` | **Created & Kept** | Explicitly encodes temporal ordering within each time-series group. |

All other features were excluded based on redundancy, low variance, or high correlation with selected representatives.  
Certain features such as `TOTUSJH`, `TOTBSQ`, and `ABSNJZH` were removed due to excessive multicollinearity, while others such as `XR_MAX` were excluded to prevent potential **label leakage**, ensuring the model learns only from pre-flare magnetic conditions.

---

### 4. Addendum: Scaling and Encoding Policy

#### **Purpose**

Two standard preprocessing steps — **Feature Scaling** and **Categorical Encoding** — were **intentionally excluded** from the `process_magnetic_data.py` pipeline.  
These operations are performed during the **modeling phase**, not during generic data preparation, to avoid **data leakage** that could compromise model integrity.

---

#### **4.1 Feature Scaling**

Scaling involves learning parameters such as the mean and standard deviation from the data.  
If performed before the train-test split, scaling would incorporate information from the test set, allowing the model to “see” data it should not access during training.  
To prevent this, scaling must occur **only after** splitting the dataset, using statistics derived **solely from the training set**.  
This ensures fair evaluation and generalization to unseen data.

---

#### **4.2 Categorical Encoding**

The target labels (e.g., flare classes ‘C’, ‘M’, ‘X’) must be numerically encoded for machine learning algorithms.  
However, the mapping from string labels to numeric codes must be learned **only from the training labels** to guarantee consistent encoding and decoding.  
Performing encoding prior to splitting could introduce unseen class information from the test set, leading to inconsistent mappings and data leakage.

---

### 5. Conclusion

The `process_magnetic_data.py` pipeline delivers a **robust, consistent, and model-ready dataset** with 10 statistically balanced and physically meaningful features.  
The pipeline ensures:
- Stable indexing and structured time-series organization  
- Controlled feature transformation and pruning  
- Prevention of redundancy and multicollinearity  
- Strict separation between preprocessing and modeling stages  

By deferring **scaling** and **encoding** to the modeling phase, the workflow adheres to best practices in machine learning, ensuring **data integrity**, **reproducibility**, and **reliable model performance**.

The resulting dataset is now fully prepared for the next stage: training and evaluation of the Solar Flare Prediction Model.