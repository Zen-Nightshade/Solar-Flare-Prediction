# Solar Flare Prediction

This repository contains the mid-stage work for a project on **solar flare prediction** using large-scale magnetic field data from the *IEEE Big Data 2020 Solar Flare Dataset*.  
The focus so far has been on **building a scalable data processing pipeline**, **feature engineering**, and **establishing baseline models** for flare classification.  
Initial metadata preparation for the **SDO image dataset** has also been completed as a foundation for future multimodal modeling.

---

## Project Description

The project aims to predict the **occurrence and class of solar flares** using time-series magnetic parameters derived from solar active regions.  
Due to the size and complexity of the dataset (~17 GB of JSON files), a **Dask-based distributed pipeline** was developed to enable efficient processing under limited hardware resources.

Currently, only the **magnetic dataset** is being modeled.  
The **SDO image dataset** has been explored and a **metadata file** has been created for future integration.

---

## Data Processing Pipeline

Implemented in `process_magnetic_data.py`.

**Main steps:**
1. **Data Conversion:** Raw JSON → Parquet format (compressed, columnar).
2. **Cleaning and Structuring:** Fixed indices, sorted temporally, added `seq_id` for sequence tracking.
3. **Feature Engineering:**  
   - Log transforms for skewed features (e.g., `USFLUX`, `TOTPOT`, `PIL_LEN`).  
   - Rolling mean and standard deviation (e.g., `_roll_mean5`, `_roll_std5`).  
   - Lag and difference features for temporal dynamics (e.g., `_lag3`, `_diff1`).
4. **Feature Selection:** Removed highly correlated and redundant features.
5. **Output:** Cleaned, memory-optimized Parquet files ready for modeling.

> The dataset still requires ~18 GB of RAM for smooth execution even with optimized Dask configuration.

---

## Modeling Progress

**Implemented Baseline Models:**

| Task | Models | Evaluation Metric |
|:--|:--|:--|
| Binary Classification | Logistic Regression, Random Forest, SVM | F1-score, TSS |
| Multiclass Classification | Softmax Regression, Random Forest, SVM | Macro F1-score |

**Notes:**
- Accuracy was avoided due to heavy class imbalance (many non-flare samples).  
- The current models serve as baselines for evaluating future deep learning architectures.

---

## Challenges Faced

- **Memory Constraints:** Dataset exceeds typical RAM limits; optimized through Dask and partitioned processing.  
- **Multicollinearity:** High feature correlation reduced using correlation matrix analysis.  
- **Data Imbalance:** Major skew toward non-flare events affects learning stability.  
- **Temporal Misalignment:** Big Data 2020 magnetic dataset lacks timestamps, while SDO images are time-indexed.

---

## Current Status

- Complete Dask-based preprocessing pipeline  
- Feature engineering and selection  
- Baseline modeling and evaluation  
- Metadata preparation for SDO dataset  
- Pending: Integration, deep learning, and multimodal exploration  

---

## Future Work

- Implement **temporal deep learning models** (e.g., LSTM, GRU) for magnetic data.  
- Develop **CNN-based models** for SDO imagery.  
- Explore **data alignment strategies** between magnetic and image datasets.  
- If data alignment is successful, then implement **fusion models** combining both modalities.  
- Conduct **hyperparameter tuning** and **cross-validation** for model robustness.  

---

## Author

[**Zen Nightshade**](https://github.com/Zen-Nightshade)
[https://github.com/Zen-Nightshade/Solar-Flare-Prediction](https://github.com/Zen-Nightshade/Solar-Flare-Prediction)