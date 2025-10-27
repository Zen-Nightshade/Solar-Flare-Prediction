# Rippository Layout

```
solar-flare-prediction/
│
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── config/
│   ├── config.yaml                # Main configuration file
│   ├── data_sources.yaml          # Dataset URLs, paths, schema info
│   ├── preprocessing.yaml         # Feature extraction parameters
│   ├── model_config.yaml          # Model hyperparameters
│   └── logging.yaml               # Logging setup
│
├── data/
│   ├── raw/
│   │   ├── solar_flare_prediction_2020/    # IEEE Big Data Cup dataset (time series)
│   │   ├── stealth_solar_flares/           # Kaggle Stealth dataset (tabular)
│   │   └── sdo_benchmark/                  # Image dataset (SDO Benchmark)
│   │
│   ├── interim/                 # Cleaned/intermediate data
│   ├── processed/               # Final merged/prepared datasets
│   ├── metadata/                # Schema files, label encodings, documentation
│   └── README.md
│
├── notebooks/
│   ├── 01_exploration_2020.ipynb        # EDA for IEEE 2020 dataset
│   ├── 02_exploration_stealth.ipynb     # EDA for Stealth dataset
│   ├── 03_exploration_sdo.ipynb         # EDA for SDO Benchmark images
│   ├── 04_preprocessing.ipynb           # Cleaning, normalization, and merging
│   ├── 05_baseline_models.ipynb         # Baseline experiments
│   ├── 06_deep_models.ipynb             # LSTM/CNN/Transformer experiments
│   ├── 07_multimodal_fusion.ipynb       # Combine time-series + image features
│   └── 08_results_analysis.ipynb        # Performance evaluation
│
├── src/
│   ├── __init__.py
│
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader_2020.py           # Load and parse the IEEE 2020 dataset
│   │   ├── loader_stealth.py        # Load Stealth Kaggle dataset
│   │   ├── loader_sdo.py            # Load image data from SDO Benchmark
│   │   ├── preprocess_time_series.py# Cleaning, resampling, and feature creation
│   │   ├── preprocess_images.py     # Image augmentation and resizing
│   │   ├── merge_datasets.py        # Merge datasets for multi-modal training
│   │   └── split_data.py            # Train/test/validation splits
│
│   ├── features/
│   │   ├── __init__.py
│   │   ├── extract_time_series_features.py   # Magnetic field, active region stats
│   │   ├── extract_tabular_features.py       # From Stealth dataset
│   │   ├── extract_image_features.py         # CNN feature extraction
│   │   └── feature_selector.py               # PCA, SHAP, or correlation pruning
│
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_models.py        # Logistic Regression, Random Forest, SVM
│   │   ├── deep_time_series.py       # LSTM, GRU models
│   │   ├── image_cnn.py              # CNN or EfficientNet for image inputs
│   │   ├── fusion_models.py          # Combine image + time-series data
│   │   ├── train.py                  # Generic training loop
│   │   ├── evaluate.py               # Metrics: TSS, HSS, Accuracy, F1
│   │   └── model_selection.py        # Compare models with evidence
│
│   ├── experiments/
│   │   ├── run_experiment.py         # Entry point for reproducible runs
│   │   ├── config_runner.py          # Load config + run pipeline
│   │   ├── hyperparameter_search.py  # Grid/Bayesian optimization
│   │   └── report_results.py         # Generate summary CSVs/plots
│
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── visualization.py
│   │   ├── metrics.py
│   │   ├── helpers.py
│   │   └── config_loader.py
│
│   └── pipeline/
│       ├── __init__.py
│       ├── pipeline_timeseries.py     # Full pipeline for IEEE + Stealth datasets
│       ├── pipeline_images.py         # Full pipeline for SDO Benchmark
│       ├── pipeline_fusion.py         # Combined multi-modal training pipeline
│       └── pipeline_utils.py          # Common functions for all pipelines
│
├── tests/
│   ├── __init__.py
│   ├── test_loaders.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_metrics.py
│   └── test_end_to_end.py
│
├── reports/
│   ├── figures/
│   ├── metrics/
│   ├── comparisons/
│   ├── model_results.csv
│   └── results_summary.md
│
└── docs/
    ├── data_sources.md
    ├── preprocessing_workflow.md
    ├── model_selection_criteria.md
    ├── multimodal_architecture.md
    ├── experiments_log.md
    └── project_plan.md

```