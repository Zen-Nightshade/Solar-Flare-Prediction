# src/data/process_magnetic_data.py (RAM-Optimized for 16GB)

import dask.dataframe as dd
import numpy as np
from dask.diagnostics import ProgressBar
import os
import gc
from sklearn.model_selection import train_test_split
import pyarrow as pa

def preprocess_data(ddf, is_test_set=False):
    """
    Preprocesses solar flare magnetic data.
    
    Parameters:
    -----------
    ddf : dask.dataframe.DataFrame
        Input Dask DataFrame
    is_test_set : bool, default=False
        If True, handles NaN labels (for holdout test set)
    
    Returns:
    --------
    dask.dataframe.DataFrame
        Preprocessed data
    """
    print("\n--- Starting Preprocessing Pipeline ---")

    # Step 1: Reset index and sort
    print("Step 1: Resetting index and sorting by record_id...")
    ddf = ddf.reset_index(drop=True).sort_values('record_id')

    # Step 2: Create sequence ID
    print("Step 2: Creating sequence identifier within each record_id...")
    def add_seq_id(df):
        df = df.copy()
        df['seq_id'] = df.groupby('record_id').cumcount()
        return df
    ddf = ddf.map_partitions(add_seq_id, meta=ddf._meta.assign(seq_id=0))
    print("Sequence ID created successfully.")

    # Step 3: Apply log transform
    print("Step 3: Applying log transform to skewed columns...")
    skewed_cols = [
        'TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH',
        'SAVNCPP', 'USFLUX', 'PIL_LEN', 'MEANPOT'
    ]
    available_skewed_cols = [c for c in skewed_cols if c in ddf.columns]
    
    if available_skewed_cols:
        def apply_log_transform(df, cols):
            df = df.copy()
            for col in cols:
                df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
            return df
        
        meta = ddf._meta.copy()
        for col in available_skewed_cols:
            meta[f'{col}_log'] = 0.0
        
        ddf = ddf.map_partitions(apply_log_transform, cols=available_skewed_cols, meta=meta)
        print(f"Log transform applied to: {available_skewed_cols}")
    else:
        print("No skewed columns found for log transform.")

    # Step 4: Advanced features (OPTIONAL - HIGH RAM!)
    print("Step 4: Engineering time-series features (HIGH RAM USAGE!)...")
    print("This may crash on systems with <32GB RAM")
    
    def _engineer_group_features(group_df):
        """Apply rolling statistics within each record_id group"""
        group_df = group_df.sort_values('seq_id').copy()
        
        # Rolling statistics for USFLUX_log
        group_df['USFLUX_log_roll_mean5'] = group_df['USFLUX_log'].rolling(
            window=5, min_periods=1
        ).mean()
        group_df['USFLUX_log_roll_std5'] = group_df['USFLUX_log'].rolling(
            window=5, min_periods=1
        ).std()
        
        # Difference features
        group_df['TOTPOT_log_diff1'] = group_df['TOTPOT_log'].diff(periods=1)
        
        # Lag features
        group_df['MEANSHR_lag3'] = group_df['MEANSHR'].shift(periods=3)
        
        return group_df
    
    # Create metadata by applying function to a sample
    print("Creating metadata for advanced features...")
    meta_sample = ddf.head(100)
    meta_df_example = meta_sample.groupby('record_id').apply(
        _engineer_group_features
    ).reset_index(drop=True)
    
    print("Applying time-series features to all data...")
    ddf_engineered = ddf.groupby('record_id').apply(
        _engineer_group_features, 
        meta=meta_df_example
    ).reset_index(drop=True)
    
    print("Advanced features engineered successfully")

    # Step 5: Select key columns
    print("Step 5: Selecting key columns for modeling...")
    
    features = [
        # --- Identifiers & Target ---
        'record_id',          # Unique ID for each time-series
        'seq_id',             # Time-step counter (0, 1, 2...)
        'label',              # Flare class label ('X', 'M', 'C', 'Q')

        # --- Foundational Features (Post-EDA) ---
        'USFLUX_log',         # Log-transformed: Total unsigned magnetic flux (proxy for size/energy)
        'TOTPOT_log',         # Log-transformed: Total photospheric magnetic energy
        'PIL_LEN_log',        # Log-transformed: Polarity Inversion Line length (proxy for complexity)
        'MEANSHR',            # Mean magnetic shear angle (proxy for stress)
        'TOTFZ',              # Total vertical magnetic flux (unique left-skewed feature)
        'EPSZ',               # Normalized helicity measure.
        'R_VALUE',            # Flux concentration (sparse, zero-inflated feature)

        # --- Engineered Time-Series Features ---
        'USFLUX_log_roll_mean5', # 5-step moving average of USFLUX_log (recent trend).
        'USFLUX_log_roll_std5',  # 5-step rolling stdev of USFLUX_log (recent volatility).
        'TOTPOT_log_diff1',      # 1-step change in TOTPOT_log (rate of change)
        'MEANSHR_lag3',          # Value of MEANSHR from 3 steps ago (memory)
    ]
    
    final_cols = [c for c in features if c in ddf_engineered.columns]
    ddf_final = ddf_engineered[final_cols]
    print(f"Selected {len(final_cols)} columns for modeling.")
    
    # Step 6: Handle label column (convert to string or handle NaN)
    print("Step 6: Processing label column...")
    if is_test_set:
        # For holdout test set, labels are NaN - convert to string "unknown"
        print("Test set detected - filling NaN labels with 'unknown'")
        ddf_final['label'] = ddf_final['label'].fillna('unknown').astype(str)
    else:
        # For train/dev/test splits, convert existing labels to string
        ddf_final['label'] = ddf_final['label'].astype(str)
    print("Label column processed successfully.")
    
    # Step 7: Handle NaNs in feature columns
    print("Step 7: Handling NaN values in feature columns...")
    feature_cols = [c for c in final_cols if c not in ['record_id', 'seq_id', 'label']]
    for col in feature_cols:
        ddf_final[col] = ddf_final[col].fillna(0)
    print(f"Filled NaN values in {len(feature_cols)} feature columns.")

    # Step 8: Reset index
    print("Step 8: Resetting index to ensure clean state...")
    ddf_final = ddf_final.reset_index(drop=True)
    print("Pipeline complete. Data will compute on-demand (RAM-safe).")
        
    return ddf_final


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("===========================================")
    print("  SOLAR FLARE DATA PROCESSING (16GB RAM)  ")
    print("===========================================")
    
    print("\n  BEFORE RUNNING - MAXIMIZE FREE RAM:")
    print("   1. Close your web browser")
    print("   2. Close VS Code (run this in terminal)")
    print("   3. Close any other applications")
    print("   4. Check free RAM: run 'free -h' in another terminal")
    print("\n You should have at least 20GB free RAM before proceeding.\n")
    
    proceed = input("Ready to proceed? (yes/no): ").strip().lower()
    if proceed != 'yes':
        print("Exiting. Run again when ready.")
        import sys
        sys.exit(0)
    
    print("\n  Configuring Dask for memory-constrained system...")
    
    # Configure Dask for limited memory
    import dask
    dask.config.set({
        'distributed.worker.memory.target': 0.6,
        'distributed.worker.memory.spill': 0.7,
        'distributed.worker.memory.pause': 0.8,
        'distributed.worker.memory.terminate': 0.95
    })
    
    print("Dask configured for memory-constrained system")
    print("\nProcessing training and test data sequentially...")
    print("Data will stream through disk to save RAM\n")
    
    # Defining schema for features
    output_schema = pa.schema([
        ('record_id', pa.int64()),
        ('seq_id', pa.int64()),
        ('label', pa.string()),
        ('USFLUX_log', pa.float64()),
        ('TOTPOT_log', pa.float64()),
        ('PIL_LEN_log', pa.float64()),
        ('MEANSHR', pa.float64()),
        ('TOTFZ', pa.float64()),
        ('EPSZ', pa.float64()),
        ('R_VALUE', pa.float64()),
        ('USFLUX_log_roll_mean5', pa.float64()),
        ('USFLUX_log_roll_std5', pa.float64()),
        ('TOTPOT_log_diff1', pa.float64()),
        ('MEANSHR_lag3', pa.float64())
    ])
    
    
    # ============================================================
    # Process TRAINING Data
    # ============================================================
    print("\n===========================================")
    print("      PROCESSING TRAINING DATA (WITH SPLIT)")
    print("===========================================")
    
    train_input_path = "../../data/preprocessed/solar_flare_2020/train/"
    train_output_path = "../../data/processed/magnetic_data/train/"
    dev_output_path = "../../data/processed/magnetic_data/dev/"
    test_output_path = "../../data/processed/magnetic_data/test/"
    
    print(f"Loading raw training data from: {train_input_path}")
    raw_train_ddf = dd.read_parquet(train_input_path)
    
    # Optimize partitions
    original_partitions = raw_train_ddf.npartitions
    print(f"Original partitions: {original_partitions}")
    
    target_partitions = max(8, min(original_partitions, 16))
    if original_partitions != target_partitions:
        print(f"Repartitioning to {target_partitions} for better memory efficiency...")
        raw_train_ddf = raw_train_ddf.repartition(npartitions=target_partitions)
    
    print(f"Using {raw_train_ddf.npartitions} partitions")
    print(f"Loaded {len(raw_train_ddf.columns)} columns")
    
    # Estimating memory
    print("Estimating memory requirements...")
    sample_size = raw_train_ddf.head(1000).memory_usage(deep=True).sum() / 1000
    estimated_gb = (sample_size * len(raw_train_ddf)) / 1e9
    print(f"Estimated RAM needed: {estimated_gb:.2f} GB")
    
    # Spliting by record_id
    print("\n===========================================")
    print("     STEP 1: SPLITTING RAW DATA FIRST      ")
    print("===========================================")
    print("Spliting raw data BEFORE expensive processing to save RAM")
    
    print("\nGetting unique record IDs from raw data...")
    with ProgressBar():
        unique_ids = raw_train_ddf['record_id'].unique().compute()
    print(f"Found {len(unique_ids)} unique record IDs")
    
    print("\nSplitting record IDs into train/dev/test...")
    train_dev_ids, test_ids = train_test_split(unique_ids, test_size=0.01, random_state=42)
    dev_size_relative = 0.01 / 0.99
    train_ids, dev_ids = train_test_split(train_dev_ids, test_size=dev_size_relative, random_state=42)
    
    print(f"Train IDs: {len(train_ids)}, Dev IDs: {len(dev_ids)}, Test IDs: {len(test_ids)}")
    
    # Free memory
    del unique_ids, train_dev_ids
    gc.collect()
    
    # ============================================================
    # Process and Save TRAIN Split
    # ============================================================
    print("\n===========================================")
    print("     STEP 2: PROCESS TRAIN SPLIT           ")
    print("===========================================")
    
    print("Filtering raw data for train split...")
    train_raw_ddf = raw_train_ddf[raw_train_ddf['record_id'].isin(train_ids)]
    print(f"Train split rows: {len(train_raw_ddf)}")
    
    print("\nProcessing train split...")
    train_processed_ddf = preprocess_data(
        train_raw_ddf,
        is_test_set=False
    )
    
    print("\nSaving train split...")
    os.makedirs(train_output_path, exist_ok=True)
    with ProgressBar():
        train_processed_ddf.to_parquet(
            train_output_path,
            engine="pyarrow",
            write_index=False,
            compression="snappy",
            schema=output_schema,
            overwrite=True
        )
    print(f"Train data saved to: {train_output_path}")
    
    # Free memory aggressively
    del train_raw_ddf, train_processed_ddf, train_ids
    gc.collect()
    print("Train memory freed\n")
    
    # ============================================================
    # Process and Save DEV Split
    # ============================================================
    print("===========================================")
    print("     STEP 3: PROCESS DEV SPLIT             ")
    print("===========================================")
    
    print("Filtering raw data for dev split...")
    dev_raw_ddf = raw_train_ddf[raw_train_ddf['record_id'].isin(dev_ids)]
    print(f"Dev split rows: {len(dev_raw_ddf)}")
    
    print("\nProcessing dev split...")
    dev_processed_ddf = preprocess_data(
        dev_raw_ddf, 
        is_test_set=False
    )
    
    print("\nSaving dev split...")
    os.makedirs(dev_output_path, exist_ok=True)
    with ProgressBar():
        dev_processed_ddf.to_parquet(
            dev_output_path,
            engine="pyarrow",
            write_index=False,
            compression="snappy",
            schema=output_schema,
            overwrite=True
        )
    print(f"Dev data saved to: {dev_output_path}")
    
    # Free memory
    del dev_raw_ddf, dev_processed_ddf, dev_ids
    gc.collect()
    print("Dev memory freed\n")
    
    # ============================================================
    # Process and Save TEST Split
    # ============================================================
    print("===========================================")
    print("     STEP 4: PROCESS TEST SPLIT            ")
    print("===========================================")
    
    print("Filtering raw data for test split...")
    test_raw_ddf = raw_train_ddf[raw_train_ddf['record_id'].isin(test_ids)]
    print(f"Test split rows: {len(test_raw_ddf)}")
    
    print("\nProcessing test split...")
    test_processed_ddf = preprocess_data(
        test_raw_ddf, 
        is_test_set=False
    )
    
    print("\nSaving test split...")
    os.makedirs(test_output_path, exist_ok=True)
    with ProgressBar():
        test_processed_ddf.to_parquet(
            test_output_path,
            engine="pyarrow",
            write_index=False,
            compression="snappy",
            schema=output_schema,
            overwrite=True
        )
    print(f"Test data saved to: {test_output_path}")
    
    # Free memory
    del test_raw_ddf, test_processed_ddf, test_ids, raw_train_ddf
    gc.collect()
    print("Test memory freed")

    # ============================================================
    # Process HOLDOUT TEST Data
    # ============================================================
    print("\n===========================================")
    print("      PROCESSING HOLDOUT TEST DATA         ")
    print("===========================================")
    
    holdout_input_path = "../../data/preprocessed/solar_flare_2020/test/"
    holdout_output_path = "../../data/processed/magnetic_data/holdout_test/"
    
    print(f"Loading raw holdout test data from: {holdout_input_path}")
    raw_holdout_ddf = dd.read_parquet(holdout_input_path)
    
    # Optimize partitions
    original_partitions = raw_holdout_ddf.npartitions
    print(f"Original partitions: {original_partitions}")
    
    target_partitions = max(4, min(original_partitions, 8))
    if original_partitions != target_partitions:
        print(f"Repartitioning to {target_partitions} for better memory efficiency...")
        raw_holdout_ddf = raw_holdout_ddf.repartition(npartitions=target_partitions)
    
    print(f"Using {raw_holdout_ddf.npartitions} partitions")
    print(f"Loaded {len(raw_holdout_ddf.columns)} columns")
    
    # Estimate memory
    print("Estimating memory requirements...")
    sample_size = raw_holdout_ddf.head(1000).memory_usage(deep=True).sum() / 1000
    estimated_gb = (sample_size * len(raw_holdout_ddf)) / 1e9
    print(f"Estimated RAM needed: {estimated_gb:.2f} GB")
    
    # Preprocess holdout test (labels are NaN)
    print("\nProcessing holdout test set (labels will be set to 'unknown')")
    processed_holdout_ddf = preprocess_data(
        raw_holdout_ddf,
        is_test_set=True  # <-- This handles NaN labels
    )
    print(f"Preprocessing pipeline built successfully.")
    
    print(f"\nTotal rows after processing: {len(processed_holdout_ddf)}")
    
    print(f"\n===========================================")
    print(f"       SAVING HOLDOUT TEST DATA           ")
    print("===========================================")
    os.makedirs(holdout_output_path, exist_ok=True)
    print(f"Total rows in holdout test set: {len(processed_holdout_ddf)}")
    
    with ProgressBar():
        processed_holdout_ddf.to_parquet(
            holdout_output_path,
            engine="pyarrow",
            write_index=False,
            compression="snappy",
            schema=output_schema,
            overwrite=True
        )
    print(f"Holdout test data saved successfully to: {holdout_output_path}")
    
    # Cleanup
    print("\nCleaning up memory after holdout test processing...")
    del raw_holdout_ddf, processed_holdout_ddf
    gc.collect()
    print("Holdout test data processing complete. Memory freed.")

    # ============================================================
    # Summary
    # ============================================================
    print("\n===========================================")
    print("     ALL PROCESSING COMPLETED SUCCESSFULLY ")
    print("===========================================")
    print(f"Train data:        {train_output_path}")
    print(f"Dev data:          {dev_output_path}")
    print(f"Test data:         {test_output_path}")
    print(f"Holdout test data: {holdout_output_path}")