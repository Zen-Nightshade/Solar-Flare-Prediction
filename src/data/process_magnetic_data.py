# src/data/process_magnetic_data.py

import dask.dataframe as dd
import numpy as np
from dask.diagnostics import ProgressBar
import os
import gc
from sklearn.model_selection import train_test_split
import pyarrow as pa
import dask

def preprocess_data(ddf, is_test_set=False):
    print("\n--- Starting Preprocessing Pipeline ---")

    print("Step 1: Setting 'safe_index' as the index and sorting...")
    ddf = ddf.set_index('safe_index', sorted=False)
    
    print("Step 2: Applying log transform to skewed columns...")
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
    
    print("Step 3: Engineering time-series features (HIGH RAM USAGE!)...")
    
    def _engineer_group_features(group_df):
        """Apply rolling statistics within each record_id group"""
        group_df = group_df.sort_values('seq_id').copy()
        
        group_df['USFLUX_log_roll_mean5'] = group_df['USFLUX_log'].rolling(window=5, min_periods=1).mean()
        group_df['USFLUX_log_roll_std5'] = group_df['USFLUX_log'].rolling(window=5, min_periods=1).std()
        group_df['TOTPOT_log_diff1'] = group_df['TOTPOT_log'].diff(periods=1)
        group_df['MEANSHR_lag3'] = group_df['MEANSHR'].shift(periods=3)
        
        return group_df
    
    print("Creating metadata for advanced features...")
    meta_sample = ddf.head(120)
    meta_df_example = meta_sample.groupby('record_id').apply(
        _engineer_group_features,
        include_groups=False
    )
    
    print("Applying time-series features to all data...")
    ddf_engineered = ddf.groupby('record_id').apply(
        _engineer_group_features, 
        meta=meta_df_example,
        include_groups=False
    ).reset_index(drop=False)
    
    print("Advanced features engineered successfully")
    
    print("Step 4: Selecting key columns for modeling...")
    
    features = [
        'record_id', 'seq_id', 'label', 'USFLUX_log', 'TOTPOT_log',
        'PIL_LEN_log', 'MEANSHR', 'TOTFZ', 'EPSZ', 'R_VALUE',
        'USFLUX_log_roll_mean5', 'USFLUX_log_roll_std5',
        'TOTPOT_log_diff1', 'MEANSHR_lag3',
    ]
    
    final_cols = [c for c in features if c in ddf_engineered.columns]
    ddf_final = ddf_engineered[final_cols]
    print(f"Selected {len(final_cols)} columns for modeling.")
    
    print("Step 5: Processing label column...")
    if is_test_set:
        print("Test set detected - filling NaN labels with 'unknown'")
        ddf_final['label'] = ddf_final['label'].fillna('unknown').astype(str)
    else:
        ddf_final['label'] = ddf_final['label'].astype(str)
    print("Label column processed successfully.")
    
    print("Step 6: Handling NaN values in feature columns...")
    feature_cols = [c for c in final_cols if c not in ['record_id', 'seq_id', 'label']]
    for col in feature_cols:
        ddf_final[col] = ddf_final[col].fillna(0)
    print(f"Filled NaN values in {len(feature_cols)} feature columns.")
    
    print("Step 7: Resetting index to ensure clean state...")
    ddf_final = ddf_final.reset_index()
    print("Pipeline complete.")
        
    return ddf_final


# ====================================================================================================================================================
# MAIN EXECUTION
# ====================================================================================================================================================
if __name__ == "__main__":
    print("===========================================")
    print("        SOLAR FLARE DATA PROCESSING        ")
    print("===========================================")
    
    proceed = input("This process is RAM-intensive. Ensure other apps are closed and there is atleast 23GB free ram available. Ready to proceed? (yes/no): ").strip().lower()
    if proceed != 'yes':
        print("Exiting.")
        import sys
        sys.exit(0)
    
    dask.config.set({
        'distributed.worker.memory.target': 0.6,
        'distributed.worker.memory.spill': 0.7,
        'distributed.worker.memory.pause': 0.8,
        'distributed.worker.memory.terminate': 0.95
    })
    
    output_schema = pa.schema([
        # ('safe_index', pa.int64()),
        ('record_id', pa.string()),
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
    
    preprocessed_base_path = "/../../data/preprocessed/solar_flare_2020"
    processed_base_path = "/../../data/processed/magnetic_data/"
    
    print("\n===========================================")
    print("      PROCESSING TRAINING DATA (WITH SPLIT)")
    print("===========================================")
    
    train_input_path = preprocessed_base_path
    train_output_path = os.path.join(processed_base_path, "train/")
    dev_output_path = os.path.join(processed_base_path, "dev/")
    test_output_path = os.path.join(processed_base_path, "test/")
    
    raw_train_ddf = dd.read_parquet(train_input_path)
    raw_train_ddf = raw_train_ddf.repartition(npartitions=16)
    raw_train_ddf['record_id'] = raw_train_ddf['record_id'].astype(str)
    
    print("Splitting raw data using Stratification...")
    with ProgressBar():
        id_to_label_map = raw_train_ddf[['record_id', 'label']].drop_duplicates(subset=['record_id']).compute()
        
    train_dev_map, test_map = train_test_split(
        id_to_label_map, 
        test_size=0.01,
        random_state=42, 
        stratify=id_to_label_map['label']
    )
    
    dev_size_relative_to_remainder = 0.01 / 0.99
    train_map, dev_map = train_test_split(
        train_dev_map,
        test_size=dev_size_relative_to_remainder,
        random_state=42,
        stratify=train_dev_map['label']
    )    

    train_ids = train_map['record_id'].values
    dev_ids = dev_map['record_id'].values
    test_ids = test_map['record_id'].values

    print(f"Train IDs: {len(train_ids)}, Dev IDs: {len(dev_ids)}, Test IDs: {len(test_ids)}")
    
    print("\nVerifying stratification...")
    print("Original distribution:\n", id_to_label_map['label'].value_counts(normalize=True))
    print("\nDev set distribution:\n", dev_map['label'].value_counts(normalize=True))
    print("\nTest set distribution:\n", test_map['label'].value_counts(normalize=True))
    
    del id_to_label_map, train_dev_map, test_map, train_map, dev_map
    gc.collect()
    
    # --- Process and Save TRAIN Split ---
    print("\n--- Processing TRAIN Split ---")
    train_raw_ddf = raw_train_ddf[raw_train_ddf['record_id'].isin(train_ids)]
    train_processed_ddf = preprocess_data(train_raw_ddf, is_test_set=False)
    os.makedirs(train_output_path, exist_ok=True)
    with ProgressBar():
        train_processed_ddf.to_parquet(train_output_path, schema=output_schema, overwrite=True, write_index=False)
    print(f"Train data saved to: {train_output_path}")
    del train_raw_ddf, train_processed_ddf, train_ids; gc.collect()

    # --- Process and Save DEV Split ---
    print("\n--- Processing DEV Split ---")
    dev_raw_ddf = raw_train_ddf[raw_train_ddf['record_id'].isin(dev_ids)]
    dev_processed_ddf = preprocess_data(dev_raw_ddf, is_test_set=False)
    os.makedirs(dev_output_path, exist_ok=True)
    with ProgressBar():
        dev_processed_ddf.to_parquet(dev_output_path, schema=output_schema, overwrite=True, write_index=False)
    print(f"Dev data saved to: {dev_output_path}")
    del dev_raw_ddf, dev_processed_ddf, dev_ids; gc.collect()

    # --- Process and Save TEST Split ---
    print("\n--- Processing TEST Split ---")
    test_raw_ddf = raw_train_ddf[raw_train_ddf['record_id'].isin(test_ids)]
    test_processed_ddf = preprocess_data(test_raw_ddf, is_test_set=False)
    os.makedirs(test_output_path, exist_ok=True)
    with ProgressBar():
        test_processed_ddf.to_parquet(test_output_path, schema=output_schema, overwrite=True, write_index=False)
    print(f"Test data saved to: {test_output_path}")
    del test_raw_ddf, test_processed_ddf, test_ids, raw_train_ddf; gc.collect()

    print("\n===========================================")
    print("     ALL PROCESSING COMPLETED SUCCESSFULLY ")
    print("===========================================")