# src/data/process_magnetic_data.py

import dask.dataframe as dd
import numpy as np
from dask.diagnostics import ProgressBar
import os

def preprocess_data(ddf):
    print("\n--- Starting Preprocessing Pipeline ---")

    # Reset index and sort for stability
    print("Resetting index and sorting by record_id...")
    ddf = ddf.reset_index(drop=True).sort_values('record_id')

    # Create sequence ID
    print("Creating sequence identifier within each record_id...")
    def add_seq_id(df):
        df = df.copy()
        df['seq_id'] = df.groupby('record_id').cumcount()
        return df
    ddf = ddf.map_partitions(add_seq_id, meta=ddf._meta.assign(seq_id=0))

    # Apply log transform to the CORRECT skewed columns from our EDA
    print("Applying log transform to skewed columns...")
    
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
        
        meta = ddf._meta
        for col in available_skewed_cols:
            meta[f'{col}_log'] = 0.0

        ddf = ddf.map_partitions(
            apply_log_transform, 
            cols=available_skewed_cols,
            meta=meta
        )
        # print(f"Log transform applied to: {available_skewed_cols}")

    print("Selecting key columns for modeling...")
    
    features_to_keep = [
        # --- Identifiers and Target ---
        'record_id', 'seq_id', 'label',

        # --- Representatives from the highly correlated 'Energy/Flux' group ---
        'USFLUX_log',  # Represents the total magnetic flux (size)
        'TOTPOT_log',  # Represents the total magnetic energy
        
        # --- Other important features from our EDA ---
        'PIL_LEN_log', # Polarity Inversion Line length (a different physical property)
        'MEANSHR',     # Represents magnetic complexity/shear
        'TOTFZ',       # A feature with a different (left-skewed) distribution
        'EPSZ',        # A normalized feature
        'R_VALUE'      # The zero-inflated feature we decided to keep
    ]

    final_cols = [c for c in features_to_keep if c in ddf.columns]
    ddf = ddf[final_cols]
    # print(f"Selected {len(final_cols)} columns for output.")

    print("Resetting index to ensure clean state...")
    ddf = ddf.reset_index(drop=True)

    print("Persisting intermediate data...")
    try:
        with ProgressBar():
            ddf = ddf.persist(scheduler="threads")
        print("Data persisted successfully in memory.")
    except Exception as e:
        print(f"Persist failed: {e}")
        print("Continuing without persist...")
        
    return ddf

# MAIN EXECUTION
if __name__ == "__main__":
    # Processing Training Data
    print("===========================================")
    print("         PROCESSING TRAINING DATA          ")
    print("===========================================")

    train_input_path = "../../data/preprocessed/solar_flare_2020/train/"
    train_output_path = "../../data/processed/magnetic_data/train/"

    print(f"Loading raw train data from: {train_input_path}")
    raw_train_ddf = dd.read_parquet(train_input_path)
    
    print(f"Loaded {len(raw_train_ddf.columns)} columns, {raw_train_ddf.npartitions} partitions")

    processed_train_ddf = preprocess_data(raw_train_ddf)

    print("\nSaving processed training data to disk...")
    os.makedirs(train_output_path, exist_ok=True)

    print("Computing basic statistics...")
    print(f"Total rows: {len(processed_train_ddf)}")
    
    with ProgressBar():
        processed_train_ddf.to_parquet(
            train_output_path,
            engine="pyarrow",
            write_index=False,
            compression="snappy",
            overwrite=True
        )

    print(f"Processed training data saved successfully to: {train_output_path}")

    # Processing Test Data
    print("\n===========================================")
    print("          PROCESSING TEST DATA             ")
    print("===========================================")

    test_input_path = "../../data/preprocessed/solar_flare_2020/test/"
    test_output_path = "../../data/processed/magnetic_data/test/"

    print(f"Loading raw test data from: {test_input_path}")
    raw_test_ddf = dd.read_parquet(test_input_path)
    
    print(f"Loaded {len(raw_test_ddf.columns)} columns, {raw_test_ddf.npartitions} partitions")

    processed_test_ddf = preprocess_data(raw_test_ddf)

    print("\nSaving processed test data to disk...")
    os.makedirs(test_output_path, exist_ok=True)

    print("Computing basic statistics...")
    print(f"Total rows: {len(processed_test_ddf)}")
    
    with ProgressBar():
        processed_test_ddf.to_parquet(
            test_output_path,
            engine="pyarrow",
            write_index=False,
            compression="snappy",
            overwrite=True
        )

    print(f"Processed test data saved successfully to: {test_output_path}")
    
    print("\n===========================================")
    print("     ALL PROCESSING COMPLETED SUCCESSFULLY ")
    print("===========================================")
    print(f"Train data: {train_output_path}")
    print(f"Test data: {test_output_path}")