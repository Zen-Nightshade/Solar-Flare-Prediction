# src/data/augment_and_resample.py

import dask.dataframe as dd
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from dask.diagnostics import ProgressBar
import os
import shutil
import gc

def generate_warping_curve(n_steps=60, n_knots=4, std_dev=0.2):
    x = np.linspace(0, n_steps - 1, n_knots)
    y = np.random.normal(loc=1.0, scale=std_dev, size=n_knots)
    cs = CubicSpline(x, y)
    return cs(np.arange(n_steps))

def apply_scaling(df: pd.DataFrame, scale_range=(0.9, 1.1)) -> pd.DataFrame:
    scaled_df = df.copy()
    feature_cols = [c for c in df.columns if c not in ['record_id', 'seq_id', 'label', 'safe_index']]
    
    for col in feature_cols:
        scaler = np.random.uniform(low=scale_range[0], high=scale_range[1])
        scaled_df[col] = scaled_df[col] * scaler
    return scaled_df

def apply_magnitude_warping(df: pd.DataFrame) -> pd.DataFrame:
    warped_df = df.copy()
    feature_cols = [c for c in df.columns if c not in ['record_id', 'seq_id', 'label', 'safe_index']]
    
    warping_curve = generate_warping_curve()
    for col in feature_cols:
        warped_df[col] = warped_df[col] * warping_curve
    return warped_df

# This global variable will be our counter for augmented IDs
new_id_counter = 0

def process_dataframe(df: pd.DataFrame, strategy: dict) -> pd.DataFrame:
    global new_id_counter
    
    if df.empty:
        return df

    # --- Generate new synthetic data for X and M classes ---
    synthetic_records = []
    
    for label, counts in strategy['synthetic'].items():
        class_df = df[df['label'] == label]
        for _, group in class_df.groupby('record_id'):
            # Create warped versions
            for _ in range(counts['warp']):
                new_group = apply_magnitude_warping(group)
                new_group['record_id'] = new_id_counter
                new_group['safe_index'] = new_id_counter * 60 + new_group['seq_id']
                new_id_counter += 1
                synthetic_records.append(new_group)
            
            # Create scaled versions
            for _ in range(counts['scale']):
                new_group = apply_scaling(group)
                new_group['record_id'] = new_id_counter
                new_group['safe_index'] = new_id_counter * 60 + new_group['seq_id']
                new_id_counter += 1
                synthetic_records.append(new_group)

    # ---  Oversampling (Duplication) ---
    records_to_duplicate = []
    original_records_to_keep = [] # Keep track of originals that are duplicated
    
    for label, params in strategy['duplicate'].items():
        class_df = df[df['label'] == label]
        unique_ids = class_df['record_id'].unique()
        num_to_duplicate = int(len(unique_ids) * params['frac'])
        
        if num_to_duplicate > 0:
            ids_to_duplicate = np.random.choice(unique_ids, size=num_to_duplicate, replace=False)
            df_to_duplicate = class_df[class_df['record_id'].isin(ids_to_duplicate)]
            original_records_to_keep.append(df_to_duplicate) # Keep the original versions
            
            for i in range(params['times']):
                # We must iterate record by record to assign unique new IDs
                for _, group in df_to_duplicate.groupby('record_id'):
                    duplicated_group = group.copy()
                    duplicated_group['record_id'] = new_id_counter
                    duplicated_group['safe_index'] = new_id_counter * 60 + duplicated_group['seq_id']
                    new_id_counter += 1
                    records_to_duplicate.append(duplicated_group)
    
    # --- Undersampling for Q class ---
    majority_df = df[df['label'] == 'Q']
    surviving_majority_df = majority_df.sample(frac=1.0 - strategy['undersample']['Q']['frac'])
    
    final_parts = [surviving_majority_df]
    
    # Adding the original minority classes
    final_parts.append(df[df['label'] != 'Q'])
    
    # Adding the synthetic records
    if synthetic_records:
        final_parts.append(pd.concat(synthetic_records, ignore_index=True))
        
    # Adding the new duplicated records
    if records_to_duplicate:
        final_parts.append(pd.concat(records_to_duplicate, ignore_index=True))
        
    return pd.concat(final_parts, ignore_index=True)

# =====================================================================================================================================================
# MAIN EXECUTION
# =====================================================================================================================================================

if __name__ == "__main__":
    train_data_path = "/../../data/processed/magnetic/train/"
    
    print(f"Loading training data from: {train_data_path}")
    ddf = dd.read_parquet(train_data_path)
    
    print("\n--- Calculating Initial State ---")
    with ProgressBar():
        print("Computing entire DataFrame into memory...(Ensure that there is enough RAM, like around 23GB available)")
        full_df = ddf.compute()
        max_id = full_df['record_id'].max()

    class_counts = full_df.drop_duplicates(subset=['record_id'])['label'].value_counts()
    print("Original sample distribution:\n", class_counts)
    print(f"Maximum existing record_id: {max_id}")
    
    new_id_counter = max_id + 1

    # Our strategy
    augmentation_strategy = {
        'synthetic': {'X': {'warp': 8, 'scale': 8}, 'M': {'warp': 1, 'scale': 1}},
        'duplicate': {'X': {'frac': 1.0, 'times': 4}, 'B': {'frac': 0.1, 'times': 1}, 'C': {'frac': 0.05, 'times': 1}},
        'undersample': {'Q': {'frac': 0.6}}
    }
    
    print("\n--- Applying Augmentation and Resampling ---")
    
    # Processing the entire Pandas DataFrame at once
    final_df = process_dataframe(full_df, augmentation_strategy)
    
    print("Augmentation and resampling complete.")

    temp_output_path = f"{train_data_path.rstrip('/')}_temp/"
    
    if os.path.exists(temp_output_path):
        shutil.rmtree(temp_output_path)
    os.makedirs(temp_output_path)

    print("\nSaving final data to temporary location...")
    final_ddf = dd.from_pandas(final_df, npartitions=ddf.npartitions)
    with ProgressBar():
        final_ddf.to_parquet(temp_output_path, write_index=False, overwrite=True)

    del ddf, full_df, final_df, final_ddf
    gc.collect()

    print(f"\nDeleting old directory: {train_data_path}")
    shutil.rmtree(train_data_path)
    
    print(f"Renaming {temp_output_path} to {train_data_path}")
    os.rename(temp_output_path, train_data_path)
    
    print("\n--- Augmentation Complete ---")
    
    print("\n--- Step 4: Verifying Final Class Distribution ---")
    final_ddf_verify = dd.read_parquet(train_data_path)
    with ProgressBar():
        final_id_label_map = final_ddf_verify[['record_id', 'label']].drop_duplicates(subset=['record_id']).compute()
        final_class_counts = final_id_label_map['label'].value_counts()
    print("New sample distribution:\n", final_class_counts)
    print("\nTotal samples in new training set:", len(final_id_label_map))