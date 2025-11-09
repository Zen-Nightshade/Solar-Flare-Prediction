# src/diagnostics/check_corrupted_distribution.py

import dask.dataframe as dd
from dask.diagnostics import ProgressBar

def find_corrupted_distribution(data_path: str):
    print(f"--- Starting Scan ---")
    print(f"Loading data from: {data_path}")
    
    try:
        ddf = dd.read_parquet(data_path)
        ddf['record_id'] = ddf['record_id'].astype(str)
    except Exception as e:
        print(f"\nError loading data: {e}")
        return

    print(f"Successfully loaded {ddf.npartitions} partitions.")
    
    print("\n Identifying all records with more than 60 rows...")
    record_counts = ddf.groupby('record_id').size()
    oversized_records = record_counts[record_counts > 60]
    
    with ProgressBar():
        culprit_ids = oversized_records.index.compute()
    
    if len(culprit_ids) == 0:
        print("\nNo oversized records found. Cannot compute distribution.")
        return
        
    print(f"Found {len(culprit_ids)} problematic record_ids.")

    print("\nFiltering data to get labels for only the corrupted records...")
    
    corrupted_ddf = ddf[ddf['record_id'].isin(culprit_ids)]
    corrupted_labels = corrupted_ddf[['record_id', 'label']].drop_duplicates(subset=['record_id'])
    
    print("\nComputing the flare label distribution...")
    with ProgressBar():
        distribution = corrupted_labels['label'].value_counts().compute()
        
    print("\n--- Scan Complete ---")
    print(f"\nFlare Label Distribution for the {len(culprit_ids)} records with > 60 rows:")
    print("--------------------------------------------------")
    print(distribution)
    print("--------------------------------------------------")

if __name__ == "__main__":
    train_data_path = "/../../data/processed/magnetic/train/"
    
    find_corrupted_distribution(train_data_path)