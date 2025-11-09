# src/diagnostics/check_magnetic_record_counts.py

import dask.dataframe as dd
from dask.diagnostics import ProgressBar

def find_oversized_records(data_path: str):
    print(f"--- Starting Scan ---")
    print(f"Loading data from: {data_path}")
    
    try:
        ddf = dd.read_parquet(data_path)
    except Exception as e:
        print(f"\nError loading data: {e}")
        return

    print(f"Successfully loaded {ddf.npartitions} partitions.")
    

    print("Calculating row count for each record_id...")
    record_counts = ddf.groupby('record_id').size()
    
    oversized_records = record_counts[record_counts > 60]
    
    print("Computing the results...")
    with ProgressBar():
        culprits = oversized_records.compute()
    
    
    print("\n--- Scan Completed ---")
    
    if culprits.empty:
        print("\nNo records found with more than 60 rows.")
        print("All records have the correct number of time steps.")
    else:
        print(f"\nFOUND {len(culprits)} problematic record(s) with more than 60 rows:")
        print("--------------------------------------------------")
        print(culprits)
        # print("--------------------------------------------------")

if __name__ == "__main__":
    train_data_path = "/../../data/preprocessed/solar_flare_2020"
    
    find_oversized_records(train_data_path)