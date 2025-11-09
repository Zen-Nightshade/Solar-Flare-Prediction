# src/data/preprocess_2020.py

import pandas as pd
import os

input_dir = "/../../data/raw/solar_flare_2020/"
output_dir = "/../../data/preprocessed/solar_flare_2020/"
chunk_size = 4000

# --- Global State Trackers ---

processed_record_ids = set()

new_id_counter = 171038


def json_row_processor(row):
    
    global new_id_counter

    row = row.dropna()
    if row.empty:
        return None
        
    try:
        original_record_id = int(row.index[0])
    except (ValueError, IndexError):
        
        return None
    
    if original_record_id in processed_record_ids:
        # This is a collision. Assigning a new ID from our counter.
        record_id_int = new_id_counter
        new_id_counter += 1
    else:
        record_id_int = original_record_id

    data_dict = row.iloc[0]
    
    if not isinstance(data_dict, dict):
        return None

    values = data_dict.get("values")
    if not values: return None
    
    usflux_series = values.get("USFLUX")
    if usflux_series is None or not any(usflux_series.values()):
        return None
    
    try:
        row_df = pd.DataFrame(values)
        row_df.fillna(0, inplace=True)
    except Exception:
        return None
    
    if len(row_df) != 60:
        print(f"\n[Warning] Skipping original record '{original_record_id}' (line had {len(row_df)} rows).")
        return None

    processed_record_ids.add(record_id_int)
    
    row_df.reset_index(inplace=True)
    row_df.rename(columns={'index': 'seq_id'}, inplace=True)
    row_df['seq_id'] = row_df['seq_id'].astype(int)
    
    row_df['record_id'] = record_id_int
    row_df['label'] = data_dict.get("label")

    row_df['safe_index'] = record_id_int * 60 + row_df['seq_id']
    
    return row_df


def json_chunk_processor(chunk):
    """Applies the row processor to each row in a chunk of data."""
    processed_rows = []
    for index, row in chunk.iterrows():
        row_df = json_row_processor(row)
        if row_df is not None:
            processed_rows.append(row_df)
    
    if not processed_rows:
        return None
    
    return pd.concat(processed_rows, ignore_index=True)


# --- Main Execution Block ---

if __name__ == "__main__":
    print("--- Starting Preprocessing with Sequential Renumbering ---")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")
    print("WARNING: This script will ADD files to the output directory. Please ensure it is empty before running.")

    name_index = 0
    files_to_process = ["train_partition1_data.json", "train_partition2_data.json", "train_partition3_data.json"]

    for file in files_to_process:
        file_path = os.path.join(input_dir, file)
        if not os.path.exists(file_path):
            print(f"\nWarning: File not found, skipping: {file_path}")
            continue
            
        print(f"\nPreprocessing {file}...")
        json_reader = pd.read_json(file_path, lines=True, chunksize=chunk_size)

        for index, chunk in enumerate(json_reader):
            print(f"  - Processing chunk {index+1}...", end='\r')
            chunk_df = json_chunk_processor(chunk)
            
            if chunk_df is not None and not chunk_df.empty:
                name_index += 1
                output_path = os.path.join(output_dir, f'train_chunk_{name_index}.parquet')
                chunk_df.to_parquet(output_path, engine='pyarrow')

    print("\n\n--- Preprocessing Complete ---")
    print(f"Total unique records generated: {len(processed_record_ids)}")
    if new_id_counter > 171038:
        print(f"New record IDs were assigned starting from 171038 up to {new_id_counter - 1}")
    else:
        print("No duplicate records were found that required renumbering.")
    print(f"Data saved in {name_index} chunks.")