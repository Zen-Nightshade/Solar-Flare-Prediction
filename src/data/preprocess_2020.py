# src/data/preprocess_2020.py

import pandas as pd
import os

input_dir = "../../data/raw/solar_flare_2020/"

output_dir = "../../data/processed/solar_flare_2020/"
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

chunk_size = 4000


def json_row_processor(row):
    row = row.dropna()
    if row.empty:
        raise ValueError("Row is empty after dropna()")
        
    record_id = row.index[0]
    data_dict = row.iloc[0]
    
    if not isinstance(data_dict, dict):
        raise ValueError(f"Expected a dictionary, but got {type(data_dict)}")

    values = data_dict.get("values")
    if not values: return None
    
    # If there's no magnetic flux, the data is useless.
    usflux_series = values.get("USFLUX")
    if usflux_series is None or not any(usflux_series.values()):
        # This will catch both missing USFLUX and all-zero USFLUX series.
        return None
    
    try:
        row_df = pd.DataFrame(values)
        row_df.fillna(0, inplace=True)
    except Exception:
        return None
    
    row_df['record_id'] = record_id
    row_df['label'] = data_dict.get("label")
    
    return row_df

# json_row_processor(df1.iloc[0])


def json_chunk_processor(chunk):
    processed_rows = []
    for index, row in chunk.iterrows():
        row_df = json_row_processor(row)
        if row_df is not None:
            if len(row_df) == 0: # Check if the DataFrame is empty
                print(f"\n[Info] Found a record that resulted in an empty DataFrame at chunk index {index}.")
            processed_rows.append(row_df)
    
    if not processed_rows:
        return None
    
    chunk_df = pd.concat(processed_rows, ignore_index=True)
    return chunk_df

# json_chunk_processor(df1)

# Preprocessing Dataset

name_index = 0

for file in ["train_partition1_data.json", "train_partition2_data.json", "train_partition3_data.json", "test_4_5_data.json"]:
    print(f"Preprocessing {file}\n")
    file_path = os.path.join(input_dir, file)
    json_reader = pd.read_json(file_path, lines=True, chunksize=chunk_size)

    for index, chunk in enumerate(json_reader):
        print(f"processing chunk- {index+1}...",)
        chunk_df = json_chunk_processor(chunk)
        
        name_index += 1
        save_dir = None
        mode = None

        if file == "test_4_5_data.json":
            save_dir = test_dir
            mode = "test"
        else:
            save_dir = train_dir
            mode = "train"
        output_path = os.path.join(save_dir, f'{mode}_chunk_{name_index}.parquet')
        chunk_df.to_parquet(output_path, engine='pyarrow')

