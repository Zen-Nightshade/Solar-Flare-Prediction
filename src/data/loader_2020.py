# src/data/loader_2020.py

import requests
import os
from tqdm import tqdm


# The original link is "https://www.kaggle.com/competitions/bigdata2020-flare-prediction/data"
base_url = "https://dmlab.cs.gsu.edu/solar/data/data-comp-2020/"
files = [
    # "test_4_5_data.json",
    "train_partition1_data.json",
    "train_partition2_data.json"
    "train_partition3_data.json"
]

filenames = [
    # "test_4_5_data.json",
    "train_partition1_data.json",
    "train_partition2_data.json",
    "train_partition3_data.json"
]


output_dir = "../../data/raw/solar_flare_2020"
os.makedirs(output_dir, exist_ok=True)

for file, filename in zip(files, filenames):
    url = base_url + file
    save_path = os.path.join(output_dir, filename)

    print(f"Starting download: {filename}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB

        progress_bar = tqdm(
            total=total_size_in_bytes, 
            unit='iB', 
            unit_scale=True, 
            desc=filename
        )

        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    progress_bar.update(len(chunk))
                    file.write(chunk)
        
        progress_bar.close()
        
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
             print(f"ERROR: Download incomplete for {filename}")

    except requests.exceptions.RequestException as e:
        print(f"\nError downloading {filename}: {e}")