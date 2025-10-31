# src/data/loader_stealth.py

import os
import subprocess


base_path = "../../data/raw/sdo"
os.makedirs(base_path, exist_ok=True)

name = "sdobenchmark"
handle = "fhnw-i4ds/sdobenchmark"


print(f"\n--- Downloading dataset: '{name}' ---")
try:
    # Construct the Kaggle CLI command
    cmd = [
        "kaggle", "datasets", "download",
        "-d", handle,            # dataset handle
        "-p", base_path,         # download path
        "--unzip"                # unzip after download
    ]

    # Runing the command
    subprocess.run(cmd, check=True)
    print(f"Successfully downloaded and extracted '{name}' into: {os.path.abspath(base_path)}")

except subprocess.CalledProcessError as e:
    print(f"Failed to download '{name}'. Kaggle CLI returned an error:\n{e}")
except Exception as e:
    print(f"Unexpected error for '{name}': {e}")

print()