# src/data/preprocess_sdo.py

import pandas as pd
import os


input_dir = "../../data/raw/sdo/sdobenchmark_full/"
train_file = "../../data/raw/sdo/sdobenchmark_full/training/meta_data.csv"
train_dir = "../../data/raw/sdo/sdobenchmark_full/training/"
test_file = "../../data/raw/sdo/sdobenchmark_full/test/meta_data.csv"
test_dir = "../../data/raw/sdo/sdobenchmark_full/test/"
output_dir = "../../data/processed/sdo/"

os.makedirs(output_dir, exist_ok=True)


def get_flare_label(peak_flux):
    if peak_flux >= 1e-4: return 'X'
    if peak_flux >= 1e-5: return 'M'
    if peak_flux >= 1e-6: return 'C'
    return 'Q'

def extract_time_id(row):
    ar_id_str = str(row['ar_num'])
    full_id_str = row['id']
    
    time_id = full_id_str.split(ar_id_str + '_', 1)[1]
    return time_id

def get_image_path(row):
    ar_id = str(row["ar_num"])
    time_id = row["id"].split(ar_id+"_")[1]
    
    save_dir = os.path.join(train_dir, ar_id, time_id)
    if os.path.isdir(save_dir):
        return save_dir

    save_dir = os.path.join(test_dir, ar_id, time_id)
    if os.path.isdir(save_dir):
        return save_dir

    return None

# df["location"] = df.apply(get_image_path, axis=1)
# df


def verify_dataset_integrity(df, root_dir):
    print(f"\nScanning file system under '{root_dir}'...")
    filesystem_ids = set()
    
    for ar_folder in os.listdir(root_dir):
        ar_path = os.path.join(root_dir, ar_folder)
        if os.path.isdir(ar_path):
            for sample_folder in os.listdir(ar_path):
                filesystem_ids.add(os.path.join(ar_folder, sample_folder))
    print(f"Found {len(filesystem_ids)} folders in the file system and we have {len(df["location"])} folder locations in metadata")
    
    meta_ids = set()
    for loc in list(df["location"]):
        meta_ids.add(os.path.join(loc.split("/")[-2], loc.split("/")[-1]))

    extra_folders = filesystem_ids - meta_ids
    if len(extra_folders) == 0: 
        print("No extra folders found")
        return
    
    print("The following extra folders found\n")
    for folder in extra_folders:
        print(folder)

# verify_dataset_integrity(df, train_dir)

# Preprocessing Training Dataset
df = pd.read_csv(train_file)
df['start'] = pd.to_datetime(df['start'])
df['end'] = pd.to_datetime(df['end'])
df['ar_num'] = df['id'].apply(lambda x: int(x.split('_')[0]))
df["time_id"] = df.apply(extract_time_id, axis =1)
df['label'] = df['peak_flux'].apply(get_flare_label)
df["location"] = df.apply(get_image_path, axis=1)

train_df = df[["ar_num", "time_id", "start", "end", "peak_flux", "label", "location"]].copy()

verify_dataset_integrity(train_df, train_dir)
train_df.to_csv(os.path.join(output_dir, "train_metadata.csv"))

# Preprocessing Test Dataset
df = pd.read_csv(test_file)
df['start'] = pd.to_datetime(df['start'])
df['end'] = pd.to_datetime(df['end'])
df['ar_num'] = df['id'].apply(lambda x: int(x.split('_')[0]))
df["time_id"] = df.apply(extract_time_id, axis =1)
df['label'] = df['peak_flux'].apply(get_flare_label)
df["location"] = df.apply(get_image_path, axis=1)

test_df = df[["ar_num", "time_id", "start", "end", "peak_flux", "label", "location"]].copy()

verify_dataset_integrity(test_df, test_dir)
test_df.to_csv(os.path.join(output_dir, "test_metadata.csv"))

