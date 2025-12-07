import os
import subprocess
import zipfile

def setup_data():
    dataset_slug = "quannguyend/preprocessed-data"
    zip_name = "preprocessed-data.zip"
    
    if os.path.exists("chaos_MR_T2_normalized") and os.path.exists("sabs_CT_normalized"):
        print("✅ Data directories already appear to exist.")
        return

    if not os.path.exists(zip_name):
        print(f"Downloading {dataset_slug}...")
        try:
            subprocess.run(["kaggle", "datasets", "download", dataset_slug], check=True)
        except subprocess.CalledProcessError:
            print("❌ Failed to download data via Kaggle API.")
            print("Please ensure you have 'kaggle' installed and 'kaggle.json' configured.")
            return
        except FileNotFoundError:
            print("❌ 'kaggle' command not found. Please install kaggle CLI.")
            return

    if os.path.exists(zip_name):
        print(f"Unzipping {zip_name}...")
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("✅ Data extraction complete.")
    else:
        print("❌ Zip file not found after download attempt.")

if __name__ == "__main__":
    setup_data()
