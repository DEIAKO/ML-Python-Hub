import os
import shutil
import kagglehub

def download_zomato_dataset(target_path="restaurant_data.csv"):
    os.environ["KAGGLE_USERNAME"] = os.environ.get("KAGGLE_USERNAME", "")
    os.environ["KAGGLE_KEY"] = os.environ.get("KAGGLE_KEY", "")
    path = kagglehub.dataset_download("shrutimehta/zomato-restaurants-data")
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".csv"):
                src = os.path.join(root, f)
                shutil.copy(src, target_path)
                print(f"Dataset saved to: {target_path}")
                return target_path
    raise FileNotFoundError("No CSV file found in downloaded dataset.")

if __name__ == "__main__":
    download_zomato_dataset()
