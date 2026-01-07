
import os
import requests
import pandas as pd
from tqdm import tqdm

# ===============================
# CONFIGURATION
# ===============================
ZOOM = 18
IMAGE_SIZE = "256x256"

# Read Mapbox token securely
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")

if MAPBOX_TOKEN is None:
    raise ValueError("MAPBOX_TOKEN not found. Set it using os.environ['MAPBOX_TOKEN'].")

# ===============================
# FUNCTIONS
# ===============================
def fetch_image(lat, lon, save_path, zoom=ZOOM):
    url = (
        f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
        f"{lon},{lat},{zoom}/{IMAGE_SIZE}"
        f"?access_token={MAPBOX_TOKEN}"
    )

    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
    else:
        print(f"Failed for {lat},{lon} | Status: {response.status_code}")

def download_images(excel_path, img_dir):
    os.makedirs(img_dir, exist_ok=True)
    df = pd.read_excel(excel_path)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(img_dir, f"{row['id']}.png")
        if not os.path.exists(img_path):
            fetch_image(row["lat"], row["long"], img_path)

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    base = "/content/drive/MyDrive/satellite_property"

    download_images(
        f"{base}/data/raw/train.xlsx",
        f"{base}/data/images/train"
    )

    download_images(
        f"{base}/data/raw/test.xlsx",
        f"{base}/data/images/test"
    )
