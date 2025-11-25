import os
import json
import csv
from tqdm import tqdm
from inference import analyze_image

# ================= CONFIG =================
BASE_DIR = os.getcwd()
SUBSET_DIR = os.path.join(BASE_DIR, "subset")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
MASTER_CSV = os.path.join(PROCESSED_DIR, "master_metadata.csv")
ASIN_TEXT = os.path.join(PROCESSED_DIR, "asin_text.json")
ASIN_TEXT_SMALL = os.path.join(PROCESSED_DIR, "asin_text_small.json")

os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load ASIN -> Description Mapping
# We need this to convert the CSV's ASIN codes into English text for the model
print("Loading text descriptions...")
target_file = ASIN_TEXT if os.path.exists(ASIN_TEXT) else ASIN_TEXT_SMALL
with open(target_file, "r", encoding="utf-8") as f:
    asin_map = json.load(f)

# 2. Load Metadata (Ground Truth)
print("Loading metadata...")
metadata_map = {}
with open(MASTER_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["image"]:
            # Clean up the ASIN list from the CSV
            raw_asins = row["asins"].replace(";", ",")
            asins = [a.strip() for a in raw_asins.split(",") if a.strip()]
            metadata_map[row["image"]] = asins

# 3. Run Inference on Subset
results = {}
# Get list of images in the subset folder
images = [f for f in os.listdir(SUBSET_DIR) if f.lower().endswith((".jpg",".png"))]

# Limit to 50 for submission artifact (Matches your validation run)
images = images[:50] 

print(f"Generating results for {len(images)} images...")

for img in tqdm(images):
    path = os.path.join(SUBSET_DIR, img)
    
    # A. Get the list of ASINs expected in this image
    target_asins = metadata_map.get(img, [])
    
    # B. Convert ASINs to Text Descriptions (CRITICAL FIX)
    target_descriptions = []
    for asin in target_asins:
        desc = asin_map.get(asin)
        if desc:
            target_descriptions.append(desc)
    
    # C. Run Inference
    # Note: Argument name is now 'target_descriptions'
    if target_descriptions:
        res = analyze_image(path, target_descriptions=target_descriptions)
        results[img] = dict(res)
    else:
        results[img] = {}

# 4. Save JSON
out_file = os.path.join(OUT_DIR, "subset_results.json")
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Done! Results saved to {out_file}")