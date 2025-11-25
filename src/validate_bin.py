import os
import json
import csv
import random
from collections import Counter
from inference import analyze_image

# ================= CONFIG =================
BASE_DIR = os.getcwd()
SUBSET_DIR = os.path.join(BASE_DIR, "subset")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
MASTER_CSV = os.path.join(PROCESSED_DIR, "master_metadata.csv")
ASIN_TEXT = os.path.join(PROCESSED_DIR, "asin_text.json") 
ASIN_TEXT_SMALL = os.path.join(PROCESSED_DIR, "asin_text_small.json")

# CPU Mode: Test 50 images
NUM_TEST_IMAGES = 50 

# ================= LOAD DATA =================
print("Loading Text Descriptions...")
target_file = ASIN_TEXT if os.path.exists(ASIN_TEXT) else ASIN_TEXT_SMALL
with open(target_file, "r", encoding="utf-8") as f:
    asin_map = json.load(f)

print("Loading Ground Truth...")
ground_truth_db = {} # Maps Image -> {Description: Qty}

try:
    with open(MASTER_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["image"]:
                # Parse ASINs
                raw_asins = row["asins"].replace(";", ",")
                asins = [a.strip() for a in raw_asins.split(",") if a.strip()]
                
                # Parse Quantities
                qtys = [1] * len(asins) # Default to 1
                
                # Convert ASINs to DESCRIPTIONS (Critical Step)
                order_dict = Counter()
                for i, asin in enumerate(asins):
                    # Get the English name, or skip if unknown
                    desc = asin_map.get(asin)
                    if desc:
                        order_dict[desc] += qtys[i]
                
                if order_dict:
                    ground_truth_db[row["image"]] = order_dict

except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()

# ================= VALIDATION LOOP =================
all_images = sorted([f for f in os.listdir(SUBSET_DIR) if f.lower().endswith((".jpg", ".png"))])
random.seed(42)
random.shuffle(all_images)
test_images = all_images[:NUM_TEST_IMAGES]

print(f"\nStarting Zero-Shot Validation on {len(test_images)} images...")
print("-" * 60)

total_items_expected = 0
total_items_verified = 0
perfect_bins = 0

for i, img_name in enumerate(test_images):
    img_path = os.path.join(SUBSET_DIR, img_name)
    gt_order = ground_truth_db.get(img_name)

    if not gt_order: continue

    # 1. EXTRACT DESCRIPTIONS
    target_descriptions = list(gt_order.keys())
    
    # 2. CALL NEW INFERENCE (Passes Text, not IDs)
    prediction_counts = analyze_image(img_path, target_descriptions=target_descriptions)

    # 3. SCORE IT
    matches = 0
    for desc, expected_qty in gt_order.items():
        # Check if we found it (Count > 0)
        found_qty = prediction_counts.get(desc, 0)
        
        # We cap the found quantity at the expected quantity 
        # (Finding 6 when 1 was ordered counts as 1 verified, not 6)
        valid_found = min(found_qty, expected_qty)
        matches += valid_found
    
    total_items_expected += sum(gt_order.values())
    total_items_verified += matches
    
    is_perfect = (matches == sum(gt_order.values()))
    if is_perfect: perfect_bins += 1

    print(f"[{i+1}] {img_name} | Expected: {sum(gt_order.values())} | Verified: {matches} | {'✅' if is_perfect else '⚠️'}")

# ================= REPORT =================
recall = (total_items_verified / total_items_expected) * 100 if total_items_expected > 0 else 0

print("\n" + "="*40)
print("FINAL RESULTS (PURE ZERO-SHOT)")
print("="*40)
print(f"Images Tested       : {len(test_images)}")
print(f"Total Items Ordered : {total_items_expected}")
print(f"Total Items Verified: {total_items_verified}")
print(f"Item Recall Rate    : {recall:.2f}%")
print(f"Perfect Bins        : {perfect_bins}")
print("="*40)