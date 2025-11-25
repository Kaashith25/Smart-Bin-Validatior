import json
import pandas as pd

META_FILE = "processed/master_metadata.csv"
FULL_ASIN_FILE = "processed/asin_text.json"
SMALL_ASIN_FILE = "processed/asin_text_small.json"

print("Loading metadata...")
df = pd.read_csv(META_FILE)

all_asins = []

for a in df["asins"]:
    if pd.isna(a):
        continue
    all_asins.extend(a.split(";"))

top = pd.Series(all_asins).value_counts().head(300).index.tolist()

print("Top ASINs selected:", len(top))

full = json.load(open(FULL_ASIN_FILE, "r", encoding="utf-8"))

small = {k: full[k] for k in top if k in full}

json.dump(small, open(SMALL_ASIN_FILE, "w", encoding="utf-8"), indent=2)

print("Created:", SMALL_ASIN_FILE)
print("Total ASINs in small file:", len(small))
