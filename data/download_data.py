# data/download_data.py
from huggingface_hub import HfFileSystem
import pandas as pd
import json
import os

os.makedirs("data/raw", exist_ok=True)

CATEGORIES = ["Electronics", "Books", "Clothing_Shoes_and_Jewelry"]
SAMPLES_PER_CATEGORY = 50000

fs = HfFileSystem()
all_reviews = []

for category in CATEGORIES:
    print(f"Downloading {category}...")
    path = f"datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/{category}.jsonl"

    rows = []
    with fs.open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= SAMPLES_PER_CATEGORY:
                break
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    df = df[["rating", "title", "text", "asin", "timestamp"]].copy()
    df["category"] = category
    df = df.dropna(subset=["text"])
    all_reviews.append(df)
    print(f"  Got {len(df):,} reviews")

combined = pd.concat(all_reviews, ignore_index=True)
combined.to_csv("data/raw/reviews_raw.csv", index=False)
print(f"\nTotal reviews saved: {len(combined):,}")
print("Saved to data/raw/reviews_raw.csv")