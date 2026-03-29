# pipeline/embed_and_index.py
import os, time
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# ── 1. Load processed reviews ────────────────────────────────────────
print("Loading processed reviews...")
df = pd.read_parquet("data/processed/reviews_processed.parquet")

# Use a subset for Pinecone free tier (100k vector limit)
# Sample 20k — balanced across categories and sentiments
df_sample = df.groupby(["category","sentiment_label"], group_keys=False) \
              .apply(lambda x: x.sample(min(len(x), 2000), random_state=42))
df_sample = df_sample.reset_index(drop=True)
print(f"  Embedding {len(df_sample):,} reviews")

# ── 2. Load embedding model ───────────────────────────────────────────
print("Loading embedding model (downloads once ~90MB)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
# Produces 384-dimensional vectors, fast on CPU, great quality

# ── 3. Generate embeddings in batches ────────────────────────────────
print("Generating embeddings...")
texts = df_sample["review_text"].fillna("").tolist()
embeddings = model.encode(
    texts,
    batch_size=128,
    show_progress_bar=True,
    convert_to_numpy=True
)
print(f"  Generated {len(embeddings):,} embeddings of dim {embeddings.shape[1]}")

# ── 4. Connect to Pinecone ────────────────────────────────────────────
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX", "reviewmind-index"))

# ── 5. Upload in batches of 200 ───────────────────────────────────────
print("Uploading to Pinecone...")
BATCH_SIZE = 200
total_uploaded = 0

for i in range(0, len(df_sample), BATCH_SIZE):
    batch_df  = df_sample.iloc[i:i+BATCH_SIZE]
    batch_emb = embeddings[i:i+BATCH_SIZE]

    vectors = []
    for j, (_, row) in enumerate(batch_df.iterrows()):
        vectors.append({
            "id": f"rev-{i+j:07d}",
            "values": batch_emb[j].tolist(),
            "metadata": {
                "category":        str(row.get("category", "")),
                "rating":          float(row.get("rating", 0)),
                "sentiment_label": str(row.get("sentiment_label", "")),
                "sentiment_score": float(row.get("sentiment_score", 0)),
                "rating_bucket":   str(row.get("rating_bucket", "")),
                "review_text":     str(row.get("review_text", ""))[:500],
                "asin":            str(row.get("asin", ""))
            }
        })

    index.upsert(vectors=vectors)
    total_uploaded += len(vectors)

    if (i // BATCH_SIZE) % 10 == 0:
        print(f"  Uploaded {total_uploaded:,} / {len(df_sample):,}")
    time.sleep(0.1)  # be gentle with the API

print(f"\nDone. Total vectors in Pinecone: {total_uploaded:,}")
stats = index.describe_index_stats()
print(f"Index stats: {stats}")