# data/load_to_sf.py
import os, pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from dotenv import load_dotenv

load_dotenv()

print("Loading reviews into Snowflake RAW.REVIEWS...")

df = pd.read_csv("data/raw/reviews_raw.csv")

# Rename columns to match Snowflake table
df = df.rename(columns={
    "text":      "REVIEW_TEXT",
    "title":     "TITLE",
    "rating":    "RATING",
    "asin":      "ASIN",
    "timestamp": "REVIEW_TS",
    "category":  "CATEGORY"
})

# Add a review_id
df["REVIEW_ID"] = [f"REV-{i:07d}" for i in range(len(df))]

# Keep only table columns
df = df[["REVIEW_ID","RATING","TITLE","REVIEW_TEXT","ASIN","REVIEW_TS","CATEGORY"]]

# Convert timestamp
df["REVIEW_TS"] = pd.to_datetime(
    pd.to_numeric(df["REVIEW_TS"], errors="coerce"), unit="ms", errors="coerce"
).dt.strftime("%Y-%m-%d %H:%M:%S")

print(f"  Rows to load: {len(df):,}")

conn = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database="REVIEWMIND",
    schema="RAW"
)

success, nchunks, nrows, _ = write_pandas(
    conn, df, "REVIEWS",
    database="REVIEWMIND", schema="RAW",
    auto_create_table=False, overwrite=True
)

print(f"  Loaded: {nrows:,} rows in {nchunks} chunks")
conn.close()
print("Done.")