# pipeline/load_processed.py
import os
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from dotenv import load_dotenv

load_dotenv()

print("Loading processed reviews into Snowflake...")

# Read PySpark Parquet output
df = pd.read_parquet("data/processed/reviews_processed.parquet")
print(f"  Rows from Parquet: {len(df):,}")

# Rename columns to match Snowflake table schema
df = df.rename(columns={
    "row_id":          "REVIEW_ID",
    "asin":            "ASIN",
    "category":        "CATEGORY",
    "rating":          "RATING",
    "review_length":   "REVIEW_LENGTH",
    "sentiment_score": "SENTIMENT_SCORE",
    "sentiment_label": "SENTIMENT_LABEL",
    "rating_bucket":   "RATING_BUCKET",
    "word_count":      "WORD_COUNT"
})

df = df[[
    "REVIEW_ID","ASIN","CATEGORY","RATING",
    "REVIEW_LENGTH","SENTIMENT_SCORE",
    "SENTIMENT_LABEL","RATING_BUCKET","WORD_COUNT"
]]

# Load into Snowflake
conn = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database="REVIEWMIND",
    schema="MARTS"
)

success, nchunks, nrows, _ = write_pandas(
    conn, df, "REVIEWS_PROCESSED",
    database="REVIEWMIND", schema="MARTS",
    auto_create_table=False, overwrite=True
)
print(f"  Loaded: {nrows:,} rows in {nchunks} chunks")

# Log the pipeline run
conn.cursor().execute("""
    INSERT INTO REVIEWMIND.PIPELINE.RUN_LOG
        (stage, records_in, records_out, status, notes)
    VALUES (%s, %s, %s, %s, %s)
""", ("spark_clean_and_load", 150000, nrows, "success",
      "PySpark clean + sentiment + Snowflake load"))

conn.close()
print("Pipeline run logged.")
print("Done.")