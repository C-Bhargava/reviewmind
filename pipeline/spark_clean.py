# pipeline/spark_clean.py
import os
import sys
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType, StringType
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── 1. Start Spark session ───────────────────────────────────────────
spark = SparkSession.builder \
    .appName("ReviewMind-Clean") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("Spark session started.")

# ── 2. Load raw CSV ──────────────────────────────────────────────────
df = spark.read.csv(
    "data/raw/reviews_raw.csv",
    header=True,
    inferSchema=True,
    multiLine=True,
    escape='"'
)
raw_count = df.count()
print(f"Loaded {raw_count:,} raw reviews")

# ── 3. Clean ─────────────────────────────────────────────────────────
df = df.dropna(subset=["text"])
df = df.filter(F.length(F.trim(F.col("text"))) > 10)

# Standardize column names
df = df.withColumnRenamed("text", "review_text") \
       .withColumnRenamed("title", "review_title")

# Clean text: strip extra whitespace
df = df.withColumn("review_text",
    F.regexp_replace(F.trim(F.col("review_text")), r"\s+", " ")
)

# Standardize category
df = df.withColumn("category",
    F.regexp_replace(F.col("category"), "_", " ")
)

# ── 4. Feature engineering ───────────────────────────────────────────
df = df.withColumn("review_length",
        F.length(F.col("review_text")).cast(IntegerType())) \
       .withColumn("word_count",
        F.size(F.split(F.col("review_text"), " ")).cast(IntegerType())) \
       .withColumn("has_title",
        F.when(F.col("review_title").isNotNull() &
               (F.length(F.trim(F.col("review_title"))) > 0),
               True).otherwise(False)) \
       .withColumn("rating_bucket",
        F.when(F.col("rating") <= 2, "low")
         .when(F.col("rating") == 3,  "mid")
         .otherwise("high"))

# ── 5. Sentiment scoring via UDF ─────────────────────────────────────
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    if not text:
        return 0.0
    return float(analyzer.polarity_scores(text)["compound"])

def get_sentiment_label(score):
    if score is None:
        return "neutral"
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"

sentiment_score_udf = F.udf(get_sentiment_score, FloatType())
sentiment_label_udf = F.udf(get_sentiment_label, StringType())

print("Scoring sentiment (this takes ~2 min for 150k rows)...")
df = df.withColumn("sentiment_score",
        sentiment_score_udf(F.col("review_text"))) \
       .withColumn("sentiment_label",
        sentiment_label_udf(F.col("sentiment_score")))

# ── 6. Select final columns ───────────────────────────────────────────
df_final = df.select(
    F.monotonically_increasing_id().cast(StringType()).alias("row_id"),
    F.col("asin"),
    F.col("category"),
    F.col("rating").cast(FloatType()),
    F.col("review_title"),
    F.col("review_text"),
    F.col("review_length"),
    F.col("word_count"),
    F.col("has_title"),
    F.col("rating_bucket"),
    F.col("sentiment_score"),
    F.col("sentiment_label"),
    F.col("timestamp")
)

clean_count = df_final.count()
print(f"Clean rows: {clean_count:,}  (dropped {raw_count - clean_count:,})")

# ── 7. Write Parquet ──────────────────────────────────────────────────
print("\nSentiment distribution:")
df_final.groupBy("sentiment_label") \
        .count() \
        .orderBy("count", ascending=False) \
        .show()

print("Avg sentiment score by category:")
df_final.groupBy("category") \
        .agg(
            F.round(F.avg("sentiment_score"), 3).alias("avg_sentiment"),
            F.round(F.avg("rating"), 2).alias("avg_rating"),
            F.count("*").alias("review_count")
        ) \
        .orderBy("avg_sentiment", ascending=False) \
        .show()

os.makedirs("data/processed", exist_ok=True)
pdf = df_final.toPandas()
pdf.to_parquet("data/processed/reviews_processed.parquet", index=False)
print("Written to data/processed/reviews_processed.parquet")

spark.stop()
print("Spark job complete.")