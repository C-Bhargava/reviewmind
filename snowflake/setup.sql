-- 1. Create database and schemas
CREATE DATABASE IF NOT EXISTS REVIEWMIND;
USE DATABASE REVIEWMIND;

CREATE SCHEMA IF NOT EXISTS RAW;
CREATE SCHEMA IF NOT EXISTS MARTS;
CREATE SCHEMA IF NOT EXISTS PIPELINE;

-- 2. Raw reviews table (loaded from CSV/PySpark output)
CREATE OR REPLACE TABLE RAW.REVIEWS (
    review_id     VARCHAR DEFAULT UUID_STRING(),
    rating        FLOAT,
    title         VARCHAR,
    review_text   VARCHAR,
    asin          VARCHAR,
    review_ts     TIMESTAMP_NTZ,
    category      VARCHAR,
    _loaded_at    TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- 3. Processed reviews mart (PySpark writes here)
CREATE OR REPLACE TABLE MARTS.REVIEWS_PROCESSED (
    review_id         VARCHAR,
    asin              VARCHAR,
    category          VARCHAR,
    rating            FLOAT,
    review_length     INTEGER,
    sentiment_score   FLOAT,
    sentiment_label   VARCHAR,
    rating_bucket     VARCHAR,
    word_count        INTEGER,
    processed_at      TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- 4. Pipeline run log
CREATE OR REPLACE TABLE PIPELINE.RUN_LOG (
    run_id          NUMBER AUTOINCREMENT PRIMARY KEY,
    run_ts          TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    stage           VARCHAR,
    records_in      INTEGER,
    records_out     INTEGER,
    status          VARCHAR,
    notes           VARCHAR
);

-- 5. Verify
SHOW SCHEMAS IN DATABASE REVIEWMIND;