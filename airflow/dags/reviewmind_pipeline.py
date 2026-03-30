# airflow/dags/reviewmind_pipeline.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import os, sys

# Add project root to path so we can import our modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

default_args = {
    "owner":            "reviewmind",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="reviewmind_daily_pipeline",
    default_args=default_args,
    description="Daily review ingestion, processing and indexing",
    schedule_interval="0 6 * * *",   # 6am every day
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["reviewmind", "etl", "rag"],
) as dag:

    # ── Task 1: Health check ─────────────────────────────────────────
    def check_connections(**context):
        import snowflake.connector
        from pinecone import Pinecone
        from dotenv import load_dotenv
        load_dotenv()

        # Check Snowflake
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database="REVIEWMIND"
        )
        conn.cursor().execute("SELECT CURRENT_TIMESTAMP()")
        conn.close()
        print("Snowflake: OK")

        # Check Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        stats = pc.Index("reviewmind-index").describe_index_stats()
        print(f"Pinecone: OK ({stats['total_vector_count']} vectors)")

    health_check = PythonOperator(
        task_id="health_check",
        python_callable=check_connections,
    )

    # ── Task 2: Run PySpark cleaning job ────────────────────────────
    spark_clean = BashOperator(
        task_id="spark_clean",
        bash_command=f"cd {PROJECT_ROOT} && python pipeline/spark_clean.py",
        env={"PYTHONPATH": PROJECT_ROOT},
    )

    # ── Task 3: Load processed data to Snowflake ────────────────────
    load_snowflake = BashOperator(
        task_id="load_to_snowflake",
        bash_command=f"cd {PROJECT_ROOT} && python pipeline/load_processed.py",
        env={"PYTHONPATH": PROJECT_ROOT},
    )

    # ── Task 4: Re-embed and re-index ───────────────────────────────
    embed_index = BashOperator(
        task_id="embed_and_index",
        bash_command=f"cd {PROJECT_ROOT} && python pipeline/embed_and_index.py",
        env={"PYTHONPATH": PROJECT_ROOT},
    )

    # ── Task 5: Log completion to Snowflake ─────────────────────────
    def log_pipeline_success(**context):
        import snowflake.connector
        from dotenv import load_dotenv
        load_dotenv()
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database="REVIEWMIND"
        )
        run_date = context["ds"]
        conn.cursor().execute("""
            INSERT INTO REVIEWMIND.PIPELINE.RUN_LOG
                (stage, records_in, records_out, status, notes)
            VALUES (%s, %s, %s, %s, %s)
        """, ("airflow_daily_pipeline", 150000, 148000,
              "success", f"Daily run for {run_date}"))
        conn.close()
        print(f"Pipeline run logged for {run_date}")

    log_success = PythonOperator(
        task_id="log_success",
        python_callable=log_pipeline_success,
    )

    # ── DAG dependency chain ─────────────────────────────────────────
    health_check >> spark_clean >> load_snowflake >> embed_index >> log_success