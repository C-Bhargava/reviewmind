# app/dashboard.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import snowflake.connector

load_dotenv()

st.set_page_config(
    page_title="ReviewMind",
    layout="wide"
)

st.title("ReviewMind — Review Intelligence Platform")
st.caption("RAG-powered insights from 150k Amazon reviews · Snowflake + Pinecone + Claude")

# ── Snowflake connection ────────────────────────────────────────────
@st.cache_resource
def get_conn():
    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database="REVIEWMIND"
    )

def run_query(sql):
    cur = get_conn().cursor()
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    return pd.DataFrame(cur.fetchall(), columns=cols)

tab1, tab2, tab3 = st.tabs([
    "Ask the reviews",
    "Sentiment insights",
    "Pipeline history"
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — RAG Agent
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Ask a question about customer reviews")
    st.write("Questions are answered using real retrieved reviews — not AI guesses.")

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input(
            "Your question",
            placeholder="What do customers complain about most with electronics?"
        )
    with col2:
        category = st.selectbox(
            "Filter by category",
            ["All", "Electronics", "Books", "Clothing Shoes and Jewelry"]
        )

    # Quick-pick buttons
    st.caption("Try these:")
    examples = [
        "What are the most common complaints?",
        "What makes customers give 5 stars?",
        "What quality issues come up most often?",
        "What do customers say about shipping?"
    ]
    cols = st.columns(4)
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state["q"] = ex

    question = st.session_state.get("q", question)

    if st.button("Ask agent", type="primary") and question:
        with st.spinner("Searching reviews and generating answer..."):
            from agents.rag_agent import ask
            cat_filter = None if category == "All" else category
            result = ask(question, category_filter=cat_filter)

        st.success(f"Answer (based on {result['reviews_used']} retrieved reviews)")
        st.write(result["answer"])

        if result.get("top_reviews"):
            with st.expander("Top matching reviews from Pinecone"):
                for i, r in enumerate(result["top_reviews"], 1):
                    st.markdown(f"**Review {i}** — {r['category']} | "
                                f"{r['rating']}★ | {r['sentiment_label']} "
                                f"(similarity: {r['score']})")
                    st.caption(r["review_text"][:300] + "...")
                    st.divider()

# ══════════════════════════════════════════════════════════════════
# TAB 2 — Sentiment insights
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Sentiment analysis across categories")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Sentiment distribution by category**")
        df_sent = run_query("""
            SELECT category, sentiment_label, COUNT(*) AS cnt
            FROM REVIEWMIND.MARTS.REVIEWS_PROCESSED
            GROUP BY category, sentiment_label
            ORDER BY category, sentiment_label
        """)
        if not df_sent.empty:
            pivot = df_sent.pivot(
                index="CATEGORY",
                columns="SENTIMENT_LABEL",
                values="CNT"
            ).fillna(0)
            st.bar_chart(pivot)

    with col2:
        st.markdown("**Average sentiment score by category**")
        df_avg = run_query("""
            SELECT
                category,
                ROUND(AVG(sentiment_score), 3) AS avg_sentiment,
                ROUND(AVG(rating), 2)           AS avg_rating,
                COUNT(*)                        AS review_count
            FROM REVIEWMIND.MARTS.REVIEWS_PROCESSED
            GROUP BY category
            ORDER BY avg_sentiment DESC
        """)
        if not df_avg.empty:
            st.dataframe(df_avg, use_container_width=True)
            st.bar_chart(df_avg.set_index("CATEGORY")["AVG_SENTIMENT"])

    st.divider()
    st.markdown("**Rating bucket breakdown**")
    df_bucket = run_query("""
        SELECT rating_bucket, category, COUNT(*) AS cnt
        FROM REVIEWMIND.MARTS.REVIEWS_PROCESSED
        GROUP BY rating_bucket, category
        ORDER BY rating_bucket, category
    """)
    if not df_bucket.empty:
        pivot2 = df_bucket.pivot(
            index="RATING_BUCKET",
            columns="CATEGORY",
            values="CNT"
        ).fillna(0)
        st.bar_chart(pivot2)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — Pipeline history
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Pipeline run history")
    st.write("Every Airflow DAG run is logged to Snowflake.")

    df_log = run_query("""
        SELECT
            run_id,
            TO_CHAR(run_ts, 'YYYY-MM-DD HH24:MI') AS run_time,
            stage,
            records_in,
            records_out,
            status,
            notes
        FROM REVIEWMIND.PIPELINE.RUN_LOG
        ORDER BY run_ts DESC
        LIMIT 20
    """)

    if not df_log.empty:
        # Color status column
        def color_status(val):
            return "color: green" if val == "success" else "color: red"

        st.dataframe(
            df_log.style.applymap(color_status, subset=["STATUS"]),
            use_container_width=True
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Total runs", len(df_log))
        col2.metric("Successful",
                    len(df_log[df_log["STATUS"] == "success"]))
        col3.metric("Avg records out",
                    f"{df_log['RECORDS_OUT'].mean():,.0f}")
    else:
        st.info("No pipeline runs logged yet. Trigger the Airflow DAG first.")