# agents/rag_agent.py
import os, json
import anthropic
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── Load model and clients once (module-level) ───────────────────────
_model    = None
_index    = None
_client   = None

def _get_resources():
    global _model, _index, _client
    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    if _index is None:
        pc     = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        _index = pc.Index(os.getenv("PINECONE_INDEX", "reviewmind-index"))
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _model, _index, _client

# ── Step 1: Embed the question ────────────────────────────────────────
def embed_query(question: str) -> list:
    model, _, _ = _get_resources()
    return model.encode(question).tolist()

# ── Step 2: Retrieve relevant reviews from Pinecone ───────────────────
def retrieve_reviews(question: str, top_k: int = 15,
                     category_filter: str = None) -> list:
    _, index, _ = _get_resources()
    query_vec   = embed_query(question)

    filter_dict = {}
    if category_filter and category_filter != "All":
        filter_dict["category"] = {"$eq": category_filter}

    results = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict if filter_dict else None
    )

    reviews = []
    for match in results["matches"]:
        meta = match["metadata"]
        reviews.append({
            "score":           round(match["score"], 3),
            "review_text":     meta.get("review_text", ""),
            "category":        meta.get("category", ""),
            "rating":          meta.get("rating", 0),
            "sentiment_label": meta.get("sentiment_label", ""),
            "sentiment_score": meta.get("sentiment_score", 0),
            "asin":            meta.get("asin", "")
        })
    return reviews

# ── Step 3: Ask Claude with retrieved context ─────────────────────────
def answer_with_claude(question: str, reviews: list) -> str:
    _, _, client = _get_resources()

    context = "\n\n".join([
        f"Review {i+1} [{r['category']} | {r['rating']}★ | {r['sentiment_label']}]:\n{r['review_text']}"
        for i, r in enumerate(reviews)
    ])

    prompt = f"""You are a product insights analyst for an e-commerce company.
You have been given {len(reviews)} real customer reviews retrieved from a vector database.
Use ONLY these reviews to answer the question — do not use outside knowledge.
If the reviews don't contain enough information, say so.

RETRIEVED REVIEWS:
{context}

QUESTION: {question}

Provide a clear, structured answer with:
1. A direct answer to the question
2. Key themes you found (with review count supporting each theme)
3. 2-3 verbatim quotes from the reviews that best illustrate your points
4. A confidence level: High / Medium / Low based on how well the reviews answer the question"""

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text

# ── Full pipeline ─────────────────────────────────────────────────────
def ask(question: str, category_filter: str = None,
        top_k: int = 15) -> dict:
    reviews  = retrieve_reviews(question, top_k, category_filter)
    answer   = answer_with_claude(question, reviews)
    return {
        "question":      question,
        "answer":        answer,
        "reviews_used":  len(reviews),
        "top_reviews":   reviews[:3],   # return top 3 for display
        "category_filter": category_filter
    }

# ── Demo ──────────────────────────────────────────────────────────────
DEMO_QUESTIONS = [
    ("What do customers complain about most with electronics?", "Electronics"),
    ("What makes customers give 5-star reviews for books?",    "Books"),
    ("What are the most common quality issues in clothing?",   "Clothing Shoes and Jewelry"),
    ("What words do customers use to describe bad experiences?", None),
    ("Which product issues lead to the most negative reviews?", None),
]

if __name__ == "__main__":
    print("\n=== ReviewMind RAG Agent ===\n")
    for question, category in DEMO_QUESTIONS:
        print(f"Q: {question}")
        if category:
            print(f"   [filtered to: {category}]")
        result = ask(question, category_filter=category)
        print(f"\nA: {result['answer']}")
        print(f"   (used {result['reviews_used']} retrieved reviews)")
        print("\n" + "="*60 + "\n")