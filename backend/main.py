"""
main.py — RAG chain with RAGAS-optimised prompt and reranking.

"""

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vector import retriever, vector_store
import logging
import time
from sentence_transformers import CrossEncoder

_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# MODEL
model = OllamaLLM(
    model="llama3.2",
    temperature=0.0,
    num_predict=400,
    repeat_penalty=1.2,
)


# PROMPT
SYSTEM_PROMPT = """You are a compassionate mental health support assistant.
Answer the user's question using ONLY the context provided below.
Be direct, concise, and stay strictly on topic.
If the context is insufficient, say: "I don't have enough information on that."

Context:
STRICT RULES — follow every rule without exception:
1. Answer ONLY using information from the context passages provided below.
2. Address every part of the question — do not skip any aspect asked.
3. Use ALL relevant information across the context passages, not just the first one.
4. Write 2–5 sentences. Be clear and complete, not padded.
5. Do NOT use phrases like "Based on the context" or "According to the context".
6. Do NOT say you cannot answer — always use the closest relevant information.
7. Always respond in plain English regardless of the language of the question.
8. Be empathetic and supportive in tone.
9. Never provide medical diagnoses or prescriptions.
10. Never recommend stopping prescribed medication or professional treatment."""

template = """{system}

Context passages:
{context}

Question: {query}

Response:"""

prompt = ChatPromptTemplate.from_template(template)


# HELPERS
def format_docs(docs) -> str:
    """
    Convert retrieved Document objects into a clean numbered string.
    Deduplicates identical content to reduce prompt noise.
    Numbers kept so RAGAS faithfulness judge can trace claims.
    """
    seen   = set()
    chunks = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content.strip()
        if content in seen:
            continue
        seen.add(content)
        chunks.append(f"[{i}] {content}")
    return "\n\n".join(chunks) if chunks else "No relevant context found."


def rerank_docs(query: str, docs, threshold: float = 0.0, top_k: int = 6):
    """
    Re-score retrieved docs via CrossEncoder, filter by threshold,
    return top_k. Falls back to first 3 docs if reranker fails.
    """
    if not docs:
        return []
    try:
        pairs  = [(query, d.page_content) for d in docs]
        scores = _reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        results = [doc for score, doc in ranked if score > threshold][:top_k]
        return results if results else docs[:3]
    except Exception as e:
        logger.warning("CrossEncoder reranking failed (%s) — using raw docs.", e)
        return docs[:3]


def is_safe_query(query: str) -> bool:
    """Reject empty or trivially short queries."""
    #return bool(query and query.strip() and len(query.strip()) >= 3)
    return bool(query and query.strip())

# CORE PIPELINE  (used by both /chat and /chat/stream)
def _build_prompt_inputs(query: str) -> dict:
    """
    Single place that does retrieval → rerank → format.
    Returns the dict the prompt template needs.
    FIX: this is called ONCE per request, shared by both buffered and streaming paths.
    """
    raw_docs = retriever.invoke(query)
    reranked = rerank_docs(query, raw_docs)
    context  = format_docs(reranked)
    return {
        "system":  SYSTEM_PROMPT,
        "context": context,
        "query":   query,
    }


# PUBLIC API  (imported by app.py)
def get_response(query: str) -> str:
    """
    Buffered response — returns the full answer string.
    app.py calls this for /chat.
    """
    inputs = _build_prompt_inputs(query)
    chain  = prompt | model | StrOutputParser()
    return chain.invoke(inputs)


def stream_response(query: str):
    """
    Generator that yields tokens as they arrive.
    app.py calls this for /chat/stream.
    """
    inputs = _build_prompt_inputs(query)
    chain  = prompt | model | StrOutputParser()
    yield from chain.stream(inputs)