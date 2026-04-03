"""
debug.py — run this BEFORE evaluating to diagnose RAGAS issues.

Checks:
  1. Retrieved docs (content + metadata)
  2. Relevance scores (detect if threshold is too aggressive)
  3. Context usability (word counts — low = faithfulness NaN risk)
  4. Judge LLM output (detect if it can produce Yes/No)
  5. LLM response quality
"""

from vector import retriever, vector_store
from main import get_response, rerank_docs, format_docs
from langchain_ollama.llms import OllamaLLM

query = "How do I manage anxiety?"

print("=" * 60)
print("=== 1. RETRIEVED DOCS (MMR retriever) ===")
print("=" * 60)
docs = retriever.invoke(query)
print(f"Retrieved {len(docs)} docs\n")
for i, doc in enumerate(docs):
    words = len(doc.page_content.split())
    print(f"[{i+1}] ({words} words) {doc.page_content[:300]}")
    print(f"     Metadata: {doc.metadata}")

print("\n" + "=" * 60)
print("=== 2. RELEVANCE SCORES (similarity search) ===")
print("=" * 60)
docs_with_scores = vector_store.similarity_search_with_relevance_scores(query, k=8)
for doc, score in docs_with_scores:
    flag = "✓" if score >= 0.10 else "✗ (below threshold)"
    print(f"Score: {score:.4f} {flag} | {doc.page_content[:120]}")

print("\n" + "=" * 60)
print("=== 3. RERANKED DOCS (after threshold filter) ===")
print("=" * 60)
reranked = rerank_docs(query, docs, threshold=0.10)
print(f"Docs after reranking: {len(reranked)}")
total_words = sum(len(d.page_content.split()) for d in reranked)
avg_words   = total_words / len(reranked) if reranked else 0
print(f"Total context words:  {total_words}")
print(f"Avg words per chunk:  {avg_words:.1f}  (target: >20 per chunk)")
if avg_words < 10:
    print("  ⚠️  Context chunks are very short — faithfulness=NaN risk is HIGH.")
    print("     Check your CSV 'answers' column for short/missing values.")

print("\n" + "=" * 60)
print("=== 4. FORMATTED CONTEXT (what LLM sees) ===")
print("=" * 60)
formatted = format_docs(reranked)
print(formatted[:800])

print("\n" + "=" * 60)
print("=== 5. JUDGE LLM SANITY CHECK ===")
print("=" * 60)
judge_prompt = (
    "Given the following context and statement, answer only 'Yes' or 'No'.\n"
    "Context: Anxiety can be managed through deep breathing exercises.\n"
    "Statement: Deep breathing helps with anxiety.\n"
    "Answer:"
)
judge_llm = OllamaLLM(model="llama3.2", temperature=0.0, num_predict=10)
judge_out  = judge_llm.invoke(judge_prompt)
print(f"Judge prompt → '{judge_out.strip()}'")
if "yes" in judge_out.lower() or "no" in judge_out.lower():
    print("✓ Judge LLM is producing parseable Yes/No — faithfulness should compute.")
else:
    print("✗ Judge LLM did NOT return Yes/No — faithfulness and context_precision will be NaN.")
    print("  Fix: try a larger model as judge, e.g. OllamaLLM(model='llama3.1:8b')")

print("\n" + "=" * 60)
print("=== 6. LLM RESPONSE ===")
print("=" * 60)
response = get_response(query)
print(response)
word_count = len(response.split())
print(f"\nResponse words: {word_count}  (target: 30–80 for good answer_relevancy)")
if word_count < 15:
    print("  ⚠️  Response too short — answer_relevancy will likely be low.")
if word_count > 150:
    print("  ⚠️  Response very long — may include off-topic content, hurting relevancy.")