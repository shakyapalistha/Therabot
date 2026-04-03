"""
vector.py — speed-optimised build with pre-computed embeddings.

"""

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import shutil
import hashlib
import pandas as pd
import chromadb
from chromadb.config import Settings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# CONFIG
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(BASE_DIR, "..", "data")
DB_LOCATION      = "./chroma_langchain_db"
EMBED_BATCH      = 300
CHROMA_BATCH     = 1000
PARALLEL_WORKERS = 4

FORCE_REBUILD    = False   # IMPORTANT: keep False after first run

MAX_CHUNK_CHARS  = 800
EMBEDDING_MODEL  = "nomic-embed-text"


# HELPERS
def load_csv(filename, **kwargs):
    path = os.path.join(DATA_DIR, filename)
    df   = pd.read_csv(path, encoding="latin1", **kwargs)
    print(f"  Loaded {filename}: {len(df)} rows")
    return df


def clean_df(df, required_cols):
    """Drop nulls, cast to str, remove literal 'nan' strings."""
    df = df.dropna(subset=required_cols).copy()
    for col in required_cols:
        df[col] = df[col].astype(str).str.strip()
    for col in required_cols:
        df = df[df[col].str.lower() != "nan"]
    return df.reset_index(drop=True)


def _truncate(text: str, max_chars: int = MAX_CHUNK_CHARS) -> str:
    """Truncate at the last sentence boundary within max_chars."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    for sep in (". ", "! ", "? ", "\n"):
        idx = truncated.rfind(sep)
        if idx > max_chars // 2:
            return truncated[: idx + 1].strip()
    return truncated.strip()


def _content_hash(text: str) -> str:
    """Short MD5 hash used for content deduplication."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


def build_documents(df, prefix, meta_fn=None):
    """Store query + answer as page_content so retrieval matches on question semantics."""
    docs, ids = [], []
    for i, row in df.iterrows():
        answer  = _truncate(str(row["answers"]).strip())
        query   = str(row.get("query", "")).strip()[:200]
        content = f"Q: {query}\nA: {answer}" if query else answer
        if not content.strip():
            continue
        doc_id   = f"{prefix}_{i}"
        metadata = {"source": prefix, "query": query}
        if meta_fn:
            metadata.update(meta_fn(row))
        docs.append(Document(page_content=content, metadata=metadata, id=doc_id))
        ids.append(doc_id)
    print(f"  Built {len(docs)} documents for prefix '{prefix}'")
    return docs, ids


def build_documents_query_only(df, prefix, meta_fn=None):
    """For datasets with no answers column (df5, df7)."""
    docs, ids = [], []
    for i, row in df.iterrows():
        doc_id   = f"{prefix}_{i}"
        metadata = {"source": prefix}
        if meta_fn:
            metadata.update(meta_fn(row))
        docs.append(Document(
            page_content=str(row["query"]).strip(),
            metadata=metadata,
            id=doc_id,
        ))
        ids.append(doc_id)
    print(f"  Built {len(docs)} documents for prefix '{prefix}'")
    return docs, ids


def deduplicate(docs: list, ids: list):
    """
    Remove documents with identical page_content before embedding.
    Duplicates waste Ollama time and add retrieval noise.
    """
    seen   = set()
    u_docs, u_ids = [], []
    for doc, id_ in zip(docs, ids):
        h = _content_hash(doc.page_content)
        if h not in seen:
            seen.add(h)
            u_docs.append(doc)
            u_ids.append(id_)
    removed = len(docs) - len(u_docs)
    if removed:
        print(f"  Deduplication: removed {removed} duplicates ({len(u_docs)} unique remain)")
    return u_docs, u_ids


def _embed_with_retry(embed_fn, texts: list, batch_idx: int) -> list:
    """
    Embed a list of texts, retrying with halved sub-batches if the model
    returns a context-length error (status 400).
    """
    try:
        return embed_fn(texts)
    except Exception as exc:
        err = str(exc).lower()
        if "context length" not in err and "input length" not in err:
            raise

        if len(texts) == 1:
            truncated = [texts[0][:200]]
            print(f"\n  ⚠️  Batch {batch_idx}: single doc too long, hard-truncating to 200 chars")
            return embed_fn(truncated)

        mid = len(texts) // 2
        left  = _embed_with_retry(embed_fn, texts[:mid], batch_idx)
        right = _embed_with_retry(embed_fn, texts[mid:], batch_idx)
        return left + right


def embed_and_insert(
    collection,
    documents: list,
    ids: list,
    embed_fn,
    embed_batch: int = EMBED_BATCH,
    chroma_batch: int = CHROMA_BATCH,
    parallel_workers: int = PARALLEL_WORKERS,
):
    """
    Pre-compute all embeddings in parallel, then insert into Chroma directly.
    """
    total     = len(documents)
    texts     = [d.page_content for d in documents]
    metadatas = [d.metadata     for d in documents]

    batches = [
        (i, texts[start : min(start + embed_batch, total)])
        for i, start in enumerate(range(0, total, embed_batch))
    ]
    total_batches = len(batches)

    print(f"\nEmbedding {total:,} documents across {total_batches} batches "
          f"(batch={embed_batch}, workers={parallel_workers}) ...")

    all_embeddings: list = [None] * total_batches

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {
            executor.submit(_embed_with_retry, embed_fn, batch_texts, batch_idx): batch_idx
            for batch_idx, batch_texts in batches
        }
        with tqdm(total=total_batches, desc="Embedding", unit="batch") as pbar:
            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    all_embeddings[batch_idx] = future.result()
                except Exception as exc:
                    print(f"\n  ✗  Batch {batch_idx} failed permanently: {exc}")
                    raise
                pbar.update(1)

    flat_embeddings = [vec for batch in all_embeddings for vec in batch]

    if len(flat_embeddings) != total:
        raise ValueError(
            f"Embedding count mismatch: expected {total}, got {len(flat_embeddings)}"
        )

    print(f"Inserting {total:,} vectors into Chroma (batch={chroma_batch}) ...")
    for start in tqdm(range(0, total, chroma_batch), desc="Inserting", unit="batch"):
        end = min(start + chroma_batch, total)
        collection.add(
            ids        = ids[start:end],
            embeddings = flat_embeddings[start:end],
            documents  = texts[start:end],
            metadatas  = metadatas[start:end],
        )

    print(f"✓ Inserted {total:,} documents.")


# FORCE REBUILD
if FORCE_REBUILD:
    if os.path.exists(DB_LOCATION):
        print(f"FORCE_REBUILD=True — deleting {DB_LOCATION} ...")
        shutil.rmtree(DB_LOCATION)
        print("Old DB deleted.")

if os.path.exists(DB_LOCATION):
    _check_client = chromadb.PersistentClient(
        path=DB_LOCATION,
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        _check_col = _check_client.get_collection("therapy")
        add_documents = _check_col.count() == 0
    except Exception:
        add_documents = True  # collection doesn't exist yet
else:
    add_documents = True


# LOAD + CLEAN DATA
print("\n=== Loading CSVs ===")
t0 = time.time()

df  = clean_df(load_csv("cleaned.csv"),                ["query", "answers"])
df1 = clean_df(load_csv("Dataset_clean.csv"),          ["query", "answers"])
df2 = clean_df(load_csv("combined1.csv"),              ["query", "answers"])
df3 = clean_df(load_csv("counselchat-data_clean.csv"), ["query", "answers", "title", "topics"])
df4 = clean_df(load_csv("empathy.csv"),                ["query", "answers"])
df5 = clean_df(load_csv("CombinedData.csv", on_bad_lines="skip", engine="python"), ["query", "status"])
df6 = clean_df(load_csv("Mental_Health_FAQ.csv"),      ["query", "answers"])
df7 = clean_df(load_csv("therapy_recommendation_dataset_cleaned.csv"), ["query", "therapy_type"])

print(f"CSVs loaded in {time.time() - t0:.1f}s")


# BUILD DOCUMENTS
all_docs: list = []
all_ids:  list = []

if add_documents:
    print("\n=== Building documents (parallel) ===")
    t1 = time.time()

    tasks = [
        (df,  "df",  build_documents,            None),
        (df1, "df1", build_documents,            None),
        (df2, "df2", build_documents,            None),
        (df3, "df3", build_documents,            lambda r: {"title": r["title"], "topics": r["topics"]}),
        (df4, "df4", build_documents,            None),
        (df5, "df5", build_documents_query_only, lambda r: {"status": r["status"]}),
        (df6, "df6", build_documents,            None),
        (df7, "df7", build_documents_query_only, lambda r: {"therapytype": r["therapy_type"]}),
    ]

    with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as executor:
        futures = {
            executor.submit(builder, df_, prefix, m_fn): prefix
            for df_, prefix, builder, m_fn in tasks
        }
        for future in as_completed(futures):
            docs, ids = future.result()
            all_docs.extend(docs)
            all_ids.extend(ids)

    print(f"Document building done in {time.time() - t1:.1f}s  |  raw total={len(all_docs):,}")

    all_docs, all_ids = deduplicate(all_docs, all_ids)
    print(f"After dedup: {len(all_docs):,} documents to embed")


# VECTOR STORE + EMBEDDINGS
print(f"\n=== Initialising vector store (model: {EMBEDDING_MODEL}) ===")

embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL)

chroma_client = chromadb.PersistentClient(
    path=DB_LOCATION,
    settings=Settings(anonymized_telemetry=False),
)

# LangChain wrapper — needed for as_retriever() and similarity_search_with_relevance_scores()
vector_store = Chroma(
    client=chroma_client,
    collection_name="therapy",
    embedding_function=embeddings_model,
)

if add_documents:
    raw_collection = chroma_client.get_or_create_collection("therapy")

    t2 = time.time()
    embed_and_insert(
        collection       = raw_collection,
        documents        = all_docs,
        ids              = all_ids,
        embed_fn         = embeddings_model.embed_documents,
        embed_batch      = EMBED_BATCH,
        chroma_batch     = CHROMA_BATCH,
        parallel_workers = PARALLEL_WORKERS,
    )
    print(f"\n✓ All documents embedded and inserted in {time.time() - t2:.1f}s")

else:
    raw_collection = chroma_client.get_collection("therapy")
    count = raw_collection.count()
    print(f"DB already exists — skipping embedding.  ({count:,} docs in collection)")

# RETRIEVER
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":           12,   # return more docs
        "fetch_k":     100,  # wider candidate pool
        "lambda_mult": 0.8,  # favor relevance over diversity
    }
)

print("\nRetriever ready.")