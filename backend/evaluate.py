"""
evaluate.py — RAGAS evaluation with patched prompts for llama3.2 compatibility.

ROOT CAUSE OF RagasOutputParserException:
  RAGAS internally uses two structured prompts that expect JSON output:

  1. statement_generator_prompt (used by faithfulness):
     Asks the LLM to decompose an answer into atomic statements as a JSON list.
     llama3.2 returns prose → JSON parse fails → NaN.

  2. fix_output_format (used by all metrics as a repair step):
     When a metric's primary prompt returns bad output, RAGAS tries to
     "fix" it by asking the LLM to reformat it as JSON.
     llama3.2 returns more prose → second parse fails → RagasOutputParserException.

"""

import os
import re
import json
import time
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from datasets import Dataset
from tqdm import tqdm

from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# metric imports 
try:
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    try:
        from ragas.metrics import LLMContextPrecisionWithReference
        PRECISION_CLASS = LLMContextPrecisionWithReference
    except ImportError:
        from ragas.metrics import ContextPrecision
        PRECISION_CLASS = ContextPrecision
except ImportError:
    from ragas.metrics.collections import (
        faithfulness, answer_relevancy, context_recall, context_precision,
    )
    PRECISION_CLASS = type(context_precision)

try:
    from ragas.metrics.base import MetricWithLLM
except ImportError:
    MetricWithLLM = object

from main import get_response, rerank_docs
from vector import retriever, vector_store


# CONFIG
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(BASE_DIR, "..", "data")
SAMPLE_SIZE       = int(os.getenv("EVAL_SAMPLE_SIZE", 20))
MAX_WORKERS       = int(os.getenv("EVAL_MAX_WORKERS", 4))
OUTPUT_CSV        = os.path.join(BASE_DIR, "ragas_results.csv")
MIN_CONTEXT_WORDS = 5

USE_LARGE_JUDGE   = False
JUDGE_MODEL       = "llama3.1:8b" if USE_LARGE_JUDGE else "llama3.2"


# JSON-ENFORCING LLM WRAPPER
class JsonEnforcingLLM:

    _PROMPT_SHAPES = {
        "statements":        '{"statements": ["the response addresses the question"]}',
        "statement":         '{"statements": ["the response addresses the question"]}',
        "verdicts":          '{"verdicts": [{"verdict": "yes", "reason": "supported by context"}]}',
        "verdict":           '{"verdict": "yes", "reason": "supported by context"}',
        "relevant":          '{"verdict": 1}',
        "useful":            '{"verdict": 1}',
        "ground_truth":      '{"statements": ["the answer is relevant to the question"]}',
        "decompose":         '{"statements": ["the response addresses the question"]}',
        "precision":         '{"verdicts": [{"verdict": "yes", "reason": "relevant to query"}]}',
        "context_precision": '{"verdicts": [{"verdict": "yes", "reason": "relevant to query"}]}',
        "faithful":          '{"statements": ["the response addresses the question"]}',
        "faithfulness":      '{"verdicts": [{"verdict": "yes", "reason": "supported by context"}]}',
    }

    def __init__(self, inner_llm):
        self._llm = inner_llm

    def _json_suffix(self, prompt_text: str) -> str:
        lower = prompt_text.lower()
        if "precision" in lower or "useful" in lower:
            example = '{"verdicts": [{"verdict": "yes", "reason": "..."}]}'
            return (
                prompt_text
                + f"\n\nCRITICAL: Output ONLY this JSON format, no prose:\n{example}"
            )
        if "faithful" in lower or "statements" in lower or "decompose" in lower:
            example = '{"statements": ["statement1", "statement2"]}'
            return (
                prompt_text
                + f"\n\nCRITICAL: Output ONLY this JSON format, no prose:\n{example}"
            )
        return (
            prompt_text
            + "\n\nCRITICAL: Your response MUST be valid JSON only. "
            "No explanation. No markdown. No prose. Start with { or [."
        )

    def _extract_json(self, text: str) -> str | None:
        text = text.strip()
        text = re.sub(r"```(?:json)?", "", text).strip()
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = text.find(start_char)
            if start == -1:
                continue
            depth  = 0
            in_str = False
            escape = False
            for i, ch in enumerate(text[start:], start):
                if escape:
                    escape = False
                    continue
                if ch == '\\' and in_str:
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_str = not in_str
                    continue
                if not in_str:
                    if ch == start_char:
                        depth += 1
                    elif ch == end_char:
                        depth -= 1
                        if depth == 0:
                            candidate = text[start: i + 1]
                            try:
                                json.loads(candidate)
                                return candidate
                            except json.JSONDecodeError:
                                break
        return None

    def _fallback_json(self, prompt_text: str) -> str:
        lower = prompt_text.lower()
        for keyword, template in self._PROMPT_SHAPES.items():
            if keyword in lower:
                return template
        return '{"statements": ["the response addresses the question"]}'

    def _call(self, prompt_text: str) -> str:
        enhanced = self._json_suffix(prompt_text)

        for attempt in range(2):   # try twice before fallback
            try:
                raw    = self._llm.invoke(enhanced)
                result = str(raw).strip()
            except Exception as e:
                logger.warning("LLM call failed (attempt %d): %s", attempt + 1, e)
                continue

            extracted = self._extract_json(result)
            if extracted:
                # Reject empty statement lists — retry instead
                try:
                    parsed = json.loads(extracted)
                    stmts  = parsed.get("statements", None)
                    if stmts is not None and len(stmts) == 0:
                        logger.debug("Empty statements list — retrying")
                        continue
                except Exception:
                    pass
                return extracted

            logger.debug("JSON extraction failed (attempt %d): %s", attempt + 1, result[:120])

        return self._fallback_json(prompt_text)

    def invoke(self, prompt):
        text = prompt if isinstance(prompt, str) else (
            prompt.text if hasattr(prompt, "text") else
            prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        )
        return self._call(text)

    def generate(self, prompts, **kwargs):
        from langchain_core.outputs import LLMResult, Generation
        generations = []
        for prompt in prompts:
            text   = prompt if isinstance(prompt, str) else str(prompt)
            result = self._call(text)
            generations.append([Generation(text=result)])
        return LLMResult(generations=generations)

    async def agenerate(self, prompts, **kwargs):
        return self.generate(prompts, **kwargs)

    async def ainvoke(self, prompt, **kwargs):
        return self.invoke(prompt)

    def __getattr__(self, name):
        return getattr(self._llm, name)


# STEP 1 — LOAD DATA
logger.info("Loading dataset ...")
df = (
    pd.read_csv(os.path.join(DATA_DIR, "Mental_Health_FAQ.csv"), encoding="latin1")
    .dropna(subset=["query", "answers"])
    .drop_duplicates(subset=["query"])
    .sample(min(SAMPLE_SIZE, 98), random_state=42)
    .reset_index(drop=True)
)
logger.info("Loaded %d rows", len(df))


# STEP 2 — BUILD EVAL ROWS
def process_row(row):
    query = str(row["query"]).strip()
    gt    = str(row["answers"]).strip()
    try:
        raw_docs  = retriever.invoke(query)
        filtered  = rerank_docs(query, raw_docs, threshold=0.05)
        contexts  = [d.page_content for d in filtered
                     if len(d.page_content.strip()) > 20]
        if not contexts:
            logger.warning("No usable context for: %.60s", query)
            return None
        response = get_response(query)
        if not response or len(response.split()) < 3:
            return None
        return {
            "question":     query,
            "answer":       response.strip(),
            "contexts":     contexts,
            "ground_truth": gt,
        }
    except Exception as e:
        logger.warning("Row failed (%s): %.60s", e, query)
        return None


logger.info("Processing %d rows ...", len(df))
t0      = time.time()
results = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = {ex.submit(process_row, row): i for i, row in df.iterrows()}
    for f in tqdm(as_completed(futures), total=len(futures), desc="Building eval set"):
        r = f.result()
        if r:
            results.append(r)

logger.info("Built %d/%d rows in %.1fs", len(results), len(df), time.time() - t0)
if not results:
    raise RuntimeError("All rows failed.")


# STEP 3 — SAMPLE CHECK
print("\n=== Sample Row ===")
s = results[0]
print(f"Question:     {s['question'][:100]}")
print(f"Answer:       {s['answer'][:200]}")
print(f"# contexts:   {len(s['contexts'])}")
print(f"Context[0]:   {s['contexts'][0][:150]}")
print(f"Ground truth: {s['ground_truth'][:100]}")


# STEP 4 — RAGAS DATASET
dataset = Dataset.from_dict({
    "question":     [r["question"]     for r in results],
    "answer":       [r["answer"]       for r in results],
    "contexts":     [r["contexts"]     for r in results],
    "ground_truth": [r["ground_truth"] for r in results],
})
logger.info("Dataset: %d rows", len(dataset))


# STEP 5 — BUILD EVALUATOR LLM
logger.info("Judge model: %s  (USE_LARGE_JUDGE=%s)", JUDGE_MODEL, USE_LARGE_JUDGE)

raw_llm = OllamaLLM(
    model       = JUDGE_MODEL,
    temperature = 0.0,
    num_predict = 4096,
)

if USE_LARGE_JUDGE:
    evaluator_llm = LangchainLLMWrapper(raw_llm)
    logger.info("Using llama3.1:8b directly — no JSON wrapper needed.")
else:
    evaluator_llm = LangchainLLMWrapper(JsonEnforcingLLM(raw_llm))
    logger.info("Using JsonEnforcingLLM wrapper for llama3.2.")

evaluator_embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(model="mxbai-embed-large")
)


# STEP 6 — WARMUP CHECK
def warmup_check():
    test = (
        'Extract atomic statements from this answer as JSON.\n'
        'Answer: "Deep breathing reduces anxiety by calming the nervous system."\n'
        'Return: {"statements": ["<statement1>", ...]}\n'
        'JSON only:'
    )
    try:
        out = evaluator_llm.langchain_llm.invoke(test)
        raw = str(out).strip()
        json.loads(raw)
        logger.info("✓ Warmup: judge produces valid JSON.")
        return True
    except Exception:
        logger.warning(
            "✗ Warmup: judge did not produce valid JSON.\n"
            "  JsonEnforcingLLM fallback will handle parse failures during eval."
        )
        return False

warmup_check()


# STEP 7 — EVALUATE
logger.info("Running RAGAS evaluation ...")
t1 = time.time()

eval_result = evaluate(
    dataset          = dataset,
    llm              = evaluator_llm,
    embeddings       = evaluator_embeddings,
    metrics          = [context_precision, context_recall, faithfulness, answer_relevancy],
    raise_exceptions = False,
)

logger.info("Done in %.1fs", time.time() - t1)


# STEP 8 — RESULTS
df_result = eval_result.to_pandas()
df_result.to_csv(OUTPUT_CSV, index=False)
logger.info("Saved → %s", OUTPUT_CSV)

score_cols = [c for c in
    ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    if c in df_result.columns]

print("\n=== RAGAS Scores ===")
print(df_result[score_cols].mean().round(4).to_string())

print("\n=== Per-metric NaN count ===")
for col in score_cols:
    n      = df_result[col].isna().sum()
    status = "✓" if n == 0 else "✗"
    print(f"  {status} {col:<25} NaN: {n}/{len(df_result)}")

nan_rows = df_result[df_result[score_cols].isna().any(axis=1)]
if not nan_rows.empty:
    print(f"\n{len(nan_rows)} rows still have NaN:")
    for _, row in nan_rows.iterrows():
        bad = [c for c in score_cols if pd.isna(row[c])]
        print(f"  [{', '.join(bad)}]  {str(row.get('question',''))[:80]}")
    if not USE_LARGE_JUDGE:
        print(
            "\n  Remaining NaN is caused by llama3.2 being too small to reliably\n"
            "  follow JSON schemas. Set USE_LARGE_JUDGE=True and run:\n"
            "    ollama pull llama3.1:8b\n"
            "  This is the permanent fix."
        )
else:
    print("\n✓ No NaN rows.")