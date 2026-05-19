import argparse
import json
import os
import re
import sys
from typing import Iterator

import requests
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

from index_pdfs import collection_for_project


_sparse_embedder: SparseTextEmbedding | None = None
_reranker = None


def _get_sparse_embedder() -> SparseTextEmbedding:
    global _sparse_embedder
    if _sparse_embedder is None:
        _sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _sparse_embedder


def _get_reranker():
    global _reranker
    if _reranker is None:
        from flashrank import Ranker
        _reranker = Ranker(model_name="ms-marco-MultiBERT-L-12", cache_dir="/sandbox/cache/flashrank")
    return _reranker


def query_terms(query: str) -> list[str]:
    return [term for term in re.findall(r"[0-9A-Za-z가-힣]+", query) if len(term) >= 2]


def rerank_score(vector_score: float, text: str, terms: list[str]) -> float:
    score = vector_score
    for term in terms:
        if term in text:
            score += 0.08
    return score


def retrieve(query: str, limit: int, candidates: int, project: str | None = None) -> list[dict]:
    qdrant_url = os.environ["QDRANT_URL"]
    collection = collection_for_project(project or os.environ.get("PROJECT_NAME", "inbox"))
    model_name = os.environ["EMBEDDING_MODEL"]

    embedder = TextEmbedding(model_name=model_name)
    dense_vector = next(embedder.embed([query])).tolist()

    client = QdrantClient(url=qdrant_url)
    n = max(limit, candidates)

    try:
        sv = next(_get_sparse_embedder().embed([query]))
        sparse_query = SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())
        results = client.query_points(
            collection_name=collection,
            prefetch=[
                Prefetch(query=dense_vector, using="dense", limit=n),
                Prefetch(query=sparse_query, using="sparse", limit=n),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=n,
            with_payload=True,
        ).points
    except Exception:
        results = client.query_points(
            collection_name=collection,
            query=dense_vector,
            limit=n,
            with_payload=True,
        ).points

    try:
        from flashrank import RerankRequest
        passages = [{"id": i, "text": (r.payload or {}).get("text", "")} for i, r in enumerate(results)]
        reranked = _get_reranker().rerank(RerankRequest(query=query, passages=passages))
        id_to_result = {i: r for i, r in enumerate(results)}
        top_results = [id_to_result[r["id"]] for r in reranked[:limit]]
    except Exception:
        terms = query_terms(query)
        top_results = sorted(
            results,
            key=lambda r: rerank_score(r.score, ((r.payload or {}).get("text") or ""), terms),
            reverse=True,
        )[:limit]

    contexts = []
    for idx, result in enumerate(top_results, start=1):
        payload = result.payload or {}
        contexts.append(
            {
                "id": idx,
                "score": result.score,
                "vector_score": result.score,
                "source": payload.get("source"),
                "page": payload.get("page"),
                "locator": payload.get("locator"),
                "text": payload.get("text", ""),
            }
        )
    return contexts


def build_prompt(query: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join(
        (
            f"[{item['id']}] source={item['source']} location={item.get('locator') or item.get('page')}\n"
            f"{item['text']}"
        )
        for item in contexts
    )
    return f"""너는 사용자의 로컬 PDF 자료만 근거로 답하는 RAG assistant다.

규칙:
- 아래 CONTEXT에 있는 내용만 근거로 답하라.
- CONTEXT에 없는 내용은 추측하지 말고 "자료에서 확인되지 않습니다"라고 말하라.
- 답변은 한국어로 하라.
- 핵심을 먼저 말하고, 필요하면 짧은 bullet로 정리하라.
- 문장마다 가능한 한 [출처번호]를 붙여라.
- 마지막에 "출처" 섹션을 만들고 파일명과 페이지를 나열하라.

QUESTION:
{query}

CONTEXT:
{context_text}
"""


def call_gemini(prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    response = requests.post(
        url,
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.9,
            },
        },
        timeout=120,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Gemini API error {response.status_code}: {response.text}")
    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected Gemini response: {data}") from exc


def stream_gemini(prompt: str) -> Iterator[str]:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"
    response = requests.post(
        url,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json={
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "topP": 0.9},
        },
        stream=True,
        timeout=300,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Gemini API error {response.status_code}: {response.text[:500]}")
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line.startswith("data: "):
            continue
        try:
            data = json.loads(line[6:])
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            if text:
                yield text
        except (KeyError, IndexError, json.JSONDecodeError):
            pass


def print_contexts(contexts: list[dict]) -> None:
    print("\nRetrieved contexts:")
    for item in contexts:
        print(
            f"- [{item['id']}] score={item['score']:.4f} "
            f"source={item['source']} page={item['page']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("--project", default=os.environ.get("PROJECT_NAME", "inbox"))
    args = parser.parse_args()

    contexts = retrieve(args.query, args.limit, args.candidates, args.project)
    if not contexts:
        raise SystemExit("No relevant context found.")

    prompt = build_prompt(args.query, contexts)
    try:
        answer = call_gemini(prompt)
    except Exception as exc:
        print(f"Answer generation failed: {exc}", file=sys.stderr)
        print_contexts(contexts)
        raise SystemExit(2)

    print(answer.strip())
    if args.show_context:
        print_contexts(contexts)


if __name__ == "__main__":
    main()
