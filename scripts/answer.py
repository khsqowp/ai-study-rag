import argparse
import json
import os
import re
import sys
import threading
from typing import Iterator

import requests
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

from index_pdfs import collection_for_project


_sparse_embedder: SparseTextEmbedding | None = None
_dense_embedders: dict[str, TextEmbedding] = {}
_reranker = None
_embedder_lock = threading.Lock()


def _get_dense_embedder(model_name: str) -> TextEmbedding:
    with _embedder_lock:
        embedder = _dense_embedders.get(model_name)
        if embedder is None:
            embedder = TextEmbedding(model_name=model_name)
            _dense_embedders[model_name] = embedder
        return embedder


def _get_sparse_embedder() -> SparseTextEmbedding:
    global _sparse_embedder
    with _embedder_lock:
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

    embedder = _get_dense_embedder(model_name)
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


def build_prompt(query: str, contexts: list[dict], web_search: bool = False) -> str:
    context_text = "\n\n".join(
        (
            f"[{item['id']}] source={item['source']} location={item.get('locator') or item.get('page')}\n"
            f"{item['text']}"
        )
        for item in contexts
    )
    if web_search:
        system = """너는 로컬 문서와 웹 검색을 함께 활용해 답하는 RAG assistant다.

규칙:
- 아래 CONTEXT(로컬 문서)와 Google 웹 검색 결과를 모두 활용하라.
- 로컬 문서를 인용할 때는 [출처번호]를 붙여라. 웹 출처는 Gemini가 자동 인용한다.
- 답변은 한국어로 하라.
- 핵심을 먼저 말하고, 필요하면 짧은 bullet로 정리하라.
- 마지막에 "로컬 문서 출처" 섹션을 만들고 파일명과 페이지를 나열하라."""
    else:
        system = """너는 사용자의 로컬 PDF 자료만 근거로 답하는 RAG assistant다.

규칙:
- 아래 CONTEXT에 있는 내용만 근거로 답하라.
- CONTEXT에 없는 내용은 추측하지 말고 "자료에서 확인되지 않습니다"라고 말하라.
- 답변은 한국어로 하라.
- 핵심을 먼저 말하고, 필요하면 짧은 bullet로 정리하라.
- 문장마다 가능한 한 [출처번호]를 붙여라.
- 마지막에 "출처" 섹션을 만들고 파일명과 페이지를 나열하라."""
    return f"""{system}

QUESTION:
{query}

CONTEXT (로컬 문서):
{context_text}
"""


def _extract_grounding_sources(candidate: dict) -> list[dict]:
    sources = []
    for chunk in candidate.get("groundingMetadata", {}).get("groundingChunks", []):
        web = chunk.get("web", {})
        if web.get("uri"):
            sources.append({"title": web.get("title", ""), "uri": web["uri"]})
    return sources


def _gemini_api_key(api_key: str | None = None) -> str:
    return (
        (api_key or "").strip()
        or os.environ.get("AI_RAG_API_KEY_GEMINI", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
    )


def call_gemini(prompt: str, web_search: bool = False, api_key: str | None = None) -> tuple[str, list[dict]]:
    api_key = _gemini_api_key(api_key)
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
    if not api_key:
        raise RuntimeError("AI_RAG_API_KEY_GEMINI is not set")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body: dict = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "topP": 0.9},
    }
    if web_search:
        body["tools"] = [{"googleSearch": {}}]

    response = requests.post(
        url,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json=body,
        timeout=120,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Gemini API error {response.status_code}: {response.text}")
    data = response.json()
    try:
        candidate = data["candidates"][0]
        answer = candidate["content"]["parts"][0]["text"]
        web_sources = _extract_grounding_sources(candidate) if web_search else []
        return answer, web_sources
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected Gemini response: {data}") from exc


def stream_gemini(prompt: str, web_search: bool = False, api_key: str | None = None) -> Iterator[dict]:
    api_key = _gemini_api_key(api_key)
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
    if not api_key:
        raise RuntimeError("AI_RAG_API_KEY_GEMINI is not set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"
    body: dict = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "topP": 0.9},
    }
    if web_search:
        body["tools"] = [{"googleSearch": {}}]

    response = requests.post(
        url,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json=body,
        stream=True,
        timeout=300,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Gemini API error {response.status_code}: {response.text[:500]}")

    all_sources: list[dict] = []
    seen_uris: set[str] = set()

    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line.startswith("data: "):
            continue
        try:
            data = json.loads(line[6:])
            candidate = data["candidates"][0]
            try:
                text = candidate["content"]["parts"][0]["text"]
                if text:
                    yield {"type": "token", "text": text}
            except (KeyError, IndexError):
                pass
            if web_search:
                for src in _extract_grounding_sources(candidate):
                    if src["uri"] not in seen_uris:
                        seen_uris.add(src["uri"])
                        all_sources.append(src)
        except (KeyError, IndexError, json.JSONDecodeError):
            pass

    if web_search and all_sources:
        yield {"type": "grounding", "sources": all_sources}


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
