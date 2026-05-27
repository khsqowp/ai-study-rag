import argparse
import os
import re

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from project_utils import collection_for_project


def query_terms(query: str) -> list[str]:
    return [term for term in re.findall(r"[0-9A-Za-z가-힣]+", query) if len(term) >= 2]


def rerank_score(vector_score: float, text: str, terms: list[str]) -> float:
    score = vector_score
    for term in terms:
        if term in text:
            score += 0.08
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=40)
    parser.add_argument("--project", default=os.environ.get("PROJECT_NAME", "inbox"))
    args = parser.parse_args()

    qdrant_url = os.environ["QDRANT_URL"]
    collection = collection_for_project(args.project)
    model_name = os.environ["EMBEDDING_MODEL"]

    embedder = TextEmbedding(model_name=model_name)
    vector = next(embedder.embed([args.query])).tolist()

    client = QdrantClient(url=qdrant_url)
    results = client.query_points(
        collection_name=collection,
        query=vector,
        limit=max(args.limit, args.candidates),
        with_payload=True,
    ).points

    terms = query_terms(args.query)
    reranked = sorted(
        results,
        key=lambda result: rerank_score(
            result.score,
            ((result.payload or {}).get("text") or ""),
            terms,
        ),
        reverse=True,
    )[: args.limit]

    for idx, result in enumerate(reranked, start=1):
        payload = result.payload or {}
        text = (payload.get("text") or "").strip()
        final_score = rerank_score(result.score, text, terms)
        print(f"\n[{idx}] score={final_score:.4f} vector={result.score:.4f}")
        print(f"source={payload.get('source')} page={payload.get('page')}")
        print(text[:900])


if __name__ == "__main__":
    main()
