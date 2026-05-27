"""Unit tests for STT ensemble voting logic (no real audio or models needed)."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Stub out heavy imports so tests run without GPU/model downloads
import types

for mod in ["fitz", "olefile", "openpyxl", "pytesseract", "docx", "fastembed",
            "faster_whisper", "pptx", "qdrant_client", "qdrant_client.models", "tqdm"]:
    parts = mod.split(".")
    parent = None
    for i, part in enumerate(parts):
        full = ".".join(parts[: i + 1])
        if full not in sys.modules:
            m = types.ModuleType(full)
            sys.modules[full] = m
            if parent is not None:
                setattr(parent, part, m)
        parent = sys.modules[full]

# Provide minimal stubs needed for module-level "from X import Y" imports
sys.modules["fitz"].Page = object  # type: ignore[attr-defined]
sys.modules["fitz"].Matrix = object  # type: ignore[attr-defined]
sys.modules["fitz"].open = object  # type: ignore[attr-defined]
sys.modules["docx"].Document = object  # type: ignore[attr-defined]
sys.modules["fastembed"].TextEmbedding = object  # type: ignore[attr-defined]
sys.modules["fastembed"].SparseTextEmbedding = object  # type: ignore[attr-defined]
sys.modules["faster_whisper"].WhisperModel = object  # type: ignore[attr-defined]
sys.modules["pptx"].Presentation = object  # type: ignore[attr-defined]
sys.modules["tqdm"].tqdm = lambda it, **kw: it  # type: ignore[attr-defined]
sys.modules["qdrant_client"].QdrantClient = object  # type: ignore[attr-defined]
sys.modules["qdrant_client.models"].Distance = object  # type: ignore[attr-defined]
sys.modules["qdrant_client.models"].FieldCondition = object  # type: ignore[attr-defined]
sys.modules["qdrant_client.models"].Filter = object  # type: ignore[attr-defined]
sys.modules["qdrant_client.models"].MatchValue = object  # type: ignore[attr-defined]
sys.modules["qdrant_client.models"].PointStruct = object  # type: ignore[attr-defined]
sys.modules["qdrant_client.models"].SparseVector = object  # type: ignore[attr-defined]
sys.modules["qdrant_client.models"].SparseVectorParams = object  # type: ignore[attr-defined]
sys.modules["qdrant_client.models"].VectorParams = object  # type: ignore[attr-defined]

from index_pdfs import (  # noqa: E402
    _text_similarity,
    _confidence_tier,
    _merge_into_events,
    _cluster_by_similarity,
    iter_batches,
    vote_and_merge,
)


def test_text_similarity() -> None:
    assert _text_similarity("hello", "hello") == 1.0
    assert _text_similarity("", "") == 1.0
    assert _text_similarity("abc", "xyz") < 0.5
    sim = _text_similarity("안녕하세요 반갑습니다", "안녕하세요 반갑습니다")
    assert sim == 1.0
    partial = _text_similarity("안녕하세요", "안녕하세요 반갑습니다")
    assert 0.5 < partial < 1.0
    print("test_text_similarity PASSED")


def test_confidence_tier() -> None:
    assert _confidence_tier(4, 4) == "CONFIRMED"
    assert _confidence_tier(3, 4) == "HIGH"
    assert _confidence_tier(2, 4) == "MEDIUM"
    assert _confidence_tier(1, 4) == "LOW"
    assert _confidence_tier(1, 1) == "CONFIRMED"
    assert _confidence_tier(2, 3) == "MEDIUM"  # 2/3 ≈ 0.667, below 0.75 threshold
    print("test_confidence_tier PASSED")


def test_merge_into_events() -> None:
    segs = [
        {"start": 0.0, "end": 2.0, "text": "a"},
        {"start": 1.5, "end": 3.5, "text": "b"},  # overlaps first
        {"start": 10.0, "end": 12.0, "text": "c"},  # separate event
    ]
    events = _merge_into_events(segs, tolerance=0.5)
    assert len(events) == 2
    assert len(events[0]) == 2
    assert len(events[1]) == 1
    print("test_merge_into_events PASSED")


def test_merge_into_events_tolerance() -> None:
    segs = [
        {"start": 0.0, "end": 2.0, "text": "a"},
        {"start": 2.3, "end": 4.0, "text": "b"},  # gap=0.3, within tolerance=0.5 → merged
    ]
    events = _merge_into_events(segs, tolerance=0.5)
    assert len(events) == 1
    assert len(events[0]) == 2

    events2 = _merge_into_events(segs, tolerance=0.1)
    assert len(events2) == 2
    print("test_merge_into_events_tolerance PASSED")


def test_cluster_by_similarity() -> None:
    segs = [
        {"start": 0.0, "end": 2.0, "text": "안녕하세요", "_model": "A"},
        {"start": 0.1, "end": 2.1, "text": "안녕하세요", "_model": "B"},  # same → same cluster
        {"start": 0.2, "end": 2.2, "text": "전혀 다른 내용입니다", "_model": "C"},  # different
    ]
    clusters = _cluster_by_similarity(segs, threshold=0.6)
    assert len(clusters) == 2
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]
    print("test_cluster_by_similarity PASSED")


def test_vote_and_merge_basic() -> None:
    # 3 models, segment at 0-2s: models A and B agree, C hallucinates
    model_segments = {
        "A": [{"start": 0.0, "end": 2.0, "text": "오늘 날씨가 좋습니다", "avg_logprob": -0.3, "no_speech_prob": 0.1, "language": "ko"}],
        "B": [{"start": 0.1, "end": 2.1, "text": "오늘 날씨가 좋습니다", "avg_logprob": -0.4, "no_speech_prob": 0.1, "language": "ko"}],
        "C": [{"start": 0.0, "end": 2.0, "text": "완전히 다른 환각 텍스트", "avg_logprob": -0.9, "no_speech_prob": 0.2, "language": "ko"}],
    }
    results = vote_and_merge(model_segments, min_votes=2, similarity_threshold=0.6, align_tolerance=0.5)
    assert len(results) == 1
    assert results[0]["text"] == "오늘 날씨가 좋습니다"
    assert results[0]["votes"] == 2
    assert results[0]["stt_confidence"] in ("MEDIUM", "HIGH")
    print("test_vote_and_merge_basic PASSED")


def test_vote_and_merge_hallucination_rejected() -> None:
    # All 3 models produce completely unrelated text → no consensus → 0 results
    # Note: texts must be truly dissimilar (not share common substrings)
    model_segments = {
        "A": [{"start": 0.0, "end": 2.0, "text": "사과 바나나 포도", "avg_logprob": -0.5, "no_speech_prob": 0.1, "language": "ko"}],
        "B": [{"start": 0.1, "end": 2.1, "text": "컴퓨터 키보드 마우스", "avg_logprob": -0.5, "no_speech_prob": 0.1, "language": "ko"}],
        "C": [{"start": 0.0, "end": 2.0, "text": "하늘 구름 바람 비", "avg_logprob": -0.5, "no_speech_prob": 0.1, "language": "ko"}],
    }
    results = vote_and_merge(model_segments, min_votes=2, similarity_threshold=0.6, align_tolerance=0.5)
    assert len(results) == 0
    print("test_vote_and_merge_hallucination_rejected PASSED")


def test_vote_and_merge_all_agree() -> None:
    # 4 models all agree → CONFIRMED
    text = "모든 모델이 동의하는 내용입니다"
    model_segments = {
        f"model_{i}": [{"start": float(i) * 0.05, "end": 2.0 + float(i) * 0.05,
                        "text": text, "avg_logprob": -0.2, "no_speech_prob": 0.05, "language": "ko"}]
        for i in range(4)
    }
    results = vote_and_merge(model_segments, min_votes=2, similarity_threshold=0.6, align_tolerance=0.5)
    assert len(results) == 1
    assert results[0]["stt_confidence"] == "CONFIRMED"
    assert results[0]["votes"] == 4
    print("test_vote_and_merge_all_agree PASSED")


def test_single_model_backward_compat() -> None:
    # 1 model, min_votes=2 → effective_min clamped to 1, all segments pass
    model_segments = {
        "small": [
            {"start": 0.0, "end": 2.0, "text": "첫 번째 세그먼트", "avg_logprob": -0.3, "no_speech_prob": 0.1, "language": "ko"},
            {"start": 5.0, "end": 7.0, "text": "두 번째 세그먼트", "avg_logprob": -0.4, "no_speech_prob": 0.1, "language": "ko"},
        ]
    }
    results = vote_and_merge(model_segments, min_votes=2, similarity_threshold=0.6, align_tolerance=0.5)
    assert len(results) == 2
    assert results[0]["stt_confidence"] == "CONFIRMED"
    print("test_single_model_backward_compat PASSED")


def test_vote_and_merge_empty() -> None:
    results = vote_and_merge({}, min_votes=2, similarity_threshold=0.6, align_tolerance=0.5)
    assert results == []

    results2 = vote_and_merge({"A": [], "B": []}, min_votes=2, similarity_threshold=0.6, align_tolerance=0.5)
    assert results2 == []
    print("test_vote_and_merge_empty PASSED")


def test_vote_and_merge_best_logprob_selected() -> None:
    # Model B has higher logprob → should be selected as best_seg
    model_segments = {
        "A": [{"start": 0.0, "end": 2.0, "text": "좋은 텍스트입니다", "avg_logprob": -0.8, "no_speech_prob": 0.1, "language": "ko"}],
        "B": [{"start": 0.1, "end": 2.1, "text": "좋은 텍스트입니다", "avg_logprob": -0.2, "no_speech_prob": 0.05, "language": "ko"}],
    }
    results = vote_and_merge(model_segments, min_votes=2, similarity_threshold=0.6, align_tolerance=0.5)
    assert len(results) == 1
    assert results[0]["model"] == "B"
    assert results[0]["avg_logprob"] == -0.2
    print("test_vote_and_merge_best_logprob_selected PASSED")


def test_iter_batches() -> None:
    batches = list(iter_batches([1, 2, 3, 4, 5], 2))
    assert batches == [[1, 2], [3, 4], [5]]
    try:
        list(iter_batches([1], 0))
    except ValueError:
        pass
    else:
        raise AssertionError("iter_batches should reject non-positive batch sizes")
    print("test_iter_batches PASSED")


if __name__ == "__main__":
    test_text_similarity()
    test_confidence_tier()
    test_merge_into_events()
    test_merge_into_events_tolerance()
    test_cluster_by_similarity()
    test_vote_and_merge_basic()
    test_vote_and_merge_hallucination_rejected()
    test_vote_and_merge_all_agree()
    test_single_model_backward_compat()
    test_vote_and_merge_empty()
    test_vote_and_merge_best_logprob_selected()
    test_iter_batches()
    print("\nAll tests PASSED")
