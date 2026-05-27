"""Unit tests for lightweight answer.py helpers."""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(__file__))


class FakeTextEmbedding:
    created: list[str] = []

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.created.append(model_name)


for mod in ["fastembed", "qdrant_client", "qdrant_client.models", "requests", "index_pdfs"]:
    parts = mod.split(".")
    parent = None
    for i, part in enumerate(parts):
        full = ".".join(parts[: i + 1])
        if full not in sys.modules:
            module = types.ModuleType(full)
            sys.modules[full] = module
            if parent is not None:
                setattr(parent, part, module)
        parent = sys.modules[full]

sys.modules["fastembed"].TextEmbedding = FakeTextEmbedding  # type: ignore[attr-defined]
sys.modules["fastembed"].SparseTextEmbedding = object  # type: ignore[attr-defined]
sys.modules["qdrant_client"].QdrantClient = object  # type: ignore[attr-defined]
sys.modules["qdrant_client.models"].Fusion = object  # type: ignore[attr-defined]
sys.modules["qdrant_client.models"].FusionQuery = object  # type: ignore[attr-defined]
sys.modules["qdrant_client.models"].Prefetch = object  # type: ignore[attr-defined]
sys.modules["qdrant_client.models"].SparseVector = object  # type: ignore[attr-defined]
sys.modules["index_pdfs"].collection_for_project = lambda project: f"{project}_docs"  # type: ignore[attr-defined]

from answer import _get_dense_embedder  # noqa: E402


def test_dense_embedder_reuses_same_model() -> None:
    first = _get_dense_embedder("model-a")
    second = _get_dense_embedder("model-a")

    assert first is second
    assert FakeTextEmbedding.created == ["model-a"]
    print("test_dense_embedder_reuses_same_model PASSED")


def test_dense_embedder_separates_model_names() -> None:
    first = _get_dense_embedder("model-b")
    second = _get_dense_embedder("model-c")

    assert first is not second
    assert FakeTextEmbedding.created[-2:] == ["model-b", "model-c"]
    print("test_dense_embedder_separates_model_names PASSED")


if __name__ == "__main__":
    test_dense_embedder_reuses_same_model()
    test_dense_embedder_separates_model_names()
    print("\nAll tests PASSED")
