"""Unit tests for server-side answer cache helpers."""
import os
import sys
import tempfile
import types

cache_dir = tempfile.TemporaryDirectory()
os.environ["ANSWER_CACHE_ROOT"] = cache_dir.name
os.environ["ENABLE_ANSWER_CACHE"] = "1"
os.environ["ANSWER_CACHE_TTL_SECONDS"] = "3600"

sys.path.insert(0, os.path.dirname(__file__))


class FakeFastAPI:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def get(self, *args, **kwargs):
        return lambda fn: fn

    def post(self, *args, **kwargs):
        return lambda fn: fn


class FakeHTTPException(Exception):
    def __init__(self, status_code: int, detail) -> None:
        self.status_code = status_code
        self.detail = detail


for mod in [
    "fastapi",
    "fastapi.responses",
    "pydantic",
    "qdrant_client",
    "answer",
    "index_pdfs",
    "settings",
]:
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

sys.modules["fastapi"].FastAPI = FakeFastAPI  # type: ignore[attr-defined]
sys.modules["fastapi"].File = lambda *args, **kwargs: None  # type: ignore[attr-defined]
sys.modules["fastapi"].Form = lambda *args, **kwargs: None  # type: ignore[attr-defined]
sys.modules["fastapi"].Header = lambda *args, **kwargs: None  # type: ignore[attr-defined]
sys.modules["fastapi"].HTTPException = FakeHTTPException  # type: ignore[attr-defined]
sys.modules["fastapi"].UploadFile = object  # type: ignore[attr-defined]
sys.modules["fastapi.responses"].HTMLResponse = object  # type: ignore[attr-defined]
sys.modules["fastapi.responses"].StreamingResponse = object  # type: ignore[attr-defined]
sys.modules["pydantic"].BaseModel = object  # type: ignore[attr-defined]
sys.modules["qdrant_client"].QdrantClient = object  # type: ignore[attr-defined]
sys.modules["answer"].build_prompt = lambda *args, **kwargs: ""  # type: ignore[attr-defined]
sys.modules["answer"].call_gemini = lambda *args, **kwargs: ("", [])  # type: ignore[attr-defined]
sys.modules["answer"].retrieve = lambda *args, **kwargs: []  # type: ignore[attr-defined]
sys.modules["answer"].stream_gemini = lambda *args, **kwargs: iter(())  # type: ignore[attr-defined]
sys.modules["index_pdfs"].SUPPORTED_EXTENSIONS = {".txt"}  # type: ignore[attr-defined]
sys.modules["index_pdfs"].collection_for_project = lambda project: f"{project}_docs"  # type: ignore[attr-defined]
sys.modules["settings"].get_data_root = lambda: cache_dir.name  # type: ignore[attr-defined]
sys.modules["settings"].set_data_root = lambda path: path  # type: ignore[attr-defined]

from web import _answer_cache_key, _read_answer_cache, _write_answer_cache  # noqa: E402


def test_answer_cache_key_changes_with_context_text() -> None:
    first = [{"source": "a.txt", "locator": "document", "page": None, "text": "alpha"}]
    second = [{"source": "a.txt", "locator": "document", "page": None, "text": "beta"}]

    assert _answer_cache_key("inbox", "q", 12, False, first) != _answer_cache_key("inbox", "q", 12, False, second)
    print("test_answer_cache_key_changes_with_context_text PASSED")


def test_answer_cache_roundtrip() -> None:
    contexts = [{"source": "a.txt", "locator": "document", "page": None, "text": "alpha"}]
    key = _answer_cache_key("inbox", "q", 12, False, contexts)

    _write_answer_cache(key, {"answer": "cached answer", "web_sources": []})
    cached = _read_answer_cache(key)

    assert cached is not None
    assert cached["answer"] == "cached answer"
    print("test_answer_cache_roundtrip PASSED")


if __name__ == "__main__":
    test_answer_cache_key_changes_with_context_text()
    test_answer_cache_roundtrip()
    cache_dir.cleanup()
    print("\nAll tests PASSED")
