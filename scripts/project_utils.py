import os
import re


DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".md", ".txt", ".hwp", ".hwpx"}
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".flac", ".aac", ".ogg", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
SUPPORTED_EXTENSIONS = DOCUMENT_EXTENSIONS | MEDIA_EXTENSIONS


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", value).strip("_")
    return cleaned or "default"


def collection_for_project(project: str) -> str:
    explicit = os.environ.get("COLLECTION_NAME", "").strip()
    default_project = os.environ.get("PROJECT_NAME", "inbox")
    if explicit and project == default_project:
        return explicit
    return f"{safe_name(project)}_docs"
