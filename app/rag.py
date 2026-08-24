from dataclasses import dataclass


CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


@dataclass(frozen=True)
class DocumentChunk:
    source_file: str
    chunk_index: int
    text: str


def split_text(source_file: str, text: str) -> list[DocumentChunk]:
    if not text.strip():
        return []

    step = CHUNK_SIZE - CHUNK_OVERLAP
    return [
        DocumentChunk(source_file, index, text[start:start + CHUNK_SIZE])
        for index, start in enumerate(range(0, len(text), step))
        if text[start:start + CHUNK_SIZE]
    ]
