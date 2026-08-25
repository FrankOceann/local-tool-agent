from dataclasses import dataclass
from math import sqrt
from pathlib import Path

from app.embeddings import EmbeddingProvider


CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


@dataclass(frozen=True)
class DocumentChunk:
    source_file: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float


def split_text(source_file: str, text: str) -> list[DocumentChunk]:
    if not text.strip():
        return []

    step = CHUNK_SIZE - CHUNK_OVERLAP
    return [
        DocumentChunk(source_file, index, text[start:start + CHUNK_SIZE])
        for index, start in enumerate(range(0, len(text), step))
        if text[start:start + CHUNK_SIZE]
    ]


class RAGIndex:
    def __init__(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
        provider: EmbeddingProvider,
    ):
        self.chunks = chunks
        self.vectors = vectors
        self.provider = provider

    @classmethod
    def build(
        cls,
        data_directory: Path,
        provider: EmbeddingProvider,
    ) -> "RAGIndex":
        chunks = []
        for file_path in sorted(data_directory.glob("*.txt")):
            text = file_path.read_text(encoding="utf-8")
            chunks.extend(split_text(file_path.name, text))

        if not chunks:
            return cls([], [], provider)

        vectors = provider.embed_texts([chunk.text for chunk in chunks])
        if vectors and any(
            len(vector) != len(vectors[0])
            for vector in vectors
        ):
            raise ValueError("向量维度不一致。")
        return cls(chunks, vectors, provider)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if not query.strip() or not self.chunks:
            return []

        query_vector = self.provider.embed_texts([query])[0]
        limit = min(max(top_k, 1), 3)
        results = [
            SearchResult(chunk, self._cosine_similarity(query_vector, vector))
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        return sorted(
            results,
            key=lambda result: (
                -result.score,
                result.chunk.source_file,
                result.chunk.chunk_index,
            ),
        )[:limit]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = sqrt(sum(value * value for value in left))
        right_norm = sqrt(sum(value * value for value in right))
        return numerator / (left_norm * right_norm)
