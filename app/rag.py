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

@dataclass(frozen=True)
class RAGEvaluationCase:
    name: str
    question: str
    expected_source: str | None

@dataclass(frozen=True)
class RAGEvaluationResult:
    case: RAGEvaluationCase
    passed: bool | None
    retrieved_sources: tuple[str, ...]

RAG_EVALUATION_CASES = (
    RAGEvaluationCase(
        "副作用确认",
        "仅依据本地资料回答：为什么副作用操作必须先由用户确认？请标注每一点的来源。",
        "rag_long_test.txt#chunk-1",
    ),
    RAGEvaluationCase(
        "路径穿越",
        "仅依据本地资料回答：程序如何阻止 ../ 造成的路径穿越？请按步骤说明并标注来源。",
        "rag_long_test.txt#chunk-2",
    ),
    RAGEvaluationCase(
        "Chunk 重叠",
        "仅依据本地资料回答：Chunk 重叠有什么作用？为什么不能把整篇长文直接交给模型？标注来源。",
        "rag_long_test.txt#chunk-4",
    ),
    RAGEvaluationCase(
        "服务超时",
        "仅依据本地资料回答：Embedding 或模型服务超时时，系统应该如何处理？标注来源。",
        "rag_long_test.txt#chunk-5",
    ),
    RAGEvaluationCase(
        "资料不足",
        "仅依据本地资料回答：比较北京和上海今天的天气，并标注来源。",
        None,
    ),
)

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

def has_expected_source(
    results: list[SearchResult],
    expected_source: str,
) -> bool:
    return any(
        f"{result.chunk.source_file}#chunk-{result.chunk.chunk_index}"
        == expected_source
        for result in results
    )

def evaluate_retrieval_cases(
    index: RAGIndex,
    cases: tuple[RAGEvaluationCase, ...],
) -> list[RAGEvaluationResult]:
    evaluations = []

    for case in cases:
        results = index.search(case.question)
        retrieved_sources = tuple(
            f"{result.chunk.source_file}#chunk-{result.chunk.chunk_index}"
            for result in results
        )
        passed = (
            None
            if case.expected_source is None
            else has_expected_source(results, case.expected_source)
        )
        evaluations.append(
            RAGEvaluationResult(
                case,
                passed,
                retrieved_sources,
            )
        )

    return evaluations

def format_evaluation_results(
    results: list[RAGEvaluationResult],
) -> str:
    lines = []
    automatic_results = [
        result
        for result in results
        if result.passed is not None
    ]

    for result in results:
        if result.passed is True:
            status = "通过"
        elif result.passed is False:
            status = "未通过"
        else:
            status = "人工检查"

        sources = "、".join(result.retrieved_sources) or "无"
        lines.append(
            f"[{status}] {result.case.name} | 实际来源：{sources}"
        )

    hits = sum(
        result.passed is True
        for result in automatic_results
    )
    lines.append(
        f"自动题命中：{hits}/{len(automatic_results)}"
    )
    lines.append(
        f"人工检查：{len(results) - len(automatic_results)}"
    )
    return "\n".join(lines)