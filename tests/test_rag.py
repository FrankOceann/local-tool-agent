import pytest

from app.rag import CHUNK_OVERLAP, CHUNK_SIZE, RAGIndex, split_text


def test_split_text_preserves_source_indexes_and_overlap():
    chunks = split_text("lesson.txt", "a" * (CHUNK_SIZE + 20))

    assert [(item.source_file, item.chunk_index) for item in chunks] == [
        ("lesson.txt", 0),
        ("lesson.txt", 1),
    ]
    assert chunks[0].text == "a" * CHUNK_SIZE
    assert chunks[1].text == "a" * (20 + CHUNK_OVERLAP)


def test_split_text_ignores_whitespace_only_documents():
    assert split_text("empty.txt", " \n\t ") == []


class FakeEmbeddingProvider:
    vectors = {
        "Python 用 pathlib 读取文件。": [1.0, 0.0],
        "Agent 需要工具权限校验。": [0.0, 1.0],
        "怎样安全读取资料？": [0.0, 1.0],
    }

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


def test_index_returns_results_in_descending_similarity(tmp_path):
    (tmp_path / "python.txt").write_text(
        "Python 用 pathlib 读取文件。",
        encoding="utf-8",
    )
    (tmp_path / "safety.txt").write_text(
        "Agent 需要工具权限校验。",
        encoding="utf-8",
    )
    index = RAGIndex.build(tmp_path, FakeEmbeddingProvider())

    results = index.search("怎样安全读取资料？", top_k=3)

    assert [(item.chunk.source_file, item.chunk.chunk_index) for item in results] == [
        ("safety.txt", 0),
        ("python.txt", 0),
    ]
    assert results[0].score > results[1].score


def test_index_returns_empty_for_blank_query_or_no_chunks(tmp_path):
    index = RAGIndex.build(tmp_path, FakeEmbeddingProvider())

    assert index.search("   ") == []
    assert index.search("怎样安全读取资料？") == []


def test_index_rejects_mismatched_vector_dimensions(tmp_path):
    class BadProvider:
        def embed_texts(self, texts):
            return [[1.0, 0.0], [1.0]]

    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")

    with pytest.raises(ValueError, match="向量维度不一致"):
        RAGIndex.build(tmp_path, BadProvider())
