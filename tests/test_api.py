from fastapi.testclient import TestClient
import pytest
from app.api import app, get_index
from app.rag import DocumentChunk, SearchResult


class FakeIndex:
    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        assert query == "如何确认副作用操作？"
        assert top_k == 2
        return [
            SearchResult(
                DocumentChunk(
                    "rag-long-test.txt",
                    1,
                    "副作用操作必须先由用户确认。",
                ),
                0.82,
            )
        ]


def test_query_knowledge_base_returns_structured_results():
    app.dependency_overrides[get_index] = lambda: FakeIndex()
    try:
        response = TestClient(app).post(
            "/knowledge-base/query",
            json={"question": "如何确认副作用操作？", "top_k": 2},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "source": "rag-long-test.txt#chunk-1",
                "score": 0.82,
                "content": "副作用操作必须先由用户确认。",
            }
        ]
    }

@pytest.mark.parametrize(
    "payload",
    [
        {"question": "   "},
        {"question": "问题", "top_k": 0},
        {"question": "问题", "top_k": 4},
        {"question": "问题", "top_k": "2"},
        {"question": "问题", "top_k": True},
        {"question": "问题", "top_k": 2.0},
    ],
)
def test_query_knowledge_base_rejects_invalid_request(payload):
    class EmptyIndex:
        def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
            return []

    app.dependency_overrides[get_index] = lambda: EmptyIndex()
    try:
        response = TestClient(app).post(
            "/knowledge-base/query",
            json=payload,
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422


def test_query_knowledge_base_returns_empty_results():
    class EmptyIndex:
        def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
            return []

    app.dependency_overrides[get_index] = lambda: EmptyIndex()
    try:
        response = TestClient(app).post(
            "/knowledge-base/query",
            json={"question": "资料库外的问题"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"results": []}


def test_query_knowledge_base_hides_internal_search_error():
    class BrokenIndex:
        def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
            raise RuntimeError("embedding provider secret detail")

    app.dependency_overrides[get_index] = lambda: BrokenIndex()
    try:
        response = TestClient(
            app,
            raise_server_exceptions=False,
        ).post(
            "/knowledge-base/query",
            json={"question": "会失败的问题"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json() == {
        "detail": "知识库查询失败，请稍后重试。"
    }