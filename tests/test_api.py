from fastapi.testclient import TestClient

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