import pytest

from tools import rag_tools


class FakeIndex:
    def search(self, query, top_k=3):
        assert query == "如何保护文件？"
        assert top_k == 3
        chunk = type(
            "Chunk",
            (),
            {
                "source_file": "agent_safety.txt",
                "chunk_index": 0,
                "text": "文件访问必须校验权限。",
            },
        )()
        return [type("Result", (), {"chunk": chunk, "score": 0.87321})()]


def test_search_knowledge_base_formats_source_and_score(monkeypatch):
    monkeypatch.setattr(rag_tools, "get_rag_index", lambda: FakeIndex())

    assert rag_tools.search_knowledge_base("如何保护文件？") == (
        "[来源: agent_safety.txt#chunk-0 | score=0.8732]\n"
        "文件访问必须校验权限。"
    )


def test_search_knowledge_base_rejects_blank_query_without_building_index(
    monkeypatch,
):
    monkeypatch.setattr(
        rag_tools,
        "get_rag_index",
        lambda: pytest.fail("不应建库"),
    )

    assert rag_tools.search_knowledge_base("  ") == "检索问题不能为空。"


def test_search_knowledge_base_returns_a_clear_message_for_no_results(
    monkeypatch,
):
    index = type("Index", (), {"search": lambda self, query, top_k: []})()
    monkeypatch.setattr(rag_tools, "get_rag_index", lambda: index)

    assert (
        rag_tools.search_knowledge_base("未知主题")
        == "知识库中没有可用的相关资料。"
    )
