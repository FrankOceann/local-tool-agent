from app import rag
from app.rag import DocumentChunk, SearchResult


def test_has_expected_source_recognizes_a_source_in_top_three():
    results = [
        SearchResult(
            DocumentChunk("agent_safety.txt", 0, "权限校验。"),
            0.91,
        ),
        SearchResult(
            DocumentChunk("rag_long_test.txt", 1, "副作用需确认。"),
            0.88,
        ),
        SearchResult(
            DocumentChunk("python_file_handling.txt", 0, "路径校验。"),
            0.72,
        ),
    ]

    assert rag.has_expected_source(
        results,
        "rag_long_test.txt#chunk-1",
    )

def test_has_expected_source_rejects_a_missing_source():
    results = [
        SearchResult(
            DocumentChunk("agent_safety.txt", 0, "权限校验。"),
            0.91,
        ),
        SearchResult(
            DocumentChunk("python_file_handling.txt", 0, "路径校验。"),
            0.72,
        ),
    ]

    assert not rag.has_expected_source(
        results,
        "rag_long_test.txt#chunk-1",
    )

def test_rag_evaluation_cases_include_the_five_manual_scenarios():
    cases = rag.RAG_EVALUATION_CASES

    assert [
        (case.name, case.expected_source)
        for case in cases
    ] == [
        ("副作用确认", "rag_long_test.txt#chunk-1"),
        ("路径穿越", "rag_long_test.txt#chunk-2"),
        ("Chunk 重叠", "rag_long_test.txt#chunk-4"),
        ("服务超时", "rag_long_test.txt#chunk-5"),
        ("资料不足", None),
    ]

def test_evaluate_retrieval_cases_reports_hit_and_retrieved_sources():
    class FakeIndex:
        def search(self, query):
            assert query == "如何确认副作用操作？"
            return [
                SearchResult(
                    DocumentChunk(
                        "rag_long_test.txt",
                        1,
                        "副作用操作需要用户确认。",
                    ),
                    0.88,
                ),
                SearchResult(
                    DocumentChunk(
                        "agent_safety.txt",
                        0,
                        "限制工具权限。",
                    ),
                    0.81,
                ),
            ]

    case = rag.RAGEvaluationCase(
        "副作用确认",
        "如何确认副作用操作？",
        "rag_long_test.txt#chunk-1",
    )

    results = rag.evaluate_retrieval_cases(
        FakeIndex(),
        (case,),
    )

    assert [
        (item.case.name, item.passed, item.retrieved_sources)
        for item in results
    ] == [
        (
            "副作用确认",
            True,
            (
                "rag_long_test.txt#chunk-1",
                "agent_safety.txt#chunk-0",
            ),
        ),
    ]

def test_evaluate_retrieval_cases_marks_missing_knowledge_as_manual_review():
    class FakeIndex:
        def search(self, query):
            assert query == "比较北京和上海今天的天气。"
            return [
                SearchResult(
                    DocumentChunk(
                        "rag_long_test.txt",
                        6,
                        "与资料无关时，应说明资料不足。",
                    ),
                    0.63,
                ),
            ]

    case = rag.RAGEvaluationCase(
        "资料不足",
        "比较北京和上海今天的天气。",
        None,
    )

    results = rag.evaluate_retrieval_cases(
        FakeIndex(),
        (case,),
    )

    assert results[0].passed is None
    assert results[0].retrieved_sources == (
        "rag_long_test.txt#chunk-6",
    )

def test_format_evaluation_results_counts_hits_and_manual_reviews():
    hit_case = rag.RAGEvaluationCase(
        "副作用确认",
        "如何确认副作用操作？",
        "rag_long_test.txt#chunk-1",
    )
    manual_case = rag.RAGEvaluationCase(
        "资料不足",
        "比较北京和上海今天的天气。",
        None,
    )
    results = [
        rag.RAGEvaluationResult(
            hit_case,
            True,
            ("rag_long_test.txt#chunk-1",),
        ),
        rag.RAGEvaluationResult(
            manual_case,
            None,
            ("rag_long_test.txt#chunk-6",),
        ),
    ]

    report = rag.format_evaluation_results(results)

    assert "[通过] 副作用确认" in report
    assert "[人工检查] 资料不足" in report
    assert "自动题命中：1/1" in report
    assert "人工检查：1" in report