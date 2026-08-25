from scripts import evaluate_rag


def test_main_runs_all_top_k_experiment_groups(monkeypatch, capsys):
    index = object()
    evaluation_top_ks = []

    monkeypatch.setattr(
        evaluate_rag,
        "get_rag_index",
        lambda: index,
    )

    def fake_evaluate_retrieval_cases(
        actual_index,
        cases,
        top_k=3,
    ):
        assert actual_index is index
        assert cases is evaluate_rag.RAG_EVALUATION_CASES
        evaluation_top_ks.append(top_k)
        return []

    monkeypatch.setattr(
        evaluate_rag,
        "evaluate_retrieval_cases",
        fake_evaluate_retrieval_cases,
    )
    monkeypatch.setattr(
        evaluate_rag,
        "format_evaluation_results",
        lambda results, top_k=3: f"Top-{top_k} 评测结果",
    )

    evaluate_rag.main()

    assert evaluation_top_ks == [1, 2, 3]
    assert capsys.readouterr().out == (
        "Top-1 评测结果\n\n"
        "Top-2 评测结果\n\n"
        "Top-3 评测结果\n\n"
    )