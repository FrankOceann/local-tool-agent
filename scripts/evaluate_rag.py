from app.rag import (
    RAG_EVALUATION_CASES,
    evaluate_retrieval_cases,
    format_evaluation_results,
)
from tools.rag_tools import get_rag_index


def main() -> None:
    index = get_rag_index()
    results = evaluate_retrieval_cases(
        index,
        RAG_EVALUATION_CASES,
    )
    print(format_evaluation_results(results))


if __name__ == "__main__":
    main()