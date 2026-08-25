from app.rag import (
    RAG_EVALUATION_CASES,
    evaluate_retrieval_cases,
    format_evaluation_results,
)
from tools.rag_tools import get_rag_index


def main() -> None:
    index = get_rag_index()

    for top_k in (1, 2, 3):
        results = evaluate_retrieval_cases(
            index,
            RAG_EVALUATION_CASES,
            top_k=top_k,
        )
        print(format_evaluation_results(results, top_k=top_k))
        print()

if __name__ == "__main__":
    main()