from app.embeddings import OpenAIEmbeddingProvider
from app.rag import RAGIndex
from tools.file_tools import DATA_DIRECTORY


_RAG_INDEX: RAGIndex | None = None


def get_rag_index() -> RAGIndex:
    global _RAG_INDEX
    if _RAG_INDEX is None:
        _RAG_INDEX = RAGIndex.build(
            DATA_DIRECTORY,
            OpenAIEmbeddingProvider(),
        )
    return _RAG_INDEX


def search_knowledge_base(text: str) -> str:
    query = text.strip()
    if not query:
        return "检索问题不能为空。"

    try:
        results = get_rag_index().search(query, top_k=3)
    except ValueError as error:
        return str(error)

    if not results:
        return "知识库中没有可用的相关资料。"

    return "\n\n".join(
        f"[来源: {item.chunk.source_file}#chunk-{item.chunk.chunk_index} "
        f"| score={item.score:.4f}]\n{item.chunk.text}"
        for item in results
    )
