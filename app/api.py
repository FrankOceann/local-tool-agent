from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from app.rag import RAGIndex
from tools.rag_tools import get_rag_index


app = FastAPI(title="Local Tool Agent RAG API")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResult(BaseModel):
    source: str
    score: float
    content: str


class QueryResponse(BaseModel):
    results: list[QueryResult]


def get_index() -> RAGIndex:
    return get_rag_index()


@app.post("/knowledge-base/query", response_model=QueryResponse)
def query_knowledge_base(
    request: QueryRequest,
    index: Annotated[RAGIndex, Depends(get_index)],
) -> QueryResponse:
    results = index.search(request.question, top_k=request.top_k)

    return QueryResponse(
        results=[
            QueryResult(
                source=(
                    f"{result.chunk.source_file}#chunk-"
                    f"{result.chunk.chunk_index}"
                ),
                score=result.score,
                content=result.chunk.text,
            )
            for result in results
        ]
    )