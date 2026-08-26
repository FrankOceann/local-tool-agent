from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.rag import RAGIndex
from tools.rag_tools import get_rag_index


app = FastAPI(title="Local Tool Agent RAG API")


class QueryRequest(BaseModel):
    question: str
    top_k: Annotated[int, Field(ge=1, le=3)] = 3

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question 不能为空")
        return value


class QueryResult(BaseModel):
    source: str
    score: float
    content: str


class QueryResponse(BaseModel):
    results: list[QueryResult]


def get_index() -> RAGIndex:
    try:
        return get_rag_index()
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="知识库初始化失败，请稍后重试。",
        ) from error


@app.post("/knowledge-base/query", response_model=QueryResponse)
def query_knowledge_base(
    request: QueryRequest,
    index: Annotated[RAGIndex, Depends(get_index)],
) -> QueryResponse:
    try:
        results = index.search(request.question, top_k=request.top_k)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="知识库查询失败，请稍后重试。",
    ) from error

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