import os
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI


EMBEDDING_MODEL_NAME = "text-embedding-3-small"
MISSING_EMBEDDING_KEY_MESSAGE = "未检测到 OPENAI_API_KEY，请在 .env 中配置后重试。"


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        client: object | None = None,
        api_key: str | None = None,
    ):
        load_dotenv()
        self.api_key = (
            api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        )
        self.client = client

        if self.client is None and self.api_key:
            self.client = OpenAI(api_key=self.api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise ValueError(MISSING_EMBEDDING_KEY_MESSAGE)

        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL_NAME,
                input=texts,
            )
        except Exception as error:
            raise ValueError(f"Embedding 服务调用失败：{error}") from error
        return [item.embedding for item in response.data]
