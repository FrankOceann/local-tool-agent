import pytest

from app.embeddings import OpenAIEmbeddingProvider


class FakeEmbeddings:
    def create(self, *, model, input):
        assert model == "text-embedding-3-small"
        assert input == ["第一段", "第二段"]
        return type(
            "Response",
            (),
            {
                "data": [
                    type("Item", (), {"embedding": [1.0, 0.0]})(),
                    type("Item", (), {"embedding": [0.0, 1.0]})(),
                ]
            },
        )()


class FakeClient:
    embeddings = FakeEmbeddings()


def test_openai_provider_returns_embeddings_from_client():
    provider = OpenAIEmbeddingProvider(client=FakeClient(), api_key="test-key")

    assert provider.embed_texts(["第一段", "第二段"]) == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]


def test_openai_provider_reports_missing_key_without_network():
    provider = OpenAIEmbeddingProvider(api_key="")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        provider.embed_texts(["第一段"])


def test_openai_provider_does_not_expose_key_on_api_error():
    class BrokenEmbeddings:
        def create(self, **kwargs):
            raise RuntimeError("rate limited")

    client = type("Client", (), {"embeddings": BrokenEmbeddings()})()
    provider = OpenAIEmbeddingProvider(client=client, api_key="secret-value")

    with pytest.raises(
        ValueError,
        match="Embedding 服务调用失败：rate limited",
    ) as error:
        provider.embed_texts(["第一段"])

    assert "secret-value" not in str(error.value)
