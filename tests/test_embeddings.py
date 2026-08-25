import pytest

from app.embeddings import DashScopeEmbeddingProvider


class FakeEmbeddings:
    def create(self, *, model, input):
        assert model == "text-embedding-v4"
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


def test_dashscope_provider_returns_embeddings_from_client():
    provider = DashScopeEmbeddingProvider(client=FakeClient(), api_key="test-key")

    assert provider.embed_texts(["第一段", "第二段"]) == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]


def test_dashscope_provider_batches_more_than_ten_texts_in_order():
    class RecordingEmbeddings:
        def __init__(self):
            self.inputs = []

        def create(self, *, model, input):
            assert model == "text-embedding-v4"
            self.inputs.append(input)
            return type(
                "Response",
                (),
                {
                    "data": [
                        type("Item", (), {"embedding": [float(text[2:])]})()
                        for text in input
                    ]
                },
            )()

    embeddings = RecordingEmbeddings()
    client = type("Client", (), {"embeddings": embeddings})()
    provider = DashScopeEmbeddingProvider(client=client, api_key="test-key")

    texts = [f"文本{number}" for number in range(11)]

    assert provider.embed_texts(texts) == [
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
        [7.0],
        [8.0],
        [9.0],
        [10.0],
    ]
    assert embeddings.inputs == [texts[:10], texts[10:]]


def test_dashscope_provider_reports_missing_key_without_network():
    provider = DashScopeEmbeddingProvider(api_key="")

    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        provider.embed_texts(["第一段"])


def test_dashscope_provider_does_not_expose_key_on_api_error():
    class BrokenEmbeddings:
        def create(self, **kwargs):
            raise RuntimeError("rate limited")

    client = type("Client", (), {"embeddings": BrokenEmbeddings()})()
    provider = DashScopeEmbeddingProvider(client=client, api_key="secret-value")

    with pytest.raises(
        ValueError,
        match="Embedding 服务调用失败：rate limited",
    ) as error:
        provider.embed_texts(["第一段"])

    assert "secret-value" not in str(error.value)
