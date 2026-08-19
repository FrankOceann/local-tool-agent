from types import SimpleNamespace

from llm_agent import LLMToolAgent, MISSING_KEY_MESSAGE


class FakeClient:
    def __init__(self, responses: list[object]):
        self.calls: list[dict] = []
        self.responses = iter(responses)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create),
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


def response_with(message: object) -> object:
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_agent_explains_how_to_configure_a_missing_api_key():
    assert LLMToolAgent(api_key="").run("大写 hello") == MISSING_KEY_MESSAGE


def test_agent_returns_a_direct_model_response_without_tools():
    client = FakeClient(
        [
            response_with(
                SimpleNamespace(content="你好，我可以处理文本。", tool_calls=None),
            )
        ]
    )

    assert LLMToolAgent(client=client, api_key="test-key").run("你能做什么？") == "你好，我可以处理文本。"


def test_agent_executes_first_tool_call_and_returns_final_model_response():
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="upper_text", arguments='{"text": "hello"}'),
    )
    client = FakeClient(
        [
            response_with(SimpleNamespace(content=None, tool_calls=[tool_call])),
            response_with(SimpleNamespace(content="处理结果是 HELLO", tool_calls=None)),
        ]
    )

    assert LLMToolAgent(client=client, api_key="test-key").run("把 hello 转成大写") == "处理结果是 HELLO"
    assert client.calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "HELLO",
    }
