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
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


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


def test_agent_returns_readable_error_when_initial_model_request_fails():
    client = FakeClient([RuntimeError("network unavailable")])

    assert LLMToolAgent(client=client, api_key="test-key").run("大写 hello") == "调用模型服务失败，请稍后重试。"
    assert len(client.calls) == 1


def test_agent_executes_single_tool_call_and_returns_final_model_response():
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
    assert "tools" not in client.calls[1]
    assert "tool_choice" not in client.calls[1]


def test_agent_returns_readable_error_when_final_model_request_fails():
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="upper_text", arguments='{"text": "hello"}'),
    )
    client = FakeClient(
        [
            response_with(SimpleNamespace(content=None, tool_calls=[tool_call])),
            RuntimeError("network unavailable"),
        ]
    )

    assert LLMToolAgent(client=client, api_key="test-key").run("把 hello 转成大写") == "调用模型服务失败，请稍后重试。"
    assert len(client.calls) == 2


def test_agent_rejects_multiple_tool_calls_before_execution():
    first_tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="upper_text", arguments='{"text": "hello"}'),
    )
    second_tool_call = SimpleNamespace(
        id="call_2",
        function=SimpleNamespace(name="count_words", arguments='{"text": "hello world"}'),
    )
    client = FakeClient(
        [response_with(SimpleNamespace(content=None, tool_calls=[first_tool_call, second_tool_call]))]
    )
    agent = LLMToolAgent(client=client, api_key="test-key")
    agent.tool_registry["upper_text"] = lambda _text: (_ for _ in ()).throw(
        AssertionError("不应执行工具")
    )

    assert agent.run("同时调用两个工具") == "模型一次请求了多个工具，当前仅支持一个工具调用。"
    assert len(client.calls) == 1


def test_agent_rejects_unknown_tool_without_second_model_request():
    tool_call = SimpleNamespace(
        id="call_unknown",
        function=SimpleNamespace(name="delete_everything", arguments='{"text": "x"}'),
    )
    client = FakeClient([response_with(SimpleNamespace(content=None, tool_calls=[tool_call]))])

    assert LLMToolAgent(client=client, api_key="test-key").run("删除一切") == "模型请求了不支持的工具: delete_everything"
    assert len(client.calls) == 1


def test_agent_rejects_malformed_tool_arguments():
    tool_call = SimpleNamespace(
        id="call_bad_json",
        function=SimpleNamespace(name="upper_text", arguments="not-json"),
    )
    client = FakeClient([response_with(SimpleNamespace(content=None, tool_calls=[tool_call]))])

    assert LLMToolAgent(client=client, api_key="test-key").run("大写") == "模型返回的工具参数不是有效 JSON。"
    assert len(client.calls) == 1


def test_agent_rejects_non_string_text_argument():
    tool_call = SimpleNamespace(
        id="call_bad_text",
        function=SimpleNamespace(name="upper_text", arguments='{"text": 123}'),
    )
    client = FakeClient([response_with(SimpleNamespace(content=None, tool_calls=[tool_call]))])

    assert LLMToolAgent(client=client, api_key="test-key").run("大写") == "工具参数 text 必须是字符串。"
    assert len(client.calls) == 1
