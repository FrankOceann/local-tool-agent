import importlib.util
from copy import deepcopy
from types import SimpleNamespace

from app.config import MISSING_KEY_MESSAGE
from app.config import SYSTEM_PROMPT
from app.llm_agent import LLMToolAgent


def test_system_prompt_requires_real_rag_sources():
    assert "search_knowledge_base" in SYSTEM_PROMPT
    assert "不要编造来源" in SYSTEM_PROMPT
    assert "只调用 search_knowledge_base" in SYSTEM_PROMPT
    assert "每个请求最多调用一次" in SYSTEM_PROMPT
from app.tool_schemas import TOOL_SCHEMAS

class FakeClient:
    def __init__(self, responses: list[object]):
        self.calls: list[dict] = []
        self.responses = iter(responses)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create),
        )

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
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


def test_agent_rejects_tool_call_markup_returned_as_plain_text():
    client = FakeClient(
        [
            response_with(
                SimpleNamespace(
                    content="<tool_calls><invoke name=\"read_files\"></invoke></tool_calls>",
                    tool_calls=None,
                )
            )
        ]
    )

    result = LLMToolAgent(client=client, api_key="test-key").run("读取资料")

    assert result == "模型没有返回可执行的结构化工具调用，请重新提问。"


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
    assert "tools" in client.calls[1]
    assert client.calls[1]["tools"] == TOOL_SCHEMAS
    assert client.calls[1].get("tool_choice") == "auto"


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

def test_agent_sends_system_identity_and_capability_boundaries_to_model():
    client = FakeClient(
        [
            response_with(
                SimpleNamespace(content="我是本地文本助手。", tool_calls=None),
            )
        ]
    )

    LLMToolAgent(client=client, api_key="test-key").run("你是谁？")

    first_message = client.calls[0]["messages"][0]
    assert first_message["role"] == "system"
    assert "DeepSeek 驱动" in first_message["content"]
    assert "不是 Claude、Anthropic" in first_message["content"]
    assert "联网" in first_message["content"]
    assert "summarize_text" in first_message["content"]

def test_agent_executes_summarize_text_tool():
    tool_call = SimpleNamespace(
        id="call_summary",
        function=SimpleNamespace(
            name="summarize_text",
            arguments='{"text": "第一句。第二句。第三句。"}',
        ),
    )
    client = FakeClient(
        [
            response_with(SimpleNamespace(content=None, tool_calls=[tool_call])),
            response_with(SimpleNamespace(content="摘要：第一句。第二句。", tool_calls=None)),
        ]
    )

    result = LLMToolAgent(client=client, api_key="test-key").run("请总结这段文字")

    assert result == "摘要：第一句。第二句。"
    assert client.calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_summary",
        "content": "第一句。第二句。",
    }

def test_agent_executes_two_tools_in_sequence_before_answering():
    first_tool_call = SimpleNamespace(
        id="call_upper",
        function=SimpleNamespace(
            name="upper_text",
            arguments='{"text": "hello agent"}',
        ),
    )
    second_tool_call = SimpleNamespace(
        id="call_count",
        function=SimpleNamespace(
            name="count_words",
            arguments='{"text": "HELLO AGENT"}',
        ),
    )
    client = FakeClient(
        [
            response_with(SimpleNamespace(content=None, tool_calls=[first_tool_call])),
            response_with(SimpleNamespace(content=None, tool_calls=[second_tool_call])),
            response_with(
                SimpleNamespace(
                    content="大写后共有 2 个英文单词。",
                    tool_calls=None,
                )
            ),
        ]
    )

    result = LLMToolAgent(client=client, api_key="test-key").run(
        "先把 hello agent 转为大写，再统计单词数"
    )

    assert result == "大写后共有 2 个英文单词。"
    assert len(client.calls) == 3
    assert client.calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_upper",
        "content": "HELLO AGENT",
    }
    assert client.calls[2]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_count",
        "content": "2",
    }
def test_agent_forces_final_answer_after_three_tool_calls():
    first_tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(
            name="upper_text",
            arguments='{"text": "one"}',
        ),
    )
    second_tool_call = SimpleNamespace(
        id="call_2",
        function=SimpleNamespace(
            name="upper_text",
            arguments='{"text": "two"}',
        ),
    )
    third_tool_call = SimpleNamespace(
        id="call_3",
        function=SimpleNamespace(
            name="upper_text",
            arguments='{"text": "three"}',
        ),
    )
    client = FakeClient(
        [
            response_with(SimpleNamespace(content=None, tool_calls=[first_tool_call])),
            response_with(SimpleNamespace(content=None, tool_calls=[second_tool_call])),
            response_with(SimpleNamespace(content=None, tool_calls=[third_tool_call])),
            response_with(
                SimpleNamespace(
                    content="已完成三次工具调用。",
                    tool_calls=None,
                )
            ),
        ]
    )

    result = LLMToolAgent(client=client, api_key="test-key").run(
        "连续处理三段文本"
    )

    assert result == "已完成三次工具调用。"
    assert len(client.calls) == 4
    assert client.calls[3]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_3",
        "content": "THREE",
    }
    assert "tools" not in client.calls[3]
    assert "tool_choice" not in client.calls[3]

def test_agent_executes_two_tool_calls_from_one_model_response():
    upper_call = SimpleNamespace(
        id="call_upper",
        function=SimpleNamespace(
            name="upper_text",
            arguments='{"text": "hello"}',
        ),
    )
    count_call = SimpleNamespace(
        id="call_count",
        function=SimpleNamespace(
            name="count_words",
            arguments='{"text": "I am learning"}',
        ),
    )
    client = FakeClient(
        [
            response_with(
                SimpleNamespace(
                    content=None,
                    tool_calls=[upper_call, count_call],
                )
            ),
            response_with(
                SimpleNamespace(
                    content="大写结果是 HELLO，英文单词数量是 3。",
                    tool_calls=None,
                )
            ),
        ]
    )

    result = LLMToolAgent(client=client, api_key="test-key").run(
        "同时处理两段文本"
    )

    assert result == "大写结果是 HELLO，英文单词数量是 3。"
    assert len(client.calls) == 2
    assert client.calls[1]["messages"][-2:] == [
        {
            "role": "tool",
            "tool_call_id": "call_upper",
            "content": "HELLO",
        },
        {
            "role": "tool",
            "tool_call_id": "call_count",
            "content": "3",
        },
    ]

def test_agent_rejects_four_tool_calls_in_one_model_response():
    tool_calls = [
        SimpleNamespace(
            id=f"call_{index}",
            function=SimpleNamespace(
                name="upper_text",
                arguments=f'{{"text": "text {index}"}}',
            ),
        )
        for index in range(4)
    ]
    client = FakeClient(
        [
            response_with(
                SimpleNamespace(
                    content=None,
                    tool_calls=tool_calls,
                )
            )
        ]
    )
    agent = LLMToolAgent(client=client, api_key="test-key")
    agent.tool_registry["upper_text"] = lambda _text: (_ for _ in ()).throw(
        AssertionError("超过上限时不应执行任何工具")
    )

    assert agent.run("一次调用四个工具") == "本次请求最多执行 3 次工具调用。"
    assert len(client.calls) == 1

def test_agent_rejects_a_batch_that_exceeds_remaining_tool_call_limit():
    call_1 = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(
            name="upper_text",
            arguments='{"text": "one"}',
        ),
    )
    call_2 = SimpleNamespace(
        id="call_2",
        function=SimpleNamespace(
            name="upper_text",
            arguments='{"text": "two"}',
        ),
    )
    call_3 = SimpleNamespace(
        id="call_3",
        function=SimpleNamespace(
            name="upper_text",
            arguments='{"text": "three"}',
        ),
    )
    client = FakeClient(
        [
            response_with(
                SimpleNamespace(
                    content=None,
                    tool_calls=[call_1, call_2],
                )
            ),
            response_with(
                SimpleNamespace(
                    content=None,
                    tool_calls=[call_3, call_1],
                )
            ),
        ]
    )
    agent = LLMToolAgent(client=client, api_key="test-key")
    executed_texts = []
    agent.tool_registry["upper_text"] = (
        lambda text: executed_texts.append(text) or text.upper()
    )

    result = agent.run("连续调用")

    assert result == "本次请求最多执行 3 次工具调用。"
    assert len(client.calls) == 2
    assert executed_texts == ["one", "two"]

def test_agent_stops_when_a_later_tool_in_one_batch_is_invalid():
    good_call = SimpleNamespace(
        id="call_good",
        function=SimpleNamespace(
            name="upper_text",
            arguments='{"text": "hello"}',
        ),
    )
    bad_call = SimpleNamespace(
        id="call_bad",
        function=SimpleNamespace(
            name="upper_text",
            arguments="not-json",
        ),
    )
    skipped_call = SimpleNamespace(
        id="call_skipped",
        function=SimpleNamespace(
            name="count_words",
            arguments='{"text": "one two"}',
        ),
    )
    client = FakeClient(
        [
            response_with(
                SimpleNamespace(
                    content=None,
                    tool_calls=[good_call, bad_call, skipped_call],
                )
            )
        ]
    )
    agent = LLMToolAgent(client=client, api_key="test-key")
    agent.tool_registry["count_words"] = lambda _text: (
        _ for _ in ()
    ).throw(AssertionError("失败后不应执行后续工具"))

    result = agent.run("批内包含错误参数")

    assert result == "模型返回的工具参数不是有效 JSON。"
    assert len(client.calls) == 1

def test_agent_requires_confirmation_before_simulating_note_save():
    tool_call = SimpleNamespace(
        id="call_save",
        function=SimpleNamespace(
            name="save_note",
            arguments='{"text": "明天学习 Agent"}',
        ),
    )
    client = FakeClient(
        [
            response_with(
                SimpleNamespace(content=None, tool_calls=[tool_call]),
            ),
        ]
    )
    agent = LLMToolAgent(client=client, api_key="test-key")

    assert agent.run("保存一条笔记") == (
        "操作需要确认：将模拟保存笔记“明天学习 Agent”。请输入“确认”或“取消”。"
    )
    assert len(client.calls) == 1

    assert agent.run("确认") == "已模拟保存笔记：明天学习 Agent"
    assert len(client.calls) == 1
    assert agent.pending_tool_call is None

def create_agent_with_pending_note_save():
    tool_call = SimpleNamespace(
        id="call_save",
        function=SimpleNamespace(
            name="save_note",
            arguments='{"text": "明天学习 Agent"}',
        ),
    )
    client = FakeClient(
        [
            response_with(
                SimpleNamespace(content=None, tool_calls=[tool_call]),
            ),
        ]
    )
    agent = LLMToolAgent(client=client, api_key="test-key")

    assert agent.run("保存一条笔记") == (
        "操作需要确认：将模拟保存笔记“明天学习 Agent”。请输入“确认”或“取消”。"
    )
    return agent, client


def test_agent_cancels_a_pending_note_save():
    agent, client = create_agent_with_pending_note_save()

    assert agent.run("取消") == "已取消待确认的操作。"
    assert agent.pending_tool_call is None
    assert len(client.calls) == 1


def test_agent_reports_when_confirmation_has_nothing_to_confirm():
    agent = LLMToolAgent(client=FakeClient([]), api_key="test-key")

    assert agent.run("确认") == "当前没有待确认的操作。"
    assert agent.run("取消") == "当前没有待确认的操作。"


def test_agent_keeps_pending_operation_when_user_enters_other_text():
    agent, client = create_agent_with_pending_note_save()

    assert agent.run("换一条笔记") == (
        "当前有待确认的操作，请输入“确认”或“取消”。"
    )
    assert agent.pending_tool_call is not None
    assert len(client.calls) == 1

def test_agent_rejects_mixed_confirmation_and_auto_tools_in_one_batch():
    upper_call = SimpleNamespace(
        id="call_upper",
        function=SimpleNamespace(
            name="upper_text",
            arguments='{"text": "hello"}',
        ),
    )
    save_call = SimpleNamespace(
        id="call_save",
        function=SimpleNamespace(
            name="save_note",
            arguments='{"text": "明天学习 Agent"}',
        ),
    )
    client = FakeClient(
        [
            response_with(
                SimpleNamespace(
                    content=None,
                    tool_calls=[upper_call, save_call],
                )
            ),
        ]
    )
    agent = LLMToolAgent(client=client, api_key="test-key")

    agent.tool_registry["upper_text"] = lambda _text: (
        _ for _ in ()
    ).throw(AssertionError("混合批次不应执行自动工具"))
    agent.tool_registry["save_note"] = lambda _text: (
        _ for _ in ()
    ).throw(AssertionError("混合批次不应执行敏感工具"))

    assert agent.run("大写并保存") == (
        "同一轮请求包含需要确认的工具，当前不支持与其他工具混合执行。"
    )
    assert len(client.calls) == 1
    assert agent.pending_tool_call is None

def test_agent_handles_confirmation_commands_without_an_api_key():
    agent = LLMToolAgent(api_key="")

    assert agent.run("确认") == "当前没有待确认的操作。"
    assert agent.run("取消") == "当前没有待确认的操作。"

def test_config_module_exists():
    try:
        module_spec = importlib.util.find_spec("app.config")
    except ModuleNotFoundError:
        module_spec = None

    assert module_spec is not None

def test_tool_schemas_module_provides_all_registered_tools():
    try:
        from app.tool_schemas import TOOL_SCHEMAS
    except ModuleNotFoundError:
        TOOL_SCHEMAS = []

    tool_names = [tool["function"]["name"] for tool in TOOL_SCHEMAS]

    assert tool_names == [
        "upper_text",
        "count_words",
        "summarize_text",
        "save_note",
        "read_file",
        "read_files",
        "search_files",
        "search_knowledge_base",
    ]

def test_app_llm_agent_module_exports_agent_class():
    try:
        from app.llm_agent import LLMToolAgent as AppLLMToolAgent
    except ModuleNotFoundError:
        AppLLMToolAgent = None

    assert AppLLMToolAgent is not None

def test_agent_tells_model_to_search_then_read_one_or_two_files():
    client = FakeClient(
        [
            response_with(
                SimpleNamespace(
                    content="我会先搜索，再读取相关文件。",
                    tool_calls=None,
                )
            )
        ]
    )

    LLMToolAgent(client=client, api_key="test-key").run("帮我查找资料")

    system_message = client.calls[0]["messages"][0]["content"]

    assert "search_files" in system_message
    assert "read_files" in system_message
    assert "最多两份候选文件" in system_message
    assert "summarize_text" in system_message
