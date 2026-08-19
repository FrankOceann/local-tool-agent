# DeepSeek Tool Calling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a DeepSeek-backed command-line Agent that safely selects and runs one local text tool.

**Architecture:** `LLMToolAgent` owns DeepSeek requests and accepts an injected client so tests stay offline. It advertises two function schemas, allowlists the returned tool name, validates JSON arguments, executes one local tool, and submits that result for a final model answer. `main.py` is only the CLI boundary.

**Tech Stack:** Python 3.10, `openai`, `python-dotenv`, DeepSeek OpenAI-compatible API, pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-deepseek-tool-calling-design.md`

## Global Constraints

- Use `base_url="https://api.deepseek.com"` and `deepseek-v4-flash`.
- Read only `DEEPSEEK_API_KEY`; do not commit `.env`.
- Do not change `tools.py` or the rule-based `agent.py`.
- Allow only `upper_text` and `count_words`, and only one tool call per request.
- Tests use fake clients; automated tests make no network request.
- Run `python -m pytest -q` before every commit.

---

### Task 1: Add Safe Configuration and Agent Shell

**Files:** Create `.env.example`, `llm_agent.py`, `test_llm_agent.py`; modify `requirements.txt`.

**Interfaces:** `LLMToolAgent(client: object | None = None, api_key: str | None = None)` and `run(request: str) -> str`.

- [ ] **Step 1: Write the failing configuration test**

```python
from llm_agent import LLMToolAgent, MISSING_KEY_MESSAGE


def test_agent_explains_how_to_configure_a_missing_api_key():
    assert LLMToolAgent(api_key="").run("大写 hello") == MISSING_KEY_MESSAGE
```

- [ ] **Step 2: Confirm the test fails because the module is absent**

```bat
python -m pytest test_llm_agent.py::test_agent_explains_how_to_configure_a_missing_api_key -v
```

Expected: `ModuleNotFoundError: No module named 'llm_agent'`.

- [ ] **Step 3: Add dependencies and the public template**

Replace `requirements.txt` with:

```text
openai
python-dotenv
pytest
```

Create `.env.example`:

```text
DEEPSEEK_API_KEY=
```

- [ ] **Step 4: Implement the smallest passing shell**

```python
import os

from dotenv import load_dotenv


MISSING_KEY_MESSAGE = "未检测到 DEEPSEEK_API_KEY，请在 .env 中配置后重试。"


class LLMToolAgent:
    def __init__(self, client: object | None = None, api_key: str | None = None):
        load_dotenv()
        self.client = client
        self.api_key = api_key if api_key is not None else os.getenv("DEEPSEEK_API_KEY")

    def run(self, request: str) -> str:
        if not self.api_key:
            return MISSING_KEY_MESSAGE
        return ""
```

- [ ] **Step 5: Install and verify green**

```bat
python -m pip install -r requirements.txt
python -m pytest test_llm_agent.py::test_agent_explains_how_to_configure_a_missing_api_key -v
```

- [ ] **Step 6: Commit**

```bat
python -m pytest -q
git add requirements.txt .env.example llm_agent.py test_llm_agent.py
git commit -m "feat: add deepseek agent configuration"
```

### Task 2: Implement Direct Replies and One Approved Tool Call

**Files:** Modify `llm_agent.py`, `test_llm_agent.py`.

**Interfaces:** `TOOL_SCHEMAS: list[dict]`; `run(request: str) -> str` returns either direct text or the final answer after one tool result.

- [ ] **Step 1: Add fake-response helpers and failing tests**

```python
from types import SimpleNamespace


class FakeCompletions:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def response(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


def test_agent_returns_a_direct_model_answer():
    client = FakeClient([response(content="你好，我可以处理文本。")])
    assert LLMToolAgent(client=client, api_key="test").run("你会做什么？") == "你好，我可以处理文本。"


def test_agent_runs_upper_text_then_returns_the_final_model_answer():
    call = SimpleNamespace(id="call_1", function=SimpleNamespace(name="upper_text", arguments='{"text": "hello"}'))
    client = FakeClient([response(tool_calls=[call]), response(content="处理结果是 HELLO")])
    assert LLMToolAgent(client=client, api_key="test").run("大写 hello") == "处理结果是 HELLO"
    assert client.completions.calls[1]["messages"][-1] == {"role": "tool", "tool_call_id": "call_1", "content": "HELLO"}
```

- [ ] **Step 2: Confirm red**

```bat
python -m pytest test_llm_agent.py -v
```

Expected: the new tests fail because `run()` returns an empty string.

- [ ] **Step 3: Implement schemas, the DeepSeek client, and valid tool dispatch**

```python
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from tools import count_words, upper_text

MODEL_NAME = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
MISSING_KEY_MESSAGE = "未检测到 DEEPSEEK_API_KEY，请在 .env 中配置后重试。"
TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "upper_text", "description": "Convert text to uppercase.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "count_words", "description": "Count English words in text.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
]


class LLMToolAgent:
    def __init__(self, client: object | None = None, api_key: str | None = None):
        load_dotenv()
        self.api_key = api_key if api_key is not None else os.getenv("DEEPSEEK_API_KEY")
        self.client = client or (OpenAI(api_key=self.api_key, base_url=BASE_URL) if self.api_key else None)
        self.tool_registry = {"upper_text": upper_text, "count_words": count_words}

    def run(self, request: str) -> str:
        if not self.api_key or self.client is None:
            return MISSING_KEY_MESSAGE
        messages = [{"role": "user", "content": request}]
        first = self.client.chat.completions.create(model=MODEL_NAME, messages=messages, tools=TOOL_SCHEMAS, tool_choice="auto")
        message = first.choices[0].message
        if not message.tool_calls:
            return message.content or "模型没有返回可显示的回答。"
        call = message.tool_calls[0]
        result, error = self._run_tool(call.function.name, call.function.arguments)
        if error:
            return error
        messages.append(message)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
        final = self.client.chat.completions.create(model=MODEL_NAME, messages=messages)
        return final.choices[0].message.content or "模型没有返回可显示的回答。"

    def _run_tool(self, name: str, arguments_json: str) -> tuple[str | int | None, str | None]:
        arguments = json.loads(arguments_json)
        return self.tool_registry[name](arguments["text"]), None
```

- [ ] **Step 4: Verify green and commit**

```bat
python -m pytest -q
git add llm_agent.py test_llm_agent.py
git commit -m "feat: add deepseek tool calling loop"
```

### Task 3: Validate Model-Returned Tools and Arguments

**Files:** Modify `llm_agent.py`, `test_llm_agent.py`.

**Interfaces:** `_run_tool(name: str, arguments_json: str) -> tuple[str | int | None, str | None]` returns `(result, None)` for valid calls and `(None, error_message)` for rejected calls.

- [ ] **Step 1: Add failing safety tests**

```python
def tool_response(name, arguments):
    call = SimpleNamespace(id="call_invalid", function=SimpleNamespace(name=name, arguments=arguments))
    return response(tool_calls=[call])


def test_agent_rejects_an_unknown_tool():
    client = FakeClient([tool_response("delete_everything", '{"text": "hello"}')])
    assert LLMToolAgent(client=client, api_key="test").run("危险请求") == "模型请求了不支持的工具: delete_everything"


def test_agent_rejects_malformed_json_arguments():
    client = FakeClient([tool_response("upper_text", "not-json")])
    assert LLMToolAgent(client=client, api_key="test").run("大写 hello") == "模型返回的工具参数不是有效 JSON。"


def test_agent_rejects_a_non_string_text_argument():
    client = FakeClient([tool_response("upper_text", '{"text": 123}')])
    assert LLMToolAgent(client=client, api_key="test").run("大写 123") == "工具参数 text 必须是字符串。"
```

- [ ] **Step 2: Confirm red**

```bat
python -m pytest test_llm_agent.py -v
```

Expected: `KeyError` for the unknown name, `JSONDecodeError` for malformed JSON, and incorrect acceptance of the non-string argument.

- [ ] **Step 3: Replace `_run_tool` with ordered validation**

```python
    def _run_tool(self, name: str, arguments_json: str) -> tuple[str | int | None, str | None]:
        if name not in self.tool_registry:
            return None, f"模型请求了不支持的工具: {name}"
        try:
            arguments = json.loads(arguments_json)
        except json.JSONDecodeError:
            return None, "模型返回的工具参数不是有效 JSON。"
        text = arguments.get("text") if isinstance(arguments, dict) else None
        if not isinstance(text, str):
            return None, "工具参数 text 必须是字符串。"
        return self.tool_registry[name](text), None
```

- [ ] **Step 4: Return validation errors before a second API call**

Replace the Task 2 dispatch line with this explicit result-and-error handling:

```python
        result, error = self._run_tool(call.function.name, call.function.arguments)
        if error:
            return error
```

- [ ] **Step 5: Verify green and commit**

```bat
python -m pytest -q
git add llm_agent.py test_llm_agent.py
git commit -m "feat: validate deepseek tool calls"
```

### Task 4: Run the LLM Agent from the CLI and Document Setup

**Files:** Modify `main.py`, `test_main.py`, `README.md`.

**Interfaces:** `main.run_once(request: str) -> str` delegates to `LLMToolAgent.run(request)`.

- [ ] **Step 1: Replace the entry-point test with a failing LLM-agent test**

```python
from main import run_once


def test_run_once_returns_the_llm_agent_result(monkeypatch):
    from llm_agent import LLMToolAgent
    monkeypatch.setattr(LLMToolAgent, "run", lambda self, request: "MODEL RESULT")
    assert run_once("任意请求") == "MODEL RESULT"
```

- [ ] **Step 2: Confirm red**

```bat
python -m pytest test_main.py::test_run_once_returns_the_llm_agent_result -v
```

Expected: FAIL because `main.py` still constructs `LocalToolAgent`.

- [ ] **Step 3: Change the CLI dependency**

Replace the first lines of `main.py` with:

```python
from llm_agent import LLMToolAgent


agent = LLMToolAgent()
```

Keep `run_once()` and the existing `input()` block unchanged.

- [ ] **Step 4: Add the safe setup section to `README.md`**

````markdown
## DeepSeek 配置

1. 安装依赖：`python -m pip install -r requirements.txt`。
2. 在项目根目录创建 `.env`：

```text
DEEPSEEK_API_KEY=你的_DeepSeek_API_Key
```

`.env` 已被 Git 忽略；不要把真实 API Key 写进代码、README 或提交记录。
````

Replace the flow with: `用户输入 → LLMToolAgent → DeepSeek 选择工具 → tools.py 执行 → 工具结果回传 DeepSeek → 输出最终回答`.

- [ ] **Step 5: Verify automatically and once with the real API**

```bat
python -m pytest -q
python main.py
```

At the prompt enter `请把 hello agent 转为大写`. The final response should contain `HELLO AGENT`. Never paste an API key into a screenshot.

- [ ] **Step 6: Commit**

```bat
git add main.py test_main.py README.md
git commit -m "feat: run deepseek tool agent from cli"
```

### Task 5: Verify the Publishable State

**Files:** Verify `.gitignore`, `.env.example`, `README.md`, `llm_agent.py`, `test_llm_agent.py`.

- [ ] **Step 1: Confirm no secret is tracked**

```bat
git ls-files .env
```

Expected: no output.

- [ ] **Step 2: Run all offline tests**

```bat
python -m pytest -q
```

Expected: every test passes without a network request.

- [ ] **Step 3: Review the final change list**

```bat
git status --short
git diff --check
```

Expected: `.env` is absent and the diff check has no output.
