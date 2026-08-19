# 多步骤单工具 Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `LLMToolAgent` 在一次用户请求中最多连续执行三次单工具调用，并在达到上限后强制生成最终回答。

**Architecture:** `run()` 维护一份 `messages` 历史。每次工具执行后追加 assistant 消息和 tool 结果；循环中的模型请求携带工具 Schema，第三次工具完成后发起不带 Schema 的最终请求。

**Tech Stack:** Python 3.10、OpenAI Python SDK（DeepSeek 兼容接口）、pytest、python-dotenv。

**Spec:** `docs/superpowers/specs/2026-08-20-multi-step-tool-agent-design.md`

## Global Constraints

- 只使用 `upper_text`、`count_words`、`summarize_text`。
- 每轮模型响应只能有一个工具调用；多个调用返回 `MULTIPLE_TOOL_CALLS_MESSAGE`。
- 单次用户请求最多执行三次工具调用。
- 保持已有 API 异常和参数校验的中文错误信息。
- 不增加联网、记忆、文件访问、并行执行或新工具。

---

### Task 1: 用失败测试定义循环行为

**Files:**

- Modify: `test_llm_agent.py`
- Test: `test_llm_agent.py`

**Interfaces:**

- Consumes: `LLMToolAgent.run(request: str) -> str`、`TOOL_SCHEMAS`、`FakeClient`、`response_with()`。
- Produces: 连续两次工具调用和三次上限收尾的行为测试。

- [ ] **Step 1: 更新现有单工具测试的第二次请求断言**

将：

```python
from llm_agent import LLMToolAgent, MISSING_KEY_MESSAGE
```

替换为：

```python
from llm_agent import LLMToolAgent, MISSING_KEY_MESSAGE, TOOL_SCHEMAS
```

在 `test_agent_executes_single_tool_call_and_returns_final_model_response()` 中将：

```python
assert "tools" not in client.calls[1]
assert "tool_choice" not in client.calls[1]
```

替换为：

```python
assert client.calls[1]["tools"] == TOOL_SCHEMAS
assert client.calls[1]["tool_choice"] == "auto"
```

- [ ] **Step 2: 添加两次连续工具调用的测试**

在文件末尾添加：

```python
def test_agent_executes_two_tools_in_sequence_before_answering():
    first_tool_call = SimpleNamespace(
        id="call_upper",
        function=SimpleNamespace(name="upper_text", arguments='{"text": "hello agent"}'),
    )
    second_tool_call = SimpleNamespace(
        id="call_count",
        function=SimpleNamespace(name="count_words", arguments='{"text": "HELLO AGENT"}'),
    )
    client = FakeClient(
        [
            response_with(SimpleNamespace(content=None, tool_calls=[first_tool_call])),
            response_with(SimpleNamespace(content=None, tool_calls=[second_tool_call])),
            response_with(SimpleNamespace(content="大写后共有 2 个英文单词。", tool_calls=None)),
        ]
    )

    result = LLMToolAgent(client=client, api_key="test-key").run(
        "先把 hello agent 转为大写，再统计单词数"
    )

    assert result == "大写后共有 2 个英文单词。"
    assert len(client.calls) == 3
    assert client.calls[1]["messages"][-1] == {
        "role": "tool", "tool_call_id": "call_upper", "content": "HELLO AGENT"
    }
    assert client.calls[2]["messages"][-1] == {
        "role": "tool", "tool_call_id": "call_count", "content": "2"
    }
```

- [ ] **Step 3: 添加三次工具上限测试**

继续添加：

```python
def test_agent_forces_final_answer_after_three_tool_calls():
    first_tool_call = SimpleNamespace(
        id="call_1", function=SimpleNamespace(name="upper_text", arguments='{"text": "one"}')
    )
    second_tool_call = SimpleNamespace(
        id="call_2", function=SimpleNamespace(name="upper_text", arguments='{"text": "two"}')
    )
    third_tool_call = SimpleNamespace(
        id="call_3", function=SimpleNamespace(name="upper_text", arguments='{"text": "three"}')
    )
    client = FakeClient(
        [
            response_with(SimpleNamespace(content=None, tool_calls=[first_tool_call])),
            response_with(SimpleNamespace(content=None, tool_calls=[second_tool_call])),
            response_with(SimpleNamespace(content=None, tool_calls=[third_tool_call])),
            response_with(SimpleNamespace(content="已完成三次工具调用。", tool_calls=None)),
        ]
    )

    result = LLMToolAgent(client=client, api_key="test-key").run("连续处理三段文本")

    assert result == "已完成三次工具调用。"
    assert len(client.calls) == 4
    assert client.calls[3]["messages"][-1] == {
        "role": "tool", "tool_call_id": "call_3", "content": "THREE"
    }
    assert "tools" not in client.calls[3]
    assert "tool_choice" not in client.calls[3]
```

- [ ] **Step 4: 运行测试，确认当前实现失败**

运行：

```bat
python -m pytest -q test_llm_agent.py
```

预期：新多步骤测试失败；现有单工具测试也会因第二次请求当前未携带 `tools` 而失败。

### Task 2: 实现最多三轮的单工具循环

**Files:**

- Modify: `llm_agent.py`
- Test: `test_llm_agent.py`

**Interfaces:**

- Consumes: `TOOL_SCHEMAS`、`self._run_tool(name, arguments_json)` 返回 `(result, error)`。
- Produces: `LLMToolAgent.run(request: str) -> str` 在最多三次工具执行后返回最终文本或已有错误消息。

- [ ] **Step 1: 添加调用上限常量**

在 `BASE_URL` 下添加：

```python
MAX_TOOL_CALLS = 3
```

- [ ] **Step 2: 用循环替换 `run()` 中固定的一次工具流程**

保留 API Key 检查与 `messages` 初始化；将首次模型调用、单次工具处理和最终调用替换为：

```python
        for _ in range(MAX_TOOL_CALLS):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                )
            except Exception:
                return API_CALL_ERROR_MESSAGE

            assistant_message = response.choices[0].message
            if not assistant_message.tool_calls:
                return assistant_message.content or "模型没有返回可显示的回答。"

            if len(assistant_message.tool_calls) != 1:
                return MULTIPLE_TOOL_CALLS_MESSAGE

            tool_call = assistant_message.tool_calls[0]
            result, error = self._run_tool(
                tool_call.function.name, tool_call.function.arguments
            )
            if error:
                return error

            messages.append(assistant_message)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )

        try:
            final_response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
            )
        except Exception:
            return API_CALL_ERROR_MESSAGE
        return final_response.choices[0].message.content or "模型没有返回可显示的回答。"
```

- [ ] **Step 3: 验证 LLM Agent 测试**

运行：

```bat
python -m pytest -q test_llm_agent.py
```

预期：`13 passed`。

- [ ] **Step 4: 验证完整测试集**

运行：

```bat
python -m pytest -q
```

预期：`20 passed`。

### Task 3: 更新文档并做最终验证

**Files:**

- Modify: `README.md`
- Create: `docs/superpowers/specs/2026-08-20-multi-step-tool-agent-design.md`
- Create: `docs/superpowers/plans/2026-08-20-multi-step-tool-agent.md`

**Interfaces:**

- Consumes: 已通过测试的多步骤 `LLMToolAgent`。
- Produces: 与代码一致的 README、设计文档与实施计划。

- [ ] **Step 1: 更新 README**

将流程文字改为：

```text
用户输入 → LLMToolAgent → DeepSeek 选择一个工具 → tools.py 执行 → 工具结果回传 DeepSeek → DeepSeek 决定继续调用工具或输出结果
```

将“当前限制与下一步”第一段改为：

```text
目前项目已有三个文本工具、没有多轮对话记忆，并且每一轮只支持一个工具调用；单次用户请求最多连续执行三次工具调用。
```

将测试示例的 `18 passed` 改为 `20 passed`。

- [ ] **Step 2: 运行最终验证**

运行：

```bat
python -m pytest -q
git diff --check
git status
```

预期：`20 passed`；`git diff --check` 没有代码格式错误；待提交文件为 `llm_agent.py`、`test_llm_agent.py`、`README.md`、设计文档和本计划文档。

- [ ] **Step 3: 由用户保存版本并上传**

```bat
git add llm_agent.py test_llm_agent.py README.md docs/superpowers/specs/2026-08-20-multi-step-tool-agent-design.md docs/superpowers/plans/2026-08-20-multi-step-tool-agent.md
git commit -m "feat: support multi-step tool calls"
git push origin main
```
