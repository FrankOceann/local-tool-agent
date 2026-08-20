# 同一轮多工具调用 Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `LLMToolAgent` 支持模型在同一轮请求多个本地工具，并在单次请求最多三次工具调用的限制内安全执行。

**Architecture:** `run()` 维护同一份 `messages` 历史和 `executed_tool_calls` 计数。每轮先检查整批调用是否超出剩余额度；未超额时按模型给出的顺序执行，并逐条追加工具结果。

**Tech Stack:** Python 3.10、OpenAI Python SDK（DeepSeek 兼容接口）、pytest、python-dotenv。

**Spec:** `docs/superpowers/specs/2026-08-21-same-round-multi-tool-calls-design.md`

## Global Constraints

- 只使用 `upper_text`、`count_words`、`summarize_text`。
- 单次用户请求最多执行三次工具调用，跨多轮累计。
- 同一轮多个工具按模型给出的顺序执行，不使用线程或并行执行。
- 一轮调用超过剩余额度时拒绝整批，不执行其中任何工具。
- 任一工具校验或执行失败时立即停止，不执行后续工具，也不再次请求模型。
- 保持已有 API 异常、未知工具、无效 JSON 和 `text` 类型错误的中文信息。
- 不增加联网、文件访问、记忆、新工具或真正并发执行。

---

### Task 1: 用失败测试定义同轮多个工具

**Files:**

- Modify: `test_llm_agent.py`
- Test: `test_llm_agent.py`

**Interfaces:**

- Consumes: `LLMToolAgent.run(request: str) -> str`、`FakeClient`、`response_with()`。
- Produces: 同轮多工具、跨轮额度、批内失败与超额的测试。

- [ ] **Step 1: 写同一轮两个合法工具的失败测试**

在 `test_llm_agent.py` 末尾添加：

```python
def test_agent_executes_two_tool_calls_from_one_model_response():
    upper_call = SimpleNamespace(
        id="call_upper",
        function=SimpleNamespace(name="upper_text", arguments='{"text": "hello"}'),
    )
    count_call = SimpleNamespace(
        id="call_count",
        function=SimpleNamespace(name="count_words", arguments='{"text": "I am learning"}'),
    )
    client = FakeClient([
        response_with(SimpleNamespace(content=None, tool_calls=[upper_call, count_call])),
        response_with(SimpleNamespace(content="大写结果是 HELLO，英文单词数量是 3。", tool_calls=None)),
    ])

    result = LLMToolAgent(client=client, api_key="test-key").run("同时处理两段文本")

    assert result == "大写结果是 HELLO，英文单词数量是 3。"
    assert len(client.calls) == 2
    assert client.calls[1]["messages"][-2:] == [
        {"role": "tool", "tool_call_id": "call_upper", "content": "HELLO"},
        {"role": "tool", "tool_call_id": "call_count", "content": "3"},
    ]
```

- [ ] **Step 2: 运行测试，确认当前实现拒绝多个工具**

运行：

```bat
python -m pytest -q test_llm_agent.py::test_agent_executes_two_tool_calls_from_one_model_response
```

预期：失败，返回当前的“模型一次请求了多个工具，当前仅支持一个工具调用。”

- [ ] **Step 3: 写跨轮累计三次上限的失败测试**

继续添加：

```python
def test_agent_allows_two_same_round_calls_then_one_more_call():
    upper_call = SimpleNamespace(id="call_upper", function=SimpleNamespace(name="upper_text", arguments='{"text": "one"}'))
    count_call = SimpleNamespace(id="call_count", function=SimpleNamespace(name="count_words", arguments='{"text": "one two"}'))
    summary_call = SimpleNamespace(id="call_summary", function=SimpleNamespace(name="summarize_text", arguments='{"text": "甲。乙。丙。"}'))
    client = FakeClient([
        response_with(SimpleNamespace(content=None, tool_calls=[upper_call, count_call])),
        response_with(SimpleNamespace(content=None, tool_calls=[summary_call])),
        response_with(SimpleNamespace(content="已完成三个工具调用。", tool_calls=None)),
    ])

    result = LLMToolAgent(client=client, api_key="test-key").run("连续处理")

    assert result == "已完成三个工具调用。"
    assert len(client.calls) == 3
    assert client.calls[1]["messages"][-2:] == [
        {"role": "tool", "tool_call_id": "call_upper", "content": "ONE"},
        {"role": "tool", "tool_call_id": "call_count", "content": "2"},
    ]
    assert "tools" not in client.calls[2]
    assert "tool_choice" not in client.calls[2]
```

- [ ] **Step 4: 写批内失败与批量超额的失败测试**

继续添加：

```python
def test_agent_rejects_a_batch_that_exceeds_remaining_tool_call_limit():
    call_1 = SimpleNamespace(id="call_1", function=SimpleNamespace(name="upper_text", arguments='{"text": "one"}'))
    call_2 = SimpleNamespace(id="call_2", function=SimpleNamespace(name="upper_text", arguments='{"text": "two"}'))
    call_3 = SimpleNamespace(id="call_3", function=SimpleNamespace(name="upper_text", arguments='{"text": "three"}'))
    client = FakeClient([
        response_with(SimpleNamespace(content=None, tool_calls=[call_1, call_2])),
        response_with(SimpleNamespace(content=None, tool_calls=[call_3, call_1])),
    ])
    agent = LLMToolAgent(client=client, api_key="test-key")
    executed_texts = []
    agent.tool_registry["upper_text"] = lambda text: executed_texts.append(text) or text.upper()

    assert agent.run("连续调用") == "本次请求最多执行 3 次工具调用。"
    assert len(client.calls) == 2
    assert executed_texts == ["one", "two"]


def test_agent_stops_when_a_later_tool_in_one_batch_is_invalid():
    good_call = SimpleNamespace(id="call_good", function=SimpleNamespace(name="upper_text", arguments='{"text": "hello"}'))
    bad_call = SimpleNamespace(id="call_bad", function=SimpleNamespace(name="upper_text", arguments="not-json"))
    skipped_call = SimpleNamespace(id="call_skipped", function=SimpleNamespace(name="count_words", arguments='{"text": "one two"}'))
    client = FakeClient([response_with(SimpleNamespace(content=None, tool_calls=[good_call, bad_call, skipped_call]))])
    agent = LLMToolAgent(client=client, api_key="test-key")
    agent.tool_registry["count_words"] = lambda _text: (_ for _ in ()).throw(AssertionError("失败后不应执行后续工具"))

    assert agent.run("批内包含错误参数") == "模型返回的工具参数不是有效 JSON。"
    assert len(client.calls) == 1
```

- [ ] **Step 5: 运行测试，确认新测试先失败**

运行：

```bat
python -m pytest -q test_llm_agent.py
```

预期：新同轮多工具和额度测试失败；原有测试保持通过。

- [ ] **Step 6: 保存测试阶段版本**

```bat
git add test_llm_agent.py
git commit -m "test: define same-round multi-tool behavior"
```

### Task 2: 实现顺序执行与总上限

**Files:**

- Modify: `llm_agent.py`
- Test: `test_llm_agent.py`

**Interfaces:**

- Consumes: `MAX_TOOL_CALLS: int` 与 `_run_tool(name, arguments_json)`。
- Produces: `run(request: str) -> str` 支持同轮多工具且最多执行三次。

- [ ] **Step 1: 添加超额错误常量**

在 `MULTIPLE_TOOL_CALLS_MESSAGE` 下添加：

```python
TOOL_CALL_LIMIT_MESSAGE = "本次请求最多执行 3 次工具调用。"
```

- [ ] **Step 2: 初始化已执行次数并修改循环条件**

在 `messages` 初始化后添加：

```python
        executed_tool_calls = 0
```

将：

```python
        for _ in range(MAX_TOOL_CALLS):
```

替换为：

```python
        while executed_tool_calls < MAX_TOOL_CALLS:
```

- [ ] **Step 3: 执行整批前检查剩余额度**

用下面代码替换原有的 `len(assistant_message.tool_calls) != 1` 拒绝判断：

```python
            if len(assistant_message.tool_calls) > MAX_TOOL_CALLS - executed_tool_calls:
                return TOOL_CALL_LIMIT_MESSAGE
```

- [ ] **Step 4: 遍历本轮的每一个工具调用**

用下面代码替换从 `tool_call = assistant_message.tool_calls[0]` 开始的单工具执行与单条 tool 消息追加代码：

```python
            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                result, error = self._run_tool(
                    tool_call.function.name,
                    tool_call.function.arguments,
                )
                if error:
                    return error

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )
                executed_tool_calls += 1
```

- [ ] **Step 5: 验证 LLM 测试与完整测试集**

运行：

```bat
python -m pytest -q test_llm_agent.py
python -m pytest -q
git diff --check
```

预期：测试全部通过，`git diff --check` 无输出。

- [ ] **Step 6: 保存实现阶段版本**

```bat
git add llm_agent.py test_llm_agent.py
git commit -m "feat: support same-round multi-tool calls"
```

### Task 3: 更新 README 与最终验证

**Files:**

- Modify: `README.md`
- Test: `test_llm_agent.py`

**Interfaces:**

- Consumes: 已支持同轮多工具的 `LLMToolAgent`。
- Produces: 与实际行为一致的学习说明。

- [ ] **Step 1: 更新功能列表**

将 README 的“支持有先后依赖的请求”条目替换为：

```markdown
- 支持连续或同一轮的多个工具调用：一次用户请求最多执行三次工具；多个工具按模型给出的顺序执行。
```

将“多个工具调用等情况返回可读错误信息”替换为：

```markdown
工具超过上限和错误参数等情况返回可读错误信息。
```

- [ ] **Step 2: 更新调度层、流程与限制说明**

将“每一轮只允许调用一个工具”改为“模型每一轮可以请求一个或多个工具；Python 按模型给出的顺序执行”。在学习重点补充：每个工具结果都要保留对应的 `tool_call_id`。在“当前限制与下一步”将同轮多工具从下一步移除，并明确“不使用真正并行”。

- [ ] **Step 3: 最终验证**

运行：

```bat
python -m pytest -q
git diff --check
git status
```

预期：完整测试通过；格式检查无输出；状态只显示本阶段修改。

- [ ] **Step 4: 保存文档并上传 GitHub**

```bat
git add README.md docs/superpowers/specs/2026-08-21-same-round-multi-tool-calls-design.md docs/superpowers/plans/2026-08-21-same-round-multi-tool-calls.md
git commit -m "docs: explain same-round multi-tool calls"
git push origin main
```
