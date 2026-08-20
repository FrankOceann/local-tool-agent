# Tool Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为需要确认的模拟保存笔记工具增加“请求、确认或取消、执行”的安全流程。

**Architecture:** `tools.py` 提供模拟工具和工具权限元数据；`LLMToolAgent` 保持一项待确认调用，并在下一次用户输入时优先由 Python 处理确认状态。模型首次请求 `save_note` 时不会执行；用户输入“确认”后，Agent 直接执行模拟工具，不再请求模型。

**Tech Stack:** Python 3.10+、DeepSeek OpenAI 兼容客户端、pytest、python-dotenv。

**Spec:** `docs/superpowers/specs/2026-08-21-tool-confirmation-design.md`

## Global Constraints

- `save_note` 只模拟成功结果，绝不读写、创建、修改或删除真实文件。
- `upper_text`、`count_words`、`summarize_text` 保持低风险且自动执行。
- `save_note` 必须先确认；确认、取消和等待状态由 Python 处理，不发送给模型。
- 单次用户请求最多执行 3 次工具调用；等待确认尚未执行的调用不计入次数。
- 若同一轮包含 `save_note` 和其他工具，拒绝整个批次，不执行其中任何工具。
- 保留工具白名单、JSON 解析和 `text` 字符串校验。
- 用户自行执行所有 Git 提交、合并与推送命令。

---

## 文件结构

- `tools.py`：新增 `save_note` 和工具权限映射，定义工具本身与其风险等级。
- `llm_agent.py`：读取权限映射、保存待确认状态、处理确认/取消、拦截混合工具批次。
- `test_llm_agent.py`：使用 `FakeClient` 验证不调用真实 API 的权限流程。
- `test_main.py`：验证模拟工具的纯函数结果。
- `README.md`：记录新增工具、权限规则、命令行用法和当前测试数量。

### Task 1: 模拟保存工具与权限元数据

**Files:**
- Modify: `tools.py`
- Modify: `test_main.py`

**Interfaces:**
- Produces: `save_note(text: str) -> str`
- Produces: `TOOL_PERMISSIONS: dict[str, str]`，其中 `save_note` 的值为 `"confirmation_required"`，三个既有文本工具的值为 `"auto"`。
- Consumes: 无。

- [ ] **Step 1: 为模拟工具写失败测试**

在 `test_main.py` 导入 `save_note`，加入：

```python
def test_save_note_returns_a_simulated_success_message():
    assert save_note("明天学习 Agent") == "已模拟保存笔记：明天学习 Agent"
```

- [ ] **Step 2: 运行该测试，确认它失败**

Run:

```bat
python -m pytest -q test_main.py::test_save_note_returns_a_simulated_success_message
```

Expected: FAIL，提示无法导入 `save_note`。

- [ ] **Step 3: 实现模拟工具和权限映射**

在 `tools.py` 的既有函数下增加：

```python
TOOL_PERMISSIONS = {
    "upper_text": "auto",
    "count_words": "auto",
    "summarize_text": "auto",
    "save_note": "confirmation_required",
}


def save_note(text: str) -> str:
    return f"已模拟保存笔记：{text}"
```

这里不能使用 `open()`、`Path.write_text()` 或其他文件系统操作。

- [ ] **Step 4: 运行测试，确认工具通过**

Run:

```bat
python -m pytest -q test_main.py::test_save_note_returns_a_simulated_success_message
```

Expected: PASS。

- [ ] **Step 5: 手动保存本任务的 Git 提交**

```bat
git add tools.py test_main.py
git commit -m "feat: add simulated note tool"
```

### Task 2: 单工具确认、取消与待确认状态

**Files:**
- Modify: `llm_agent.py`
- Modify: `test_llm_agent.py`

**Interfaces:**
- Consumes: `save_note` 和 `TOOL_PERMISSIONS`。
- Produces: `LLMToolAgent.pending_tool_call: dict[str, str] | None`。
- Produces: `run("确认")` 和 `run("取消")` 的本地处理行为。

- [ ] **Step 1: 写确认流程的失败测试**

在 `test_llm_agent.py` 加入一个 `save_note` 工具调用测试。它的关键断言必须是：第一次请求只要求确认；第二次输入确认后才执行，且第二次没有新的模型请求。

```python
def test_agent_requires_confirmation_before_simulating_note_save():
    tool_call = SimpleNamespace(
        id="call_save",
        function=SimpleNamespace(
            name="save_note",
            arguments='{"text": "明天学习 Agent"}',
        ),
    )
    client = FakeClient([
        response_with(SimpleNamespace(content=None, tool_calls=[tool_call])),
    ])
    agent = LLMToolAgent(client=client, api_key="test-key")

    assert agent.run("保存一条笔记") == (
        "操作需要确认：将模拟保存笔记“明天学习 Agent”。请输入“确认”或“取消”。"
    )
    assert len(client.calls) == 1
    assert agent.run("确认") == "已模拟保存笔记：明天学习 Agent"
    assert len(client.calls) == 1
    assert agent.pending_tool_call is None
```

- [ ] **Step 2: 运行确认测试，确认它失败**

Run:

```bat
python -m pytest -q test_llm_agent.py::test_agent_requires_confirmation_before_simulating_note_save
```

Expected: FAIL，因为 `save_note` 尚未被注册到 Agent 或 Agent 会直接执行它。

- [ ] **Step 3: 写取消、无待确认与其他输入的失败测试**

加入以下三个测试；它们都使用先返回 `save_note` 调用的 `FakeClient` 来建立待确认状态：

```python
def test_agent_cancels_a_pending_note_save():
    # 建立 pending_tool_call 后执行：
    assert agent.run("取消") == "已取消待确认的操作。"
    assert agent.pending_tool_call is None


def test_agent_reports_when_confirmation_has_nothing_to_confirm():
    agent = LLMToolAgent(client=FakeClient([]), api_key="test-key")
    assert agent.run("确认") == "当前没有待确认的操作。"
    assert agent.run("取消") == "当前没有待确认的操作。"


def test_agent_keeps_pending_operation_when_user_enters_other_text():
    # 建立 pending_tool_call 后执行：
    assert agent.run("换一条笔记") == "当前有待确认的操作，请输入“确认”或“取消”。"
    assert agent.pending_tool_call is not None
    assert len(client.calls) == 1
```

- [ ] **Step 4: 运行状态测试，确认它们失败**

Run:

```bat
python -m pytest -q test_llm_agent.py -k "confirmation or pending_note or cancels"
```

Expected: FAIL，因为 `pending_tool_call` 与确认分支尚未实现。

- [ ] **Step 5: 实现最小确认状态机**

在 `llm_agent.py`：

1. 导入 `save_note`、`TOOL_PERMISSIONS`，将 `save_note` 添加进 `TOOL_SCHEMAS` 和 `tool_registry`。
2. 在 `__init__` 中初始化：

```python
self.pending_tool_call: dict[str, str] | None = None
```

3. 在 `run()` 的 API Key 检查之后、构造 `messages` 之前调用一个私有方法：

```python
pending_result = self._handle_pending_tool_call(request)
if pending_result is not None:
    return pending_result
```

4. 实现 `_handle_pending_tool_call(self, request: str) -> str | None`。无 pending 且输入是 `确认` 或 `取消` 时返回 `"当前没有待确认的操作。"`；有 pending 时只接受 `确认` 或 `取消`；确认时调用 `_run_tool`，成功后清空 pending 并返回工具结果；取消时清空 pending 并返回 `"已取消待确认的操作。"`；其他输入返回 `"当前有待确认的操作，请输入“确认”或“取消”。"`。
5. 在执行工具之前，用一个只解析和校验、不执行工具的私有方法验证 `name`、JSON 和 `text`。确认工具调用有效后，若 `TOOL_PERMISSIONS[name] == "confirmation_required"`，保存：

```python
self.pending_tool_call = {
    "name": tool_call.function.name,
    "arguments_json": tool_call.function.arguments,
    "tool_call_id": tool_call.id,
}
```

然后返回精确的确认提示，不执行工具，也不发起下一次模型调用。

6. 让 `_run_tool()` 复用相同校验方法，避免两处出现不同的 JSON 或类型判断。

- [ ] **Step 6: 运行状态测试与原有 Agent 测试**

Run:

```bat
python -m pytest -q test_llm_agent.py
```

Expected: PASS，且既有工具调用测试仍通过。

- [ ] **Step 7: 手动保存本任务的 Git 提交**

```bat
git add llm_agent.py test_llm_agent.py
git commit -m "feat: require confirmation for sensitive tools"
```

### Task 3: 同轮混合工具批次保护与项目文档

**Files:**
- Modify: `llm_agent.py`
- Modify: `test_llm_agent.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `TOOL_PERMISSIONS` 和 `pending_tool_call`。
- Produces: 对“`save_note` 与其他工具同批”的统一拒绝消息：`"同一轮请求包含需要确认的工具，当前不支持与其他工具混合执行。"`。

- [ ] **Step 1: 写混合批次的失败测试**

构造一个模型响应，其中第一个调用为 `upper_text`，第二个调用为 `save_note`。替换 `upper_text` 和 `save_note` 的注册函数为会抛出 `AssertionError` 的 lambda，证明一个也不会执行：

```python
def test_agent_rejects_mixed_confirmation_and_auto_tools_in_one_batch():
    # client 的唯一响应含 [upper_call, save_call]
    agent = LLMToolAgent(client=client, api_key="test-key")
    agent.tool_registry["upper_text"] = lambda _text: (_ for _ in ()).throw(
        AssertionError("混合批次不应执行自动工具")
    )
    agent.tool_registry["save_note"] = lambda _text: (_ for _ in ()).throw(
        AssertionError("混合批次不应执行敏感工具")
    )

    assert agent.run("大写并保存") == (
        "同一轮请求包含需要确认的工具，当前不支持与其他工具混合执行。"
    )
    assert len(client.calls) == 1
    assert agent.pending_tool_call is None
```

- [ ] **Step 2: 运行混合批次测试，确认它失败**

Run:

```bat
python -m pytest -q test_llm_agent.py::test_agent_rejects_mixed_confirmation_and_auto_tools_in_one_batch
```

Expected: FAIL，因为当前循环会尝试执行批次里的第一个工具。

- [ ] **Step 3: 在执行前实现整批预检**

在已有工具调用数量上限预检之后、`messages.append(assistant_message)` 之前增加：

```python
requires_confirmation = [
    tool_call
    for tool_call in assistant_message.tool_calls
    if TOOL_PERMISSIONS.get(tool_call.function.name) == "confirmation_required"
]
if requires_confirmation and len(assistant_message.tool_calls) > 1:
    return "同一轮请求包含需要确认的工具，当前不支持与其他工具混合执行。"
```

这段代码必须在任何 `_run_tool()` 之前运行，保证整批零执行。

- [ ] **Step 4: 运行混合批次测试与全量测试**

Run:

```bat
python -m pytest -q test_llm_agent.py::test_agent_rejects_mixed_confirmation_and_auto_tools_in_one_batch
python -m pytest -q
```

Expected: 两条命令均 PASS；全量预计为 `30 passed`。

- [ ] **Step 5: 更新 README**

在“项目功能”“工具层”“Agent 调度层”“运行项目”“当前限制与下一步”中加入：

- `save_note` 是模拟保存工具，不写真实文件；
- 三个文本工具自动执行，`save_note` 必须输入 `确认` 或 `取消`；
- 待确认时其他输入不会调用模型或执行工具；
- 同轮混合敏感工具和自动工具会被整体拒绝；
- 当前测试期望改为 `30 passed`。

添加一段可复制的命令行示例：

```text
请输入请求：保存笔记：明天学习 Agent
操作需要确认：将模拟保存笔记“明天学习 Agent”。请输入“确认”或“取消”。

请输入请求：确认
已模拟保存笔记：明天学习 Agent
```

- [ ] **Step 6: 最终质量检查**

Run:

```bat
python -m pytest -q
git diff --check
git status
```

Expected: `30 passed`；`git diff --check` 没有检查错误（Windows 的 LF/CRLF 提示可忽略）；`git status` 只显示本任务的四个文件。

- [ ] **Step 7: 手动保存本任务的 Git 提交**

```bat
git add README.md llm_agent.py test_llm_agent.py
git commit -m "feat: add tool confirmation safeguards"
```
