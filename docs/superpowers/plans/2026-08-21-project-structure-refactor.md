# Week06 项目化整理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Tool Agent 整理为 `app/`、`tools/` 和 `tests/`，保持全部 30 个测试和现有功能不变。

**Architecture:** `app/` 管理 Agent、配置和 Schema；`tools/` 管理工具函数与权限；`tests/` 保存测试。根目录保留 `main.py` 与既有的 `agent.py`。

**Tech Stack:** Python 3.10、pytest、python-dotenv、OpenAI Python SDK、DeepSeek API。

**Spec:** `docs/superpowers/specs/2026-08-21-project-structure-refactor-design.md`

## Global Constraints

- 命令保持为 `python main.py` 和 `python -m pytest -q`。
- `.env` 仍在根目录，不提交真实 `DEEPSEEK_API_KEY`。
- 不改变四个工具、权限、确认/取消、三次调用上限或中文错误信息。
- 不实现真实写入、外部 API、记忆、并行或确认后恢复对话。

---

## Final Structure

```text
week06-llm-tool-calling/
├─ main.py
├─ agent.py
├─ app/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ llm_agent.py
│  └─ tool_schemas.py
├─ tools/
│  ├─ __init__.py
│  ├─ text_tools.py
│  ├─ note_tools.py
│  └─ registry.py
├─ tests/
│  ├─ test_main.py
│  └─ test_llm_agent.py
└─ README.md
```

## Task 1: 整理工具函数与注册表

**Files:**
- Create: `tools/__init__.py`, `tools/text_tools.py`, `tools/note_tools.py`, `tools/registry.py`
- Modify: `agent.py`, `test_main.py`
- Move: `test_main.py` → `tests/test_main.py`
- Delete: `tools.py`

**Interfaces:**
- Produces: `tools.text_tools.upper_text(text: str) -> str`、`count_words(text: str) -> int`、`summarize_text(text: str) -> str`。
- Produces: `tools.note_tools.save_note(text: str) -> str`。
- Produces: `tools.registry.TOOL_REGISTRY` 与 `tools.registry.TOOL_PERMISSIONS`。

- [ ] **Step 1: 写入文本工具文件**

创建空文件 `tools/__init__.py`。创建 `tools/text_tools.py`：

```python
import re


def upper_text(text: str) -> str:
    return text.upper()


def count_words(text: str) -> int:
    return len(text.split())


def summarize_text(text: str) -> str:
    sentences = re.findall(r"[^。！？.!?]+[。！？.!?]?", text)
    return "".join(sentences[:2]) or text
```

- [ ] **Step 2: 写入模拟笔记工具与注册表**

创建 `tools/note_tools.py`：

```python
def save_note(text: str) -> str:
    return f"已模拟保存笔记：{text}"
```

创建 `tools/registry.py`：

```python
from collections.abc import Callable

from tools.note_tools import save_note
from tools.text_tools import count_words, summarize_text, upper_text


TOOL_REGISTRY: dict[str, Callable[[str], str | int]] = {
    "upper_text": upper_text,
    "count_words": count_words,
    "summarize_text": summarize_text,
    "save_note": save_note,
}

TOOL_PERMISSIONS = {
    "upper_text": "auto",
    "count_words": "auto",
    "summarize_text": "auto",
    "save_note": "confirmation_required",
}
```

- [ ] **Step 3: 修正现有导入并运行测试**

将 `agent.py` 的导入替换为：

```python
from tools.text_tools import count_words, upper_text
```

将根目录 `test_main.py` 顶部替换为：

```python
from agent import LocalToolAgent
from tools.note_tools import save_note
from tools.text_tools import count_words, summarize_text, upper_text
```

删除摘要测试中的 `getattr(tools, ...)` 和 `assert summarize_text is not None`，保留直接调用 `summarize_text(...)` 的断言。

Run:

```bat
python -m pytest -q test_main.py
```

Expected: PASS。

- [ ] **Step 4: 移动测试并删除旧模块**

Run:

```bat
mkdir tests
git mv test_main.py tests/test_main.py
git rm tools.py
python -m pytest -q tests/test_main.py
```

Expected: PASS。

- [ ] **Step 5: 提交工具层**

```bat
git add agent.py tools tests/test_main.py
git commit -m "refactor: organize tools into package"
```

## Task 2: 拆分 Agent 配置、Schema 和运行逻辑

**Files:**
- Create: `app/__init__.py`, `app/config.py`, `app/tool_schemas.py`, `app/llm_agent.py`
- Modify: `main.py`, `test_llm_agent.py`
- Move: `test_llm_agent.py` → `tests/test_llm_agent.py`
- Delete: `llm_agent.py`

**Interfaces:**
- Produces: `app.llm_agent.LLMToolAgent`。
- Produces: `app.config` 内的模型配置与错误消息。
- Produces: `app.tool_schemas.TOOL_SCHEMAS`。
- Consumes: `tools.registry.TOOL_REGISTRY` 和 `TOOL_PERMISSIONS`。

- [ ] **Step 1: 新建配置模块**

创建空文件 `app/__init__.py`。创建 `app/config.py`，把当前 `llm_agent.py` 中的 `MISSING_KEY_MESSAGE`、`API_CALL_ERROR_MESSAGE`、`TOOL_CALL_LIMIT_MESSAGE`、`MODEL_NAME`、`BASE_URL`、`MAX_TOOL_CALLS` 和完整 `SYSTEM_PROMPT` 原样移动进去，不修改任何字符串。

- [ ] **Step 2: 新建 Schema 模块**

创建 `app/tool_schemas.py`，把当前 `llm_agent.py` 完整的 `TOOL_SCHEMAS = [...]` 原样移动进去。该文件不得导入或执行工具函数。

- [ ] **Step 3: 迁移 Agent 类并替换依赖**

把完整 `LLMToolAgent` 类移动到 `app/llm_agent.py`。新文件顶部使用：

```python
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from app.config import (
    API_CALL_ERROR_MESSAGE,
    BASE_URL,
    MAX_TOOL_CALLS,
    MISSING_KEY_MESSAGE,
    MODEL_NAME,
    SYSTEM_PROMPT,
    TOOL_CALL_LIMIT_MESSAGE,
)
from app.tool_schemas import TOOL_SCHEMAS
from tools.registry import TOOL_PERMISSIONS, TOOL_REGISTRY
```

将原本的手写 `self.tool_registry = {...}` 替换为：

```python
self.tool_registry = TOOL_REGISTRY.copy()
```

不修改 `run`、`_handle_pending_tool_call`、`_parse_tool_arguments`、`_run_tool` 的行为。

- [ ] **Step 4: 迁移导入和 Agent 测试**

把 `test_llm_agent.py` 顶部导入替换为：

```python
from app.config import MISSING_KEY_MESSAGE
from app.llm_agent import LLMToolAgent
from app.tool_schemas import TOOL_SCHEMAS
```

把 `main.py` 首行替换为：

```python
from app.llm_agent import LLMToolAgent
```

Run:

```bat
git mv test_llm_agent.py tests/test_llm_agent.py
git rm llm_agent.py
python -m pytest -q tests/test_llm_agent.py
```

Expected: PASS。

- [ ] **Step 5: 运行全量测试并提交 Agent 层**

Run:

```bat
python -m pytest -q
```

Expected: `30 passed`。

Then run:

```bat
git add app main.py tests/test_llm_agent.py
git commit -m "refactor: separate agent configuration and schemas"
```

## Task 3: 更新 README 并进行最终验证

**Files:**
- Modify: `README.md`
- Test: `tests/test_main.py`, `tests/test_llm_agent.py`

**Interfaces:**
- Consumes: Tasks 1 and 2 的最终目录。
- Produces: 与实际结构一致的项目说明。

- [ ] **Step 1: 更新 README**

替换 README 的目录图为本计划的 `Final Structure`。加入以下精确职责说明：

```text
text_tools.py：三个自动文本工具。
note_tools.py：需要确认的模拟保存工具。
registry.py：唯一的工具函数注册表与权限表。
config.py：模型配置、系统提示和错误信息。
tool_schemas.py：发送给模型的工具 Schema。
llm_agent.py：模型调用、参数校验、权限确认和工具调度。
```

明确根目录 `agent.py` 是早期规则型 Agent 练习。

- [ ] **Step 2: 执行完整验证**

Run:

```bat
python -c "from main import run_once; print('入口导入成功')"
python -m pytest -q
git diff --check
git status
```

Expected: 输出 `入口导入成功`、`30 passed`；`git diff --check` 无输出；状态只显示 `README.md`。

- [ ] **Step 3: 提交 README**

```bat
git add README.md
git commit -m "docs: explain project module structure"
```

## Final Manual Check

Run:

```bat
python main.py
```

Enter: `把 hello agent 转为大写`。

Expected: 程序正常进入 DeepSeek 调用流程；若 `.env` 不存在，应只返回现有 API Key 中文提示，不得出现 Python 导入错误。
