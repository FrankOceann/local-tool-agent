# Local Tool Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a command-line agent that selects and runs one of two local text tools.

**Architecture:** `main.py` receives one request and prints a result. `agent.py` selects a tool and calls it from `tools.py`. `test_main.py` checks the tool functions, selection logic, and command-line helper without an LLM or API key.

**Tech Stack:** Python 3.10, pytest, standard library only.

## Global Constraints

- Stage one must not use an API key, an LLM SDK, or network requests.
- Supported input formats: `大写 <文字>` and `统计单词 <英文文字>`.
- Unsupported input returns a clear message instead of selecting a tool.
- Run `python -m pytest -q` before every commit.

---

### Task 1: Create and Test Local Tools

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `tools.py`
- Create: `test_main.py`

**Interfaces:**
- Produces: `upper_text(text: str) -> str`
- Produces: `count_words(text: str) -> int`

- [ ] **Step 1: Create `.gitignore`**

```text
.venv/
__pycache__/
.pytest_cache/
.env
```

- [ ] **Step 2: Create `requirements.txt` and install pytest**

Write this one line in `requirements.txt`:

```text
pytest
```

Then run:

```bat
python -m pip install -r requirements.txt
```

- [ ] **Step 3: Write the failing tests in `test_main.py`**

```python
from tools import count_words, upper_text


def test_upper_text():
    assert upper_text("hello") == "HELLO"


def test_count_words():
    assert count_words("I am learning agent development") == 5
```

- [ ] **Step 4: Run the tests to verify failure**

```bat
python -m pytest test_main.py -v
```

Expected: collection fails because `tools.py` does not exist.

- [ ] **Step 5: Create `tools.py`**

```python
def upper_text(text: str) -> str:
    return text.upper()


def count_words(text: str) -> int:
    return len(text.split())
```

- [ ] **Step 6: Verify and commit**

```bat
python -m pytest test_main.py -v
git add .gitignore requirements.txt tools.py test_main.py docs/superpowers/plans
git commit -m "feat: add local text tools"
```

Expected: `2 passed` before committing.

### Task 2: Add Rule-Based Tool Selection

**Files:**
- Create: `agent.py`
- Modify: `test_main.py`

**Interfaces:**
- Consumes: `upper_text(text: str) -> str`, `count_words(text: str) -> int`
- Produces: `LocalToolAgent.run(request: str) -> str`

- [ ] **Step 1: Append failing tests to `test_main.py`**

```python
from agent import LocalToolAgent


def test_agent_calls_upper_text_tool():
    agent = LocalToolAgent()
    assert agent.run("大写 hello world") == "HELLO WORLD"


def test_agent_calls_count_words_tool():
    agent = LocalToolAgent()
    assert agent.run("统计单词 I am learning agent development") == "英文单词数量: 5"


def test_agent_rejects_unsupported_request():
    agent = LocalToolAgent()
    assert agent.run("今天北京天气如何") == "暂不支持这个请求。可使用：大写 <文字> 或 统计单词 <英文文字>"
```

- [ ] **Step 2: Run the tests to verify failure**

```bat
python -m pytest test_main.py -v
```

Expected: collection fails because `agent.py` does not exist.

- [ ] **Step 3: Create `agent.py`**

```python
from tools import count_words, upper_text


class LocalToolAgent:
    def run(self, request: str) -> str:
        if request.startswith("大写 "):
            text = request.removeprefix("大写 ").strip()
            return upper_text(text)

        if request.startswith("统计单词 "):
            text = request.removeprefix("统计单词 ").strip()
            return f"英文单词数量: {count_words(text)}"

        return "暂不支持这个请求。可使用：大写 <文字> 或 统计单词 <英文文字>"
```

- [ ] **Step 4: Verify and commit**

```bat
python -m pytest -q
git add agent.py test_main.py
git commit -m "feat: add rule based tool selection"
```

Expected: `5 passed` before committing.

### Task 3: Add the Command-Line Entry Point

**Files:**
- Create: `main.py`
- Modify: `test_main.py`

**Interfaces:**
- Consumes: `LocalToolAgent.run(request: str) -> str`
- Produces: `run_once(request: str) -> str`

- [ ] **Step 1: Append this failing test to `test_main.py`**

```python
from main import run_once


def test_run_once_returns_agent_result():
    assert run_once("大写 codex") == "CODEX"
```

- [ ] **Step 2: Verify the test fails**

```bat
python -m pytest test_main.py::test_run_once_returns_agent_result -v
```

Expected: collection fails because `main.py` does not exist.

- [ ] **Step 3: Create `main.py`**

```python
from agent import LocalToolAgent


agent = LocalToolAgent()


def run_once(request: str) -> str:
    return agent.run(request)


if __name__ == "__main__":
    request = input("请输入请求: ")
    print(run_once(request))
```

- [ ] **Step 4: Verify manually and commit**

```bat
python -m pytest -q
python main.py
```

At the prompt, enter `大写 hello agent`. Expected output: `HELLO AGENT`.

```bat
git add main.py test_main.py
git commit -m "feat: add local tool agent cli"
```

### Task 4: Document the Prototype

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create `README.md` with this content**

````markdown
# Local Tool Agent

一个用于学习 Agent Tool Calling 基础流程的 Python 命令行项目。

## 支持的请求

```text
大写 hello agent
统计单词 I am learning agent development
```

## 运行与测试

```bat
python main.py
python -m pytest -q
```

## 流程

用户输入 -> LocalToolAgent 选择工具 -> tools.py 执行 -> 输出结果

当前由规则选择工具；下一阶段会把规则替换为真实 LLM 的 Tool Calling。
````

````

- [ ] **Step 2: Verify and commit**

```bat
python -m pytest -q
git add README.md
git commit -m "docs: add local tool agent README"
```

Expected: all tests pass before committing.
