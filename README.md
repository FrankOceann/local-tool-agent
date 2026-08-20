# DeepSeek Tool Calling Agent

这是 Week06 的 Python 学习项目：用 DeepSeek 大模型为一个本地 Agent 选择并调用 Python 工具。

它的核心不是让 Python 猜用户想做什么，而是让大模型把请求转换为规范的工具调用；Python 再负责校验参数、执行工具，并把结果交还给模型组织最终回答。

## 项目功能

- 支持普通问题：没有需要调用工具时，直接返回大模型回答。
- 支持把文本转成大写，例如：`请把 hello agent 转为大写`。
- 支持统计英文单词，例如：`统计单词 I am learning agent development`。
- 支持文本摘要，例如：`请总结这段文字：第一句。第二句。第三句。`；工具会保留前两句。
- 支持模拟保存笔记：`save_note` 不会写入真实文件，必须由用户输入“确认”后才能执行。
- 支持连续或同一轮的多个工具调用：一次用户请求最多执行三次工具；多个工具按模型给出的顺序执行。
- 工具按风险分级：三个文本工具自动执行；`save_note` 需要人工确认。混合了自动工具与需确认工具的同轮批次会被整体拒绝。
- 只允许调用已注册的工具，并校验工具名称、JSON 参数和 `text` 参数类型。
- 对缺少 API Key、模型调用异常、工具超过上限和错误参数等情况返回可读错误信息。
- 提供离线测试，不需要真实 API Key 也能验证主要逻辑。

## 技术栈

- Python 3.10+
- DeepSeek API（OpenAI 兼容调用格式）
- `openai`：调用 DeepSeek 模型
- `python-dotenv`：从 `.env` 读取 API Key
- `pytest`：自动化测试

## 项目结构

```text
week06-llm-tool-calling/
├── .env.example        # API Key 配置示例，不含真实密钥
├── .gitignore          # 忽略 .env、缓存和隔离工作区
├── tools.py            # 四个本地工具与权限映射
├── agent.py            # 第一阶段：规则式 LocalToolAgent
├── llm_agent.py        # 第二阶段：由 DeepSeek 决定是否调用工具
├── main.py             # 命令行入口
├── test_main.py        # 第一阶段的规则式 Agent 测试
├── test_llm_agent.py   # LLM Tool Calling 的离线测试
├── requirements.txt    # 项目依赖
└── README.md           # 项目说明
```

## 核心模块说明

### `tools.py`：工具层

这里是真正做事的 Python 函数：

- `upper_text(text)`：将文本转为大写。
- `count_words(text)`：统计英文单词数量。
- `summarize_text(text)`：保留文本前两句作为简短摘要。
- `save_note(text)`：模拟保存笔记；只返回成功文字，不创建或修改任何文件。

`TOOL_PERMISSIONS` 是工具权限表：前三个文本工具是 `auto`（自动执行），`save_note` 是 `confirmation_required`（需要确认）。

大模型不会直接执行电脑中的 Python；它只能提出“调用哪个工具、传什么参数”。真正执行工具的是这一层代码。

### `llm_agent.py`：Agent 调度层

`LLMToolAgent` 负责整个 Tool Calling 流程：

1. 读取 `.env` 中的 `DEEPSEEK_API_KEY`。
2. 把可用工具的名称、用途和参数格式发送给 DeepSeek。
3. 接收模型的决定：直接回答，或请求调用一个或多个工具。
4. 校验工具调用是否安全、参数是否正确，并按权限等级决定是否需要人工确认。
5. 执行 `tools.py` 中对应的函数，或保存一项待确认操作。
6. 用户输入“确认”时由 Python 直接执行待确认的模拟工具；输入“取消”时清空该操作。
7. 将自动工具的结果发回模型，获得适合用户阅读的最终回答。

模型每一轮可以请求一个或多个工具；Python 始终按模型给出的顺序执行。单次用户请求最多执行三次工具调用；达到上限后，Agent 会要求模型直接生成最终回答，避免无限循环和持续消耗 API 额度。若同一轮混合了 `save_note` 和其他工具，Python 会在执行前整体拒绝该批次，避免部分执行、部分等待确认。

### `agent.py`：规则式 Agent 对照版本

`LocalToolAgent` 不使用大模型，而是用 `if/elif` 判断请求是否以“`大写 `”或“`统计单词 `”开头。它没有接入摘要工具，这正好体现了规则式 Agent 每增加一种能力都需要手动修改判断逻辑。

它保留在项目中，是为了对比两种方案：

- 规则式 Agent：简单、稳定，但每增加一种表达方式都要手写规则。
- LLM Agent：更能理解自然语言，可以在多个工具之间选择，但需要 API、提示词和安全校验。

### `main.py`：命令行入口

运行 `python main.py` 后，它会读取你的输入，交给 `LLMToolAgent`，最后把回答打印到终端。

### 测试文件：`test_main.py` 与 `test_llm_agent.py`

测试保证修改代码后，原有能力不会被意外破坏。

- `test_main.py`：验证基础工具和规则式 Agent。
- `test_llm_agent.py`：使用假的模型客户端，测试直接回答、工具调用、异常处理和参数校验；运行时不消耗 API 额度。

## 配置与安装

建议先进入项目目录：

```bat
cd /d "D:\桌面\所有codex项目\AI agent 开发\python 学习\week06-llm-tool-calling"
```

安装依赖：

```bat
python -m pip install -r requirements.txt
```

在项目根目录创建 `.env` 文件，并填入自己的 Key：

```text
DEEPSEEK_API_KEY=你的_DeepSeek_API_Key
```

`.env` 已被 `.gitignore` 忽略。真实密钥绝不能写进代码、README、截图或 Git 提交记录。

## 运行项目

```bat
python main.py
```

示例：

```text
请输入请求：请把 hello agent 转为大写
HELLO AGENT
```

```text
请输入请求：统计单词 I am learning agent development
英文单词数量：5
```

也可以尝试让 Agent 调用摘要工具：

```text
请输入请求：请总结这段文字：第一句。第二句。第三句。
```

本地工具会产生 `第一句。第二句。`，模型再将这个结果组织成最终回答。

普通问题也可以直接询问：

```text
请输入请求：什么是 Python 函数？
```

此时模型会直接回答；但它没有联网、天气、数据库等工具时，不能可靠地获取实时外部信息或执行外部操作。

### 手动演示确认流程

目前 `main.py` 只处理一条输入后就结束；确认流程需要保留同一个 Agent 对象，因此可用 Python 交互模式演示：

```bat
python
```

```python
from llm_agent import LLMToolAgent
agent = LLMToolAgent()
agent.run("请把明天学习 Agent 保存为笔记")
agent.run("确认")
```

第一次应提示输入“确认”或“取消”；第二次返回模拟保存结果。该过程不会创建真实笔记文件。

## 运行测试

```bat
python -m pytest -q
```

当前应看到：

```text
29 passed
```

## Agent 工作流程

```text
用户自然语言请求
        ↓
LLMToolAgent 把工具说明发送给 DeepSeek
        ↓
DeepSeek 决定：直接回答，或调用一个或多个工具
        ↓
Python 校验工具名称、参数与权限
        ↓
低风险工具直接执行；敏感工具等待用户确认
        ↓
自动工具结果返回给 DeepSeek；确认工具由 Python 返回模拟结果
        ↓
DeepSeek 决定继续调用工具或输出最终回答
```

## 这周的学习重点

1. **工具定义（Schema）**：明确告诉模型有哪些工具、每个工具接收什么参数。
2. **模型负责决策，Python 负责执行**：模型不能越过程序直接操作本地系统。
3. **参数校验与白名单**：不能盲信模型输出，未知工具或错误参数必须拒绝。
4. **Agent 循环**：每轮可包含一个或多个工具；每个结果必须携带对应的 `tool_call_id`，模型才能继续决定下一项工具；单次请求最多执行三次。
5. **可测试性**：用 Fake Client 模拟模型响应，避免测试依赖网络和真实 API Key。
6. **最小权限与人工确认**：模型可以提出调用建议，但是否执行敏感操作由 Python 和用户共同决定；确认、取消和待确认状态不能交给模型猜测。

## 当前限制与下一步

目前项目已有三个自动文本工具和一个模拟敏感工具、没有多轮对话记忆；同一轮多个工具会按顺序执行，不使用真正并行，单次请求最多执行三次。`main.py` 仍是单次输入入口，尚未做成可持续等待确认的命令行循环。后续可以继续扩展：

- 增加天气、文件读取、网页搜索、数据库查询等工具。
- 对互不依赖的工具实现真正并行执行。
- 加入聊天历史与长期记忆。
- 将模拟保存替换为受限目录内的真实写入，并增加审计日志和更严格的参数规则。
- 使用 RAG，让 Agent 先检索自己的资料再回答。
