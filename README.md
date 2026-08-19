# DeepSeek Tool Calling Agent

这是 Week06 的 Python 学习项目：用 DeepSeek 大模型为一个本地 Agent 选择并调用 Python 工具。

它的核心不是让 Python 猜用户想做什么，而是让大模型把请求转换为规范的工具调用；Python 再负责校验参数、执行工具，并把结果交还给模型组织最终回答。

## 项目功能

- 支持普通问题：没有需要调用工具时，直接返回大模型回答。
- 支持把文本转成大写，例如：`请把 hello agent 转为大写`。
- 支持统计英文单词，例如：`统计单词 I am learning agent development`。
- 只允许调用已注册的工具，并校验工具名称、JSON 参数和 `text` 参数类型。
- 对缺少 API Key、模型调用异常、多个工具调用等情况返回可读错误信息。
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
├── tools.py            # 两个实际执行的本地工具
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

大模型不会直接执行电脑中的 Python；它只能提出“调用哪个工具、传什么参数”。真正执行工具的是这一层代码。

### `llm_agent.py`：Agent 调度层

`LLMToolAgent` 负责整个 Tool Calling 流程：

1. 读取 `.env` 中的 `DEEPSEEK_API_KEY`。
2. 把可用工具的名称、用途和参数格式发送给 DeepSeek。
3. 接收模型的决定：直接回答，或请求调用一个工具。
4. 校验工具调用是否安全、参数是否正确。
5. 执行 `tools.py` 中对应的函数。
6. 将工具结果发回模型，获得适合用户阅读的最终回答。

它还限制一次只能调用一个工具。这是当前学习版本的简化设计，方便先把单次 Tool Calling 的基本流程学清楚。

### `agent.py`：规则式 Agent 对照版本

`LocalToolAgent` 不使用大模型，而是用 `if/elif` 判断请求是否以“`大写 `”或“`统计单词 `”开头。

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

普通问题也可以直接询问：

```text
请输入请求：什么是 Python 函数？
```

此时模型会直接回答；但它没有联网、天气、数据库等工具时，不能可靠地获取实时外部信息或执行外部操作。

## 运行测试

```bat
python -m pytest -q
```

当前应看到：

```text
15 passed
```

## Agent 工作流程

```text
用户自然语言请求
        ↓
LLMToolAgent 把工具说明发送给 DeepSeek
        ↓
DeepSeek 决定：直接回答，或调用某个工具
        ↓
Python 校验工具名称和参数
        ↓
tools.py 执行本地函数
        ↓
工具结果返回给 DeepSeek
        ↓
输出最终回答给用户
```

## 这周的学习重点

1. **工具定义（Schema）**：明确告诉模型有哪些工具、每个工具接收什么参数。
2. **模型负责决策，Python 负责执行**：模型不能越过程序直接操作本地系统。
3. **参数校验与白名单**：不能盲信模型输出，未知工具或错误参数必须拒绝。
4. **两次模型调用**：第一次决定要不要用工具；工具执行后，第二次根据结果生成最终回答。
5. **可测试性**：用 Fake Client 模拟模型响应，避免测试依赖网络和真实 API Key。

## 当前限制与下一步

目前项目只有两个文本工具、没有多轮对话记忆，并且一次只支持一个工具调用。后续可以继续扩展：

- 增加天气、文件读取、网页搜索、数据库查询等工具。
- 支持多个工具的连续调用。
- 加入聊天历史与长期记忆。
- 为高风险工具增加权限确认、日志和更严格的参数规则。
- 使用 RAG，让 Agent 先检索自己的资料再回答。
