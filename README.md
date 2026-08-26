# DeepSeek Tool Calling Agent

这是 Week06 的 Python 学习项目：用 DeepSeek 大模型为一个本地 Agent 选择并调用 Python 工具。

它的核心不是让 Python 猜用户想做什么，而是让大模型把请求转换为规范的工具调用；Python 再负责校验参数、执行工具，并把结果交还给模型组织最终回答。

## 项目功能

- 支持普通问题：没有需要调用工具时，直接返回大模型回答。
- 支持把文本转成大写，例如：`请把 hello agent 转为大写`。
- 支持统计英文单词，例如：`统计单词 I am learning agent development`。
- 支持文本摘要，例如：`请总结这段文字：第一句。第二句。第三句。`；工具会保留前两句。
- 支持模拟保存笔记：`save_note` 不会写入真实文件，必须由用户输入“确认”后才能执行。
- 支持受限文件读取：`read_file` 只读取项目 `data/` 目录中的 UTF-8 文本文件；它会拒绝越界路径，并对不存在的文件返回可读提示。
- 支持受限批量文件读取：`read_files` 接收每行一个文件名，一次最多读取 2 份 `data/` 目录内的 UTF-8 文本，并保留每段内容的来源文件名。
- 支持资料关键词搜索：`search_files` 只搜索项目 `data/` 目录中直接包含的 `.txt` 文件；关键词可匹配文件名或文件正文，英文关键词不区分大小写；空关键词会被拒绝，结果按相关度排序：文件名命中优先，正文命中次数更多的候选排在前面，仍相同时按文件名稳定排序；最多返回 3 个候选。需要多份资料时，Agent 会再调用 `read_files`。
- 支持带引用的向量检索：`search_knowledge_base` 只索引 `data/` 目录直接包含的 `.txt` 文件，按 400 字符切分并重叠 50 字符；它返回最多 3 段相关资料，每段包含来源文件、片段编号和相似度分数。阿里云百炼 `text-embedding-v4` 每批最多处理 10 段文本，程序会自动分批并保持结果顺序。
- 支持连续或同一轮的多个工具调用：一次用户请求最多执行三次工具；多个工具按模型给出的顺序执行。
- 工具按风险分级：三个文本工具自动执行；`save_note` 需要人工确认。混合了自动工具与需确认工具的同轮批次会被整体拒绝。
- 只允许调用已注册的工具，并校验工具名称、JSON 参数和 `text` 参数类型。
- 对缺少 API Key、模型调用异常、工具超过上限、错误参数和非结构化 DSML 伪工具调用文本等情况返回可读错误信息。
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
├── agent.py            # 第一阶段：规则式 LocalToolAgent
├── app/
│   ├── api.py           # FastAPI 知识库查询接口
│   ├── config.py        # 模型配置、系统提示和错误信息
│   ├── embeddings.py    # 阿里云百炼 Embedding Provider 与分批调用
│   ├── rag.py           # Chunk、向量索引、余弦相似度与 Top-K 排序
│   ├── tool_schemas.py  # 从工具定义导出的、发送给模型的 Schema
│   └── llm_agent.py     # 模型调用、参数校验、权限确认和工具调度
├── tools/
│   ├── text_tools.py    # 三个自动文本工具
│   ├── note_tools.py    # 需要确认的模拟保存工具
│   ├── file_tools.py    # 受限目录中的文本读取与关键词搜索工具
│   ├── rag_tools.py     # 受限知识库检索、索引缓存与来源格式化
│   └── registry.py      # 工具定义，以及自动生成的注册表、权限表和 Schema
├── data/                # Agent 可读取的受限资料目录
│   ├── agent_basics.txt         # Agent 基础概念
│   ├── agent_safety.txt         # Agent 安全与权限控制
│   ├── python_file_handling.txt # Python 文件处理
│   └── rag_long_test.txt        # 多 Chunk 长文本检索测试资料
├── main.py             # 命令行入口
├── tests/
│   ├── test_file_tools.py       # 受限读取与关键词搜索测试
│   ├── test_embeddings.py       # 云端 Embedding Provider 与分批调用测试
│   ├── test_main.py             # 第一阶段的规则式 Agent 测试
│   ├── test_api.py              # FastAPI 查询接口的离线测试
│   ├── test_llm_agent.py        # LLM Tool Calling 的离线测试
│   ├── test_rag.py              # Chunk、向量索引与 Top-K 测试
│   ├── test_rag_tools.py        # RAG 工具、缓存与引用格式测试
│   └── test_tool_definitions.py # 工具注册一致性测试
├── requirements.txt    # 项目依赖
└── README.md           # 项目说明
```

## 核心模块说明

### `tools/`：工具层

这里是真正做事的 Python 函数：

- `text_tools.py`：`upper_text(text)`、`count_words(text)` 和 `summarize_text(text)` 三个自动文本工具。
- `note_tools.py`：`save_note(text)`；模拟保存笔记，只返回成功文字，不创建或修改任何文件。
- `file_tools.py`：提供三个受限资料工具：`read_file(text)` 将 `text` 视为 `data/` 中的相对文件名，只读取 UTF-8 文本；`read_files(text)` 将每行视为一个相对文件名，一次最多读取 2 份文件，并在输出中标明每段内容的来源；`search_files(text)` 将 `text` 视为关键词，只搜索该目录直接包含的 `.txt` 文件。关键词可匹配文件名或正文，英文关键词使用 `casefold()` 忽略大小写；空白关键词直接返回提示。匹配结果会按相关度排序：文件名命中优先，其次正文命中次数更多优先，最后按文件名稳定排序；并只保留前 3 个候选，保证输出稳定且不会把过长列表交给模型。所有读取前都会确认路径仍位于 `data/`，拒绝 `../` 等越界输入；文件不存在或搜索无结果时返回可读提示。
- `registry.py`：唯一的工具定义列表 `TOOL_DEFINITIONS`。每项定义同时包含工具函数、权限、说明和参数 Schema；代码会据此自动生成函数注册表 `TOOL_REGISTRY`、权限表 `TOOL_PERMISSIONS` 和模型 Schema `TOOL_SCHEMAS`。

`TOOL_PERMISSIONS` 中，前三个文本工具是 `auto`（自动执行），`save_note` 是 `confirmation_required`（需要确认）。

大模型不会直接执行电脑中的 Python；它只能提出“调用哪个工具、传什么参数”。真正执行工具的是这一层代码。

### `app/`：Agent 调度层

- `config.py`：模型配置、系统提示和中文错误信息。
- `tool_schemas.py`：发送给模型的工具 Schema；只描述工具，不执行工具函数。
- `llm_agent.py`：模型调用、参数校验、权限确认和工具调度。

`LLMToolAgent` 负责整个 Tool Calling 流程：

1. 读取 `.env` 中的 `DEEPSEEK_API_KEY`。
2. 把可用工具的名称、用途和参数格式发送给 DeepSeek。
3. 接收模型的决定：直接回答，或请求调用一个或多个工具。
4. 校验工具调用是否安全、参数是否正确，并按权限等级决定是否需要人工确认。
5. 执行 `tools/` 中对应的函数，或保存一项待确认操作。
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

### `tests/`：测试文件

测试保证修改代码后，原有能力不会被意外破坏。

- `tests/test_main.py`：验证基础工具和规则式 Agent。
- `tests/test_llm_agent.py`：使用假的模型客户端，测试直接回答、工具调用、异常处理和参数校验；运行时不消耗 API 额度。

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
DASHSCOPE_API_KEY=你的_阿里云百炼_API_Key
DASHSCOPE_BASE_URL=https://你的_WorkspaceId.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
```

`.env` 已被 `.gitignore` 忽略。真实密钥绝不能写进代码、README、截图或 Git 提交记录。

向量检索会把 `data/` 中的文本片段和用户检索问题发送到阿里云百炼的 `text-embedding-v4` API；索引只保存在当前 Python 进程内，重启后会重新建立。该接口每批最多处理 10 段文本，程序会自动分批。返回结果会标注来源文件和片段编号，且不会索引或读取 `data/` 目录外的内容。

## FastAPI 查询接口

启动本地服务：

```bat
"D:\桌面\所有codex项目\AI agent 开发\python 学习\week06-llm-tool-calling\.venv\Scripts\python.exe" -m uvicorn app.api:app --reload
```

打开 `http://127.0.0.1:8000/docs`，在 `POST /knowledge-base/query` 点击 **Try it out**，输入：

```json
{
  "question": "如何确认副作用操作？",
  "top_k": 3
}
```

接口会返回 `results`；每条结果包含 `source`、`score` 和 `content`。`question` 为空或 `top_k` 不在 1 到 3 时，接口会返回 HTTP 422。检索不到资料时返回 HTTP 200 和空数组 `{"results": []}`。

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

也可以先让 Agent 在自己的受限资料中搜索：

```text
请输入请求：请搜索包含“工具”的资料，并用一句话告诉我找到的内容。
```

模型会先调用 `search_files`，Python 只在 `data/` 目录的 `.txt` 文件中搜索并返回匹配文件名；若需要一份资料，模型调用 `read_file`；若需要比较或汇总多份资料，模型调用 `read_files`，一次最多读取两份。随后模型可调用 `summarize_text`，再组织最终回答。这让搜索负责定位、读取负责获取内容，是一个小型的 RAG 工作流程。

当搜索到多个候选文件时，Agent 会结合用户问题选择最相关的一份再读取；如果用户要求摘要，它会继续调用 `summarize_text`。例如：

```text
请输入请求：请搜索与 Agent 有关的资料，选择与安全最相关的一份，读取后使用 summarize_text 工具用一句话总结。
```

在当前示例资料中，搜索会找到 `agent_basics.txt` 和 `agent_safety.txt`，随后 Agent 会选择并读取 `agent_safety.txt`。

如果需要两份资料的共同总结，可以这样请求：

```text
请输入请求：请查找 data 目录中与 Agent 有关的两份资料，读取两份内容后，使用 summarize_text 工具总结它们的共同重点。
```

模型会依次调用 `search_files`、`read_files` 和 `summarize_text`。`read_files` 最多读取两份文件，避免把过多原文放进模型上下文。

如果只记得资料文件名的一部分，也可以直接搜索文件名；例如搜索 `safety` 能找到 `agent_safety.txt`，即使文件正文中没有英文 `safety`。文件名搜索同样不区分英文大小写。

搜索词不能只包含空格；这时工具会返回“搜索关键词不能为空。”。若同一关键词匹配多份资料，工具会按相关度排序：文件名命中优先，正文命中次数更多的候选排在前面，仍相同时按文件名稳定排序；并最多返回 3 个候选，避免把过长的列表交给模型。

### 手动演示确认流程

目前 `main.py` 只处理一条输入后就结束；确认流程需要保留同一个 Agent 对象，因此可用 Python 交互模式演示：

```bat
python
```

```python
from app.llm_agent import LLMToolAgent
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
66 passed
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
7. **单一事实来源**：新增工具时只添加一项 `ToolDefinition`；函数注册表、权限表和模型 Schema 会自动同步生成，避免三处重复维护。
8. **受限文件访问**：文件工具只能访问预先允许的 `data/` 目录；路径规范化、目录范围校验与文件存在性检查必须发生在读取之前。
9. **先检索再读取**：当用户不知道文件名时，Agent 可先用 `search_files` 在受限资料中定位内容，再用 `read_file` 取得指定文件的全文。这是 RAG 检索流程的一个小型基础版本。
10. **多候选资料筛选**：关键词可能匹配多份资料。Agent 应先获得候选文件名，再依据用户问题选择最相关的一份读取；英文搜索使用 `casefold()` 忽略大小写差异。
11. **文件名与正文搜索**：用户可能只记得资料名称，也可能只记得正文内容。`search_files` 因此同时检查文件名与正文，只要其中之一匹配就返回该文件。
12. **检索结果质量**：搜索入口应拒绝空白关键词；候选结果应按相关度稳定排序（文件名命中、正文命中次数、文件名）并设置数量上限，避免同样请求产生不同顺序，或向模型传递过长的候选列表。
13. **多文件读取与汇总**：当问题需要比较或归纳多份资料时，模型可使用 `read_files` 一次读取最多两份来源明确的文件，再使用 `summarize_text` 归纳共同重点；数量上限用于控制上下文长度。

## 当前限制与下一步

目前项目已有三个自动文本工具、四个受限资料工具（单文件读取、批量读取、关键词搜索与向量检索）和一个模拟敏感工具；它已具备真正的最小 RAG：文档 Chunk、云端 Embedding、本地余弦相似度 Top-K 与来源引用。索引只在当前进程内缓存；重启后会重新向量化资料，因此长资料会产生外部 API 调用成本。同一轮多个工具会按顺序执行，不使用真正并行，单次请求最多执行三次。`main.py` 仍是单次输入入口，尚未做成可持续等待确认的命令行循环。后续可以继续扩展：

- 增加天气、网页搜索、数据库查询等工具。
- 对互不依赖的工具实现真正并行执行。
- 加入聊天历史与长期记忆。
- 将模拟保存替换为受限目录内的真实写入，并增加审计日志和更严格的参数规则。
- 建立 RAG 质量评测集，记录预期来源、Top-K 命中和失败原因，再改进 Chunk、Top-K 或排序策略。
- 当前已用 FastAPI 暴露只读查询接口；后续再增加受限上传与建库接口。
