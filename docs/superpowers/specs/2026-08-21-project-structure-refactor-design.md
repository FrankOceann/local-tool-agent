# Week06 项目化整理设计

## 目标

将 Week06 的 Tool Agent 从根目录平铺文件整理为职责清晰的 Python 项目结构。整理不改变现有功能、模型配置、工具权限或安全确认行为，为后续增加真实工具、对话记忆、RAG 和并行调用建立模块边界。

## 范围

本阶段只移动和拆分现有代码，并更新导入路径、测试路径和 README。保留以下行为：

- DeepSeek Tool Calling、最多三次工具调用和同轮工具调用顺序。
- `upper_text`、`count_words`、`summarize_text` 自动执行。
- `save_note` 的模拟保存、确认、取消和待确认状态。
- `.env` 中的 `DEEPSEEK_API_KEY` 私密配置方式。
- 现有命令：`python main.py` 和 `python -m pytest -q`。

本阶段不增加真实文件写入、外部 API、聊天记忆、并行执行或确认后恢复模型对话。

## 目录结构

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
├─ README.md
├─ requirements.txt
├─ .env.example
└─ .gitignore
```

根目录保留既有的 `agent.py`，因此 Agent 主模块使用 `app/` 目录，避免 Windows 中同名文件与目录冲突。

## 模块职责

### `main.py`

命令行入口。读取一条用户请求、创建 `LLMToolAgent` 并打印结果；不包含工具 Schema、模型调用循环或权限判断。

### `app/config.py`

集中放置模型名、DeepSeek 基础地址、系统提示和用户可读的错误消息。`llm_agent.py` 不再直接散落配置常量。

### `app/tool_schemas.py`

集中声明发送给 DeepSeek 的 `TOOL_SCHEMAS`。Schema 只描述模型可请求的工具名称与参数格式，不执行工具。

### `app/llm_agent.py`

保留 Agent 调度职责：加载环境变量、调用模型、检查调用上限、校验工具参数、根据权限执行或保存待确认调用、向模型回传自动工具结果。

### `tools/text_tools.py`

放置 `upper_text`、`count_words` 和 `summarize_text`。这些函数只处理文本，不了解模型、API Key 或确认状态。

### `tools/note_tools.py`

放置 `save_note`。它继续是模拟保存函数，只返回说明文字，不进行真实文件写入。

### `tools/registry.py`

汇总工具函数与风险权限：

- `TOOL_REGISTRY`：工具名称到 Python 函数的映射；
- `TOOL_PERMISSIONS`：工具名称到 `auto` 或 `confirmation_required` 的映射。

Agent 只从这里取得可执行工具和权限，不直接导入各个工具函数。

### `tests/`

测试文件迁移到独立目录。测试仍使用 Fake Client 模拟模型响应，不依赖网络或真实 API Key。

## 数据流

```text
用户输入
  → main.py
  → app.llm_agent.LLMToolAgent
  → app.config / app.tool_schemas / tools.registry
  → tools.text_tools 或 tools.note_tools
  → Agent 返回最终结果
```

`save_note` 被模型请求时，Agent 先读取 `TOOL_PERMISSIONS`，保存待确认调用并要求用户确认；确认、取消和待确认状态仍只由 Python 本地处理。

## 兼容与错误边界

- `.env` 仍在项目根目录，且继续由 `.gitignore` 忽略；不移动或提交真实 Key。
- 旧根目录导入会在本阶段统一改为新包导入，避免同时维护两套实现。
- 未知工具、无效 JSON、错误 `text` 参数、API 调用失败和超过调用上限的中文错误信息保持不变。
- 任何整理引入的导入错误必须由测试先发现并修正。

## 测试与验收

1. `python -m pytest -q` 从项目根目录运行，全部测试通过；当前基线为 30 个测试。
2. `python main.py` 仍能读取 `.env` 并处理一条普通请求。
3. `git diff --check` 没有代码格式错误。
4. README 的目录图、运行命令和模块说明与实际结构一致。

## 后续衔接

完成整理后，优先学习通用工具注册机制；届时新工具可在 `tools/` 下增加实现，并由注册模块生成或提供对应元数据。之后再扩展受限文件读取、对话记忆、RAG 和并行执行。
