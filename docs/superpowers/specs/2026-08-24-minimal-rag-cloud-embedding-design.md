# Week06 最小 RAG（云端 Embedding）设计

## 目标

在现有受限 `data/` 文件检索能力上增加一个可演示、可测试的最小 RAG 链路：将 `.txt` 资料切分为带来源信息的文本块，调用云端 Embedding API 生成向量，用本地余弦相似度选出最相关的内容，并让 Agent 在回答中获得可引用的来源片段。

第一版默认使用 OpenAI `text-embedding-3-small`。该模型通过 Embeddings API 生成文本向量；模型选择封装在 Provider 内，后续可替换而不改变切分、索引或检索逻辑。

## 范围与非目标

本阶段包含：固定字符切分、元数据、云端向量生成、内存索引、余弦相似度 Top-K、受限检索工具、来源标识、自动化测试和一次真实 Agent 演示。

本阶段不包含：Chroma、FAISS、持久化向量数据库、上传文件、重排模型、对话记忆、批量 API、异步调用或自动重建跨进程索引。索引只保存在当前 Python 进程的内存中；程序重启后下次检索会重新建立索引。

## 约束与配置

- 只读取 `DATA_DIRECTORY` 直接目录下的 UTF-8 `.txt` 文件；不接受用户提供的路径或目录。
- `CHUNK_SIZE = 400` 个 Python 字符；`CHUNK_OVERLAP = 50` 个字符。重叠必须小于块大小。
- 默认 `top_k = 3`，允许范围为 1 至 3；避免模型上下文被检索结果占满。
- 真实 Provider 从 `.env` 读取 `OPENAI_API_KEY`，默认模型为 `text-embedding-3-small`；密钥绝不写入代码、README 或测试。
- 测试只用确定性 Fake Provider，禁止联网、下载模型或调用真实 API。
- 原资料与查询文本会被发送到 Embedding API；README 必须说明该隐私和网络边界。

## 目录与模块职责

```text
app/
  embeddings.py       # Provider 协议、OpenAI Provider、环境配置与 API 错误转换
  rag.py              # Chunk、内存索引、余弦相似度与检索结果
tools/
  rag_tools.py        # Agent 可调用的受限检索工具与格式化来源
  registry.py         # 注册 search_knowledge_base 工具
tests/
  test_rag.py         # 切分、索引、排序、空资料、来源元数据
  test_rag_tools.py   # 工具输出、Top-K 与安全边界
```

`tools/file_tools.py` 保留已有关键词搜索与读取功能，不改变其接口。RAG 不读取 `read_file` 传入的任意文件名，而是直接复用 `DATA_DIRECTORY` 的受限 `.txt` 枚举规则。

## 数据模型与接口

```python
@dataclass(frozen=True)
class DocumentChunk:
    source_file: str
    chunk_index: int
    text: str

@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float

class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
```

`split_text(source_file: str, text: str) -> list[DocumentChunk]` 产生从 0 开始编号的非空块。空白文件不产生块。`RAGIndex.build(data_directory, provider)` 只枚举 `*.txt`，批量对非空块生成向量，并验证返回数量与块数量一致。

`RAGIndex.search(query: str, top_k: int = 3) -> list[SearchResult]` 会拒绝空白查询；若索引没有文本块，返回空列表；否则嵌入查询并按余弦相似度从高到低排序。相同分数时按 `source_file`、`chunk_index` 升序稳定排序，确保结果可测试、可复现。

向量维度不一致、零长度向量、全零向量和 API 失败都转换为清晰的中文错误信息，不产生不可信分数。

## 索引与 Agent 数据流

```text
data/*.txt
  → split_text（400 字符、50 字符重叠）
  → DocumentChunk + source_file/chunk_index
  → OpenAIEmbeddingProvider.embed_texts
  → RAGIndex（内存中的 Chunk + 向量）

用户问题
  → search_knowledge_base(query)
  → 查询向量 → 余弦相似度 → Top-3 SearchResult
  → "[来源: 文件名#chunk-N | score=…]" + Chunk 文本
  → LLMToolAgent → 带来源的最终回答
```

`search_knowledge_base` 是自动执行的只读工具，参数仍使用现有统一的 `text` Schema。它首次被调用时建立进程内索引，后续调用复用该索引；测试可以显式注入 Fake Provider 和新的临时目录。构建或检索失败时工具返回中文错误消息，Agent 不得伪造来源。

工具返回最多三段，格式为：

```text
[来源: agent_safety.txt#chunk-0 | score=0.8732]
文件访问必须校验权限……
```

System Prompt 增加约束：使用该工具后的回答应保留其提供的 `[来源: ...]` 标识；没有结果或工具报错时，应说明资料不足或检索失败，不得引用不存在的片段。

## 错误、安全与成本边界

- 文件系统边界沿用 `DATA_DIRECTORY`：仅 `*.txt`，不递归，不允许路径穿越。
- 没有 `OPENAI_API_KEY` 时，真实 Provider 返回配置提示，而不是尝试网络请求。
- 网络、认证、限流或响应格式错误被包装为“Embedding 服务调用失败：原因”的用户可读错误；密钥不出现在错误文本或日志中。
- 每个新或变更的资料块在进程首次建库时产生一次 API 调用输入；本阶段缓存仅在本进程有效。资料变更后需要重启进程以重建索引。
- 检索不等于事实保证：相似度低、无结果或资料未涵盖问题时，Agent 必须明确资料不足。

## 测试与验收

所有新增行为先写失败测试，再实现最小代码。Fake Provider 用固定文本到向量映射，覆盖：

1. 固定长度、重叠和 `chunk_index`；空白资料不产生 Chunk。
2. 仅枚举受限目录的 `.txt` 文件，不递归、不可通过查询访问其他路径。
3. 余弦相似度 Top-K 的分数降序、相同分数稳定排序和上限。
4. 空查询、空索引、维度不一致与零向量的明确处理。
5. 工具输出中的文件名与 `chunk_index` 引用，以及至多三个结果。
6. 注册表包含 `search_knowledge_base` 且 Agent Schema 可见。
7. 完整回归：`python -m pytest -q`；随后用真实 API 做一次“提问 → 工具检索 → 带引用回答”的手动演示。

## 后续衔接

第 2 周可在不改变 `DocumentChunk`、`SearchResult` 或 `EmbeddingProvider` 的前提下，对比 Chroma/FAISS、LangChain 与 LlamaIndex；本地 Provider 也可作为离线与隐私方案补入。
