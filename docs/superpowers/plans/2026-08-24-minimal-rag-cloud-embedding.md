# Minimal Cloud-Embedding RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested, source-citing RAG tool that embeds allowed text files with OpenAI and retrieves relevant chunks locally.

**Architecture:** `app.rag` owns records, splitting, vector validation, cosine ranking, and the in-memory index. `app.embeddings` owns the replaceable Provider protocol and OpenAI boundary. `tools.rag_tools` owns the Agent-facing cache and citation format; the registry exposes it as an automatic read-only tool.

**Tech Stack:** Python 3, pytest, python-dotenv, existing `openai` SDK, OpenAI `text-embedding-3-small`.

**Spec:** `docs/superpowers/specs/2026-08-24-minimal-rag-cloud-embedding-design.md`

## Global Constraints

- Enumerate only direct UTF-8 `*.txt` files under `tools.file_tools.DATA_DIRECTORY`; never accept a model-supplied path.
- Use `CHUNK_SIZE = 400`, `CHUNK_OVERLAP = 50`, and a default/maximum `top_k = 3`.
- Preserve `source_file`, zero-based `chunk_index`, text, and a citation formatted `[来源: filename#chunk-N | score=0.8732]`.
- Production embedding uses `OPENAI_API_KEY` and `text-embedding-3-small`; no secret appears in code, tests, logs, or commits.
- Tests use only deterministic fakes; no network or real API key.
- Existing `read_file`, `read_files`, and `search_files` behavior remains unchanged.

---

## File Structure

| File | Responsibility |
|---|---|
| `app/embeddings.py` | Provider protocol and OpenAI-backed embedding implementation. |
| `app/rag.py` | Chunk/result records, splitter, safe vector math, index build/search. |
| `tools/rag_tools.py` | Restricted cached `search_knowledge_base(text)` and citations. |
| `tools/registry.py` | Automatic RAG ToolDefinition. |
| `app/config.py` | RAG prompt instructions and configuration messages. |
| `tests/test_rag.py` | Split/index/ranking tests using a fake provider. |
| `tests/test_embeddings.py` | Provider configuration and fake-client tests. |
| `tests/test_rag_tools.py` | Safe directory, output, cache, and limit tests. |

### Task 1: Chunk records and deterministic splitting

**Files:**

- Create: `app/rag.py`
- Create: `tests/test_rag.py`

**Interfaces:**

- Consumes: none.
- Produces: `DocumentChunk` and `split_text(source_file: str, text: str) -> list[DocumentChunk]`; `CHUNK_SIZE` and `CHUNK_OVERLAP`.

- [ ] **Step 1: Write the failing tests**

```python
from app.rag import CHUNK_OVERLAP, CHUNK_SIZE, split_text


def test_split_text_preserves_source_indexes_and_overlap():
    chunks = split_text("lesson.txt", "a" * (CHUNK_SIZE + 20))

    assert [(item.source_file, item.chunk_index) for item in chunks] == [
        ("lesson.txt", 0),
        ("lesson.txt", 1),
    ]
    assert chunks[0].text == "a" * CHUNK_SIZE
    assert chunks[1].text == "a" * (20 + CHUNK_OVERLAP)


def test_split_text_ignores_whitespace_only_documents():
    assert split_text("empty.txt", " \n\t ") == []
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_rag.py -q`

Expected: FAIL during collection because `app.rag` does not exist.

- [ ] **Step 3: Implement the minimum splitter**

```python
from dataclasses import dataclass

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


@dataclass(frozen=True)
class DocumentChunk:
    source_file: str
    chunk_index: int
    text: str


def split_text(source_file: str, text: str) -> list[DocumentChunk]:
    if not text.strip():
        return []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    return [
        DocumentChunk(source_file, index, text[start:start + CHUNK_SIZE])
        for index, start in enumerate(range(0, len(text), step))
        if text[start:start + CHUNK_SIZE]
    ]
```

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_rag.py -q; python -m pytest -q`

Expected: two new tests pass; all existing 51 tests remain green.

- [ ] **Step 5: Commit**

```bash
git add app/rag.py tests/test_rag.py
git commit -m "feat: add RAG document chunking"
```

### Task 2: Provider protocol, index, and local cosine Top-K

**Files:**

- Create: `app/embeddings.py`
- Modify: `app/rag.py`
- Modify: `tests/test_rag.py`

**Interfaces:**

- Consumes: Task 1 `DocumentChunk` and `split_text`.
- Produces: `EmbeddingProvider.embed_texts(texts: list[str]) -> list[list[float]]`, `SearchResult`, `RAGIndex.build(data_directory: Path, provider: EmbeddingProvider) -> RAGIndex`, and `RAGIndex.search(query: str, top_k: int = 3) -> list[SearchResult]`.

- [ ] **Step 1: Write failing ranking and empty-index tests**

```python
from app.rag import RAGIndex


class FakeEmbeddingProvider:
    vectors = {
        "Python 用 pathlib 读取文件。": [1.0, 0.0],
        "Agent 需要工具权限校验。": [0.0, 1.0],
        "怎样安全读取资料？": [0.0, 1.0],
    }

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


def test_index_returns_results_in_descending_similarity(tmp_path):
    (tmp_path / "python.txt").write_text("Python 用 pathlib 读取文件。", encoding="utf-8")
    (tmp_path / "safety.txt").write_text("Agent 需要工具权限校验。", encoding="utf-8")
    index = RAGIndex.build(tmp_path, FakeEmbeddingProvider())

    results = index.search("怎样安全读取资料？", top_k=3)

    assert [(item.chunk.source_file, item.chunk.chunk_index) for item in results] == [
        ("safety.txt", 0),
        ("python.txt", 0),
    ]
    assert results[0].score > results[1].score


def test_index_returns_empty_for_blank_query_or_no_chunks(tmp_path):
    index = RAGIndex.build(tmp_path, FakeEmbeddingProvider())

    assert index.search("   ") == []
    assert index.search("怎样安全读取资料？") == []
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_rag.py -q`

Expected: FAIL because `RAGIndex` is not defined.

- [ ] **Step 3: Implement the smallest index**

```python
# app/embeddings.py
from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
```

```python
# app/rag.py additions
@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float


class RAGIndex:
    @classmethod
    def build(cls, data_directory: Path, provider: EmbeddingProvider) -> "RAGIndex": ...

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]: ...
```

`build` must sort `data_directory.glob("*.txt")`, skip whitespace-only files, embed non-empty chunks, and reject count/dimension/zero-norm errors with Chinese `ValueError` messages. `search` returns `[]` for blank query or an empty index, clamps `top_k` to 1–3, and orders by `(-score, source_file, chunk_index)`.

- [ ] **Step 4: Add and pass a vector-validation regression test**

```python
def test_index_rejects_mismatched_vector_dimensions(tmp_path):
    class BadProvider:
        def embed_texts(self, texts):
            return [[1.0, 0.0], [1.0]]

    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")

    with pytest.raises(ValueError, match="向量维度不一致"):
        RAGIndex.build(tmp_path, BadProvider())
```

Run: `python -m pytest tests/test_rag.py -q`

Expected before the guard: FAIL because dimensions are not checked. Add the validation; rerun until PASS.

- [ ] **Step 5: Verify and commit**

```bash
python -m pytest tests/test_rag.py -q
python -m pytest -q
git add app/embeddings.py app/rag.py tests/test_rag.py
git commit -m "feat: add in-memory RAG retrieval index"
```

### Task 3: OpenAI embedding boundary

**Files:**

- Modify: `app/embeddings.py`
- Modify: `app/config.py`
- Modify: `.env.example`
- Create: `tests/test_embeddings.py`

**Interfaces:**

- Consumes: Task 2 `EmbeddingProvider` and the installed `openai.OpenAI` SDK.
- Produces: `OpenAIEmbeddingProvider(client: object | None = None, api_key: str | None = None)` and `EMBEDDING_MODEL_NAME = "text-embedding-3-small"`.

- [ ] **Step 1: Write failing fake-client and missing-key tests**

```python
from app.embeddings import OpenAIEmbeddingProvider


class FakeEmbeddings:
    def create(self, *, model, input):
        assert model == "text-embedding-3-small"
        assert input == ["第一段", "第二段"]
        return type("Response", (), {"data": [
            type("Item", (), {"embedding": [1.0, 0.0]})(),
            type("Item", (), {"embedding": [0.0, 1.0]})(),
        ]})()


class FakeClient:
    embeddings = FakeEmbeddings()


def test_openai_provider_returns_embeddings_from_client():
    provider = OpenAIEmbeddingProvider(client=FakeClient(), api_key="test-key")

    assert provider.embed_texts(["第一段", "第二段"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_openai_provider_reports_missing_key_without_network():
    provider = OpenAIEmbeddingProvider(api_key="")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        provider.embed_texts(["第一段"])
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_embeddings.py -q`

Expected: FAIL during collection because `OpenAIEmbeddingProvider` is absent.

- [ ] **Step 3: Implement and isolate the cloud call**

```python
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
MISSING_EMBEDDING_KEY_MESSAGE = "未检测到 OPENAI_API_KEY，请在 .env 中配置后重试。"


class OpenAIEmbeddingProvider:
    def __init__(self, client: object | None = None, api_key: str | None = None): ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise ValueError(MISSING_EMBEDDING_KEY_MESSAGE)
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL_NAME,
                input=texts,
            )
        except Exception as error:
            raise ValueError(f"Embedding 服务调用失败：{error}") from error
        return [item.embedding for item in response.data]
```

Call `load_dotenv()` in the constructor. Read `OPENAI_API_KEY` only when `api_key` is not provided. Create `OpenAI(api_key=self.api_key)` only when a key exists and no fake client was injected. Add empty `OPENAI_API_KEY=` to `.env.example`.

- [ ] **Step 4: Test error wrapping, then make it green**

```python
def test_openai_provider_does_not_expose_key_on_api_error():
    class BrokenEmbeddings:
        def create(self, **kwargs):
            raise RuntimeError("rate limited")

    client = type("Client", (), {"embeddings": BrokenEmbeddings()})()
    provider = OpenAIEmbeddingProvider(client=client, api_key="secret-value")

    with pytest.raises(ValueError, match="Embedding 服务调用失败：rate limited") as error:
        provider.embed_texts(["第一段"])

    assert "secret-value" not in str(error.value)
```

Run: `python -m pytest tests/test_embeddings.py -q`

Expected before wrapping: FAIL with `RuntimeError`. Add the wrapper shown above, then rerun until PASS.

- [ ] **Step 5: Verify and commit**

```bash
python -m pytest tests/test_embeddings.py -q
python -m pytest -q
git add app/embeddings.py app/config.py .env.example tests/test_embeddings.py
git commit -m "feat: add OpenAI embedding provider"
```

### Task 4: Restricted Agent tool, citations, and docs

**Files:**

- Create: `tools/rag_tools.py`
- Modify: `tools/registry.py`
- Modify: `app/config.py`
- Modify: `README.md`
- Modify: `tests/test_rag_tools.py`
- Modify: `tests/test_tool_definitions.py`
- Modify: `tests/test_llm_agent.py`

**Interfaces:**

- Consumes: Tasks 2–3 `RAGIndex` and `OpenAIEmbeddingProvider`, plus existing `DATA_DIRECTORY` and `TEXT_PARAMETERS`.
- Produces: automatically registered `search_knowledge_base(text: str) -> str`.

- [ ] **Step 1: Write failing citation and blank-query tests**

```python
from tools import rag_tools


class FakeIndex:
    def search(self, query, top_k=3):
        assert query == "如何保护文件？"
        assert top_k == 3
        chunk = type("Chunk", (), {
            "source_file": "agent_safety.txt",
            "chunk_index": 0,
            "text": "文件访问必须校验权限。",
        })()
        return [type("Result", (), {"chunk": chunk, "score": 0.87321})()]


def test_search_knowledge_base_formats_source_and_score(monkeypatch):
    monkeypatch.setattr(rag_tools, "get_rag_index", lambda: FakeIndex())

    assert rag_tools.search_knowledge_base("如何保护文件？") == (
        "[来源: agent_safety.txt#chunk-0 | score=0.8732]\n"
        "文件访问必须校验权限。"
    )


def test_search_knowledge_base_rejects_blank_query_without_building_index(monkeypatch):
    monkeypatch.setattr(rag_tools, "get_rag_index", lambda: pytest.fail("不应建库"))

    assert rag_tools.search_knowledge_base("  ") == "检索问题不能为空。"
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_rag_tools.py -q`

Expected: FAIL during collection because `tools.rag_tools` is absent.

- [ ] **Step 3: Implement the restricted cache and formatter**

```python
_RAG_INDEX: RAGIndex | None = None


def get_rag_index() -> RAGIndex:
    global _RAG_INDEX
    if _RAG_INDEX is None:
        _RAG_INDEX = RAGIndex.build(DATA_DIRECTORY, OpenAIEmbeddingProvider())
    return _RAG_INDEX


def search_knowledge_base(text: str) -> str:
    query = text.strip()
    if not query:
        return "检索问题不能为空。"
    try:
        results = get_rag_index().search(query, top_k=3)
    except ValueError as error:
        return str(error)
    if not results:
        return "知识库中没有可用的相关资料。"
    return "\n\n".join(
        f"[来源: {item.chunk.source_file}#chunk-{item.chunk.chunk_index} | score={item.score:.4f}]\n{item.chunk.text}"
        for item in results
    )
```

Import `DATA_DIRECTORY` from `tools.file_tools`; no input may alter it. Append an automatic `search_knowledge_base` definition using `TEXT_PARAMETERS` in `tools/registry.py`.

- [ ] **Step 4: Add and pass integration regressions**

Extend `tests/test_tool_definitions.py` so its expected final name is `search_knowledge_base`. Add a `tests/test_llm_agent.py` assertion that `SYSTEM_PROMPT` names `search_knowledge_base` and says not to invent source labels. Add a no-result fake-index test expecting `知识库中没有可用的相关资料。`. Run the focused tests before changes and observe failure; add registry/prompt code, then rerun to green.

- [ ] **Step 5: Update docs, verify, and commit**

Add to `README.md`: set `OPENAI_API_KEY` in `.env`; documents and queries are sent to the Embeddings API; the index rebuilds after a process restart; citations identify file and chunk; no content outside `data/` is indexed. Then run:

```bash
python -m pytest tests/test_rag_tools.py tests/test_tool_definitions.py tests/test_llm_agent.py -q
python -m pytest -q
git diff --check
git add tools/rag_tools.py tools/registry.py app/config.py .env.example README.md tests/test_rag_tools.py tests/test_tool_definitions.py tests/test_llm_agent.py
git commit -m "feat: expose cited RAG search tool"
```

### Task 5: Full verification and real Agent demonstration

**Files:**

- Modify: no production file unless a failed demonstration is first captured by a regression test.
- Test: `tests/test_rag.py`, `tests/test_embeddings.py`, `tests/test_rag_tools.py`, and full suite.

**Interfaces:**

- Consumes: Tasks 1–4.
- Produces: verified demonstration evidence.

- [ ] **Step 1: Verify automated behavior**

```bash
python -m pytest -q
git diff --check
git status --short
```

Expected: all tests pass, `git diff --check` has no output, and `test_file_tools.before-pr3.py.bak` remains untracked and untouched.

- [ ] **Step 2: Prepare a real key privately**

Confirm `.env` contains a non-empty `OPENAI_API_KEY`. Do not print, stage, or commit that file. If it is absent, stop this manual demonstration; all automated tests remain complete.

- [ ] **Step 3: Run one constrained demonstration**

Run: `python main.py`

Input: `根据本地资料说明 Agent 为什么需要工具权限校验，并保留资料来源。`

Expected: the Agent calls `search_knowledge_base`, receives no more than three cited blocks, and preserves at least one returned source label. If prompt guidance is insufficient, write a failing prompt test before changing production code.

- [ ] **Step 4: Commit only a test-proven correction**

```bash
python -m pytest -q
git add <regression-test-and-minimal-fix>
git commit -m "fix: preserve RAG source citations"
```

Skip this commit when no correction is needed. Never commit `.env`, `test_file_tools.before-pr3.py.bak`, or either protected existing worktree.

## Plan Self-Review

- Spec coverage: Tasks 1–2 cover chunks, metadata, directory boundary, cosine Top-K, empty documents/results, and invalid vectors. Task 3 covers cloud configuration and error/secret boundaries. Task 4 covers cache, tool registration, citations, prompt, and docs. Task 5 covers regression verification and the live Agent demonstration.
- Placeholder scan: every task has exact paths, interfaces, commands, test assertions, expected RED state, minimal implementation content, and commit message.
- Type consistency: `EmbeddingProvider.embed_texts` returns `list[list[float]]`; `RAGIndex.build` consumes it; `RAGIndex.search` returns `list[SearchResult]`; the Agent tool converts results into the existing `str` tool contract.

