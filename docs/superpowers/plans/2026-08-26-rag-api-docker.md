# RAG API Docker 可复现启动 Implementation Plan

**Goal:** 为既有只读 RAG API 提供安全、可复现的单容器 Docker 启动方式。

**Spec:** `docs/superpowers/specs/2026-08-26-rag-api-docker-design.md`

## 全局约束

- 使用 `python:3.11-slim`、`/app`、端口 `8000` 与 `uvicorn app.api:app --host 0.0.0.0 --port 8000`。
- 不新增依赖或任何超出只读 API 的功能。
- 不将 `.env` 或密钥放进镜像；资料 `data/` 随镜像只读复制。
- 不删除、移动、重置、暂存或提交受保护备份、现有 worktree、无关未跟踪文档或 `wip: top-k README notes` stash。
- 全部用户命令使用 Windows CMD 的 `cd /d`。

### Task 1: TDD 配置契约测试

创建 `tests/test_docker_config.py`，先读取不存在的 `Dockerfile` 与 `.dockerignore`，验证它们会要求：Python 3.11 slim、`/app`、依赖先安装、复制 `app/`/`tools/`/`data/`、端口 8000、Uvicorn 命令、Dockerfile 不含 `.env`，以及 `.dockerignore` 排除 `.env`、`.venv/`、缓存、`.git/`、`.worktrees/`、`tests/`、`docs/` 和 `*.bak`。先运行测试确认失败。

### Task 2: 最小 Docker 配置

由学习者创建 `Dockerfile` 与 `.dockerignore` 使 Task 1 通过。Dockerfile 仅包含基础镜像、工作目录、按缓存友好顺序复制和安装依赖、运行时目录、端口和 CMD；`.dockerignore` 仅排除本机非运行时文件和敏感文件。

### Task 3: README

在现有 FastAPI 查询接口说明后新增 Docker 小节，提供 CMD 格式的 `docker build -t local-tool-agent-rag-api .`、使用 `--env-file .env` 的 `docker run --rm --name local-tool-agent-rag-api -p 8000:8000 ...`，以及 `/docs` 和真实查询说明。不得出现真实 Key。

### Task 4: 验收

完整 pytest 预期为 84 passed；构建镜像；启动容器；另一 CMD 窗口以 `curl` 访问 `/docs` 和 `POST /knowledge-base/query`；用 Ctrl+C 停止前台容器。

### Task 5: GitHub 工作流

仅暂存 Dockerfile、`.dockerignore`、README、Docker 配置测试和这两份 Docker 学习文档，创建 `feat: add Docker startup for RAG API` 提交，推送 PR、合并并同步 main。绝不使用 `git add .`。
