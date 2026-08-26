# RAG API Docker 可复现启动设计

## 目标与边界

将现有 FastAPI 只读知识库查询接口打包为单一 Docker 容器。陌生环境只需 Docker 和运行时环境变量即可启动 `POST /knowledge-base/query`，不必手工创建虚拟环境、安装依赖或记忆 Uvicorn 命令。

不引入 Docker Compose、数据库容器、上传、索引重建接口、认证、Rerank、短期记忆或长期记忆。

## 设计

- 基础镜像为 `python:3.11-slim`，工作目录为 `/app`。
- 先复制 `requirements.txt` 并运行 `pip install --no-cache-dir -r requirements.txt`，再复制 `app/`、`tools/` 和只读 `data/`，以复用依赖缓存层。
- 暴露端口 `8000`，入口命令为 `uvicorn app.api:app --host 0.0.0.0 --port 8000`，不使用开发热重载。
- `.dockerignore` 排除 `.env`、虚拟环境、缓存、Git 元数据、worktree、测试、文档与 `*.bak`；不会删除这些本地文件。
- 构建镜像不传递密钥。启动时以 `--env-file .env` 或 `-e DASHSCOPE_API_KEY=...` 从宿主机注入 `DASHSCOPE_API_KEY`，可选传入 `DASHSCOPE_BASE_URL`。
- `data/` 被复制为镜像内的只读资料。索引只在容器进程内缓存，首次真实查询会调用 DashScope Embedding，容器重启后会重新建立。

## 验收

1. 既有 `python -m pytest -q` 仍通过，并新增 Docker 配置契约测试。
2. `docker build` 成功。
3. 传入有效环境变量后，容器可启动，`/docs` 可访问。
4. 真实查询返回 HTTP 200，结果含 `source`、`score` 和 `content`。
