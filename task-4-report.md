# Task 4 Report

## 状态

已完成 CLI 与文档更新。

## 变更

- `main.py` 现在构造 `LLMToolAgent`，并保留 `run_once()` 和交互式入口。
- `test_main.py` 使用 monkeypatch 验证 `run_once()` 返回 `LLMToolAgent.run()` 的模型结果。
- `README.md` 补充 DeepSeek 依赖和 `.env` 配置说明，并更新工具调用流程。

## 验证

- TDD 红灯：`python -m pytest -q test_main.py` 在改动 `main.py` 前按预期失败，旧入口返回本地代理结果而非 `MODEL RESULT`。
- TDD 绿灯：`python -m pytest -q test_main.py` — 6 passed。
- 完整测试：`python -m pytest -q` — 12 passed。
- 未检测到真实 DeepSeek 密钥，因此未运行 `python main.py`。

## 注意事项

`.env` 已被 `.gitignore` 忽略；未读取或输出任何密钥值。
