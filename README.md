# Local Tool Agent

这是一个用于学习 Agent Tool Calling 基础流程的 Python 命令行项目。

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

## DeepSeek 配置

安装依赖：

```bat
python -m pip install -r requirements.txt
```

在项目根目录创建 `.env` 文件：

```text
DEEPSEEK_API_KEY=你的_DeepSeek_API_Key
```

`.env` 已被忽略；真实密钥绝不能进入代码、README 或提交记录。

## 流程

用户输入 → LLMToolAgent → DeepSeek 选择工具 → tools.py 执行 → 工具结果回传 DeepSeek → 输出最终回答
