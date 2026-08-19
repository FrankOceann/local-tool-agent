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

## 流程

用户输入 → LocalToolAgent 选择工具 → tools.py 执行 → 输出结果