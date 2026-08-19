from tools import count_words, upper_text
from agent import LocalToolAgent

def test_upper_text():
    assert upper_text("hello") == "HELLO"


def test_count_words():
    assert count_words("I am learning agent development") == 5

def test_agent_calls_upper_text_tool():
    agent = LocalToolAgent()

    assert agent.run("大写 hello world") == "HELLO WORLD"

def test_agent_calls_count_words_tool():
    agent = LocalToolAgent()

    assert agent.run("统计单词 I am learning agent development") == "英文单词数量: 5"

def test_agent_rejects_unsupported_request():
    agent = LocalToolAgent()

    assert agent.run("今天北京天气如何") == (
        "暂不支持这个请求。可使用：大写 <文字> 或 统计单词 <英文文字>"
    )

from main import run_once


def test_run_once_returns_agent_result():
    assert run_once("大写 codex") == "CODEX"