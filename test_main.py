from tools import count_words, upper_text
from agent import LocalToolAgent
import tools

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

from llm_agent import LLMToolAgent
from main import run_once


def test_run_once_returns_llm_tool_agent_result(monkeypatch):
    monkeypatch.setattr(LLMToolAgent, "run", lambda self, request: "MODEL RESULT")

    assert run_once("任意请求") == "MODEL RESULT"

def test_summarize_text_keeps_first_two_sentences():
    summarize_text = getattr(tools, "summarize_text", None)

    assert summarize_text is not None
    assert summarize_text("第一句。第二句。第三句。") == "第一句。第二句。"