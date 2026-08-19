from llm_agent import LLMToolAgent, MISSING_KEY_MESSAGE


def test_agent_explains_how_to_configure_a_missing_api_key():
    assert LLMToolAgent(api_key="").run("大写 hello") == MISSING_KEY_MESSAGE
