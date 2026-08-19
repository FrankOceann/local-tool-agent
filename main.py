from llm_agent import LLMToolAgent


agent = LLMToolAgent()


def run_once(request: str) -> str:
    return agent.run(request)

if __name__ == "__main__":
    request = input("请输入请求: ")
    print(run_once(request))
