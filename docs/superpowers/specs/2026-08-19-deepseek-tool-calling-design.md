# Week06 DeepSeek Tool Calling Design

## Goal

Extend the completed local rule-based Agent into a command-line Agent that asks DeepSeek to choose a text tool, executes only an approved local function, and gives the tool result back to DeepSeek for a final natural-language reply.

## Scope

- Use the OpenAI Python SDK with DeepSeek's OpenAI-compatible endpoint: `https://api.deepseek.com`.
- Use `deepseek-v4-flash` for this learning prototype.
- Support the existing `upper_text` and `count_words` tools only.
- Allow no more than one tool call per user request.
- Keep the existing `agent.py` rule-based Agent as a comparison implementation.

## Components

- `tools.py`: unchanged local tool functions.
- `agent.py`: existing keyword-based Agent, retained for comparison.
- `llm_agent.py`: new `LLMToolAgent` that sends the user message and tool schemas to DeepSeek, validates the returned tool call, runs an allowed tool, and requests a final answer.
- `main.py`: command-line entry point that uses `LLMToolAgent`.
- `.env`: local-only `DEEPSEEK_API_KEY`; ignored by Git.
- `.env.example`: committed empty template showing the required environment variable.
- `test_llm_agent.py`: tests with a fake model client, so automated tests do not use the network or consume API credit.

## Data Flow

```text
user input
  -> LLMToolAgent
  -> DeepSeek chat completion with tool schemas
  -> validated tool call (or direct model answer)
  -> approved local tool execution
  -> tool result sent back to DeepSeek
  -> final answer printed by main.py
```

## Tool Contract

DeepSeek receives two function schemas:

- `upper_text(text: string)`
- `count_words(text: string)`

The Python tool registry maps those exact names to local functions. A model-provided name outside this registry is rejected. Tool arguments are parsed from JSON and require a string `text` value before a tool can run.

## Errors and Safety

- Missing `DEEPSEEK_API_KEY`: return a setup message without sending a request.
- API or network failure: return a readable error message.
- Invalid JSON, unsupported tool name, or invalid argument shape: return a readable error message and do not execute a tool.
- `.env` remains ignored; API keys never enter source code, Git history, or GitHub.

## Testing

Unit tests inject a fake client and cover a valid tool call, a direct model answer, an unknown tool, malformed JSON arguments, and missing configuration. One manual command-line test with a real key verifies the complete DeepSeek loop.

## Out of Scope

- Multiple or parallel tool calls.
- Persistent chat history, retries, usage/cost reporting, and streaming.
- Connecting the Week05 Memory API.
