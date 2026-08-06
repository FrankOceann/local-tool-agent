# Week06 Local Tool Agent Design

## Goal

Build a small command-line agent that receives a user's text, selects a local tool, runs it, and returns the result. This first version does not call an LLM or require an API key.

## Components

- `main.py`: reads user input and prints the agent result.
- `agent.py`: selects a tool using simple keyword rules and calls it.
- `tools.py`: contains the tool implementations.
- `test_main.py`: checks tool behavior and agent tool selection.

## Tools

- `upper_text(text)`: returns `text.upper()`.
- `count_words(text)`: returns the number of English words in `text`.

## Flow

1. The user enters a request.
2. `main.py` passes the request to `agent.py`.
3. `agent.py` selects a tool from the request keywords.
4. The selected function in `tools.py` runs.
5. `main.py` prints the result.

## Errors

If no supported tool matches the request, the agent returns a clear message instead of guessing.

## Tests

Tests will cover both tool functions, tool selection, and the unsupported-request path.

## Next Phase

The local rule in `agent.py` will later be replaced with an LLM Tool Calling request. The tool function interface remains the same.
