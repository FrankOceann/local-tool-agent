from collections.abc import Callable

from tools.note_tools import save_note
from tools.text_tools import count_words, summarize_text, upper_text


TOOL_REGISTRY: dict[str, Callable[[str], str | int]] = {
    "upper_text": upper_text,
    "count_words": count_words,
    "summarize_text": summarize_text,
    "save_note": save_note,
}

TOOL_PERMISSIONS = {
    "upper_text": "auto",
    "count_words": "auto",
    "summarize_text": "auto",
    "save_note": "confirmation_required",
}