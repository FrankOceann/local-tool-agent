import re

def upper_text(text: str) -> str:
    return text.upper()


def count_words(text: str) -> int:
    return len(text.split())

def summarize_text(text: str) -> str:
    sentences = re.findall(r"[^。！？.!?]+[。！？.!?]?", text)
    return "".join(sentences[:2]) or text

TOOL_PERMISSIONS = {
    "upper_text": "auto",
    "count_words": "auto",
    "summarize_text": "auto",
    "save_note": "confirmation_required",
}


def save_note(text: str) -> str:
    return f"已模拟保存笔记：{text}"