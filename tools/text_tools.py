import re


def upper_text(text: str) -> str:
    return text.upper()


def count_words(text: str) -> int:
    return len(text.split())


def summarize_text(text: str) -> str:
    sentences = re.findall(r"[^。！？.!?]+[。！？.!?]?", text)
    return "".join(sentences[:2]) or text