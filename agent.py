from tools import count_words, upper_text


class LocalToolAgent:
    def run(self, request: str) -> str:
        if request.startswith("大写 "):
            text = request.removeprefix("大写 ").strip()
            return upper_text(text)
        if request.startswith("统计单词 "):
            text = request.removeprefix("统计单词 ").strip()
            return f"英文单词数量: {count_words(text)}"
        
        return "暂不支持这个请求。可使用：大写 <文字> 或 统计单词 <英文文字>"