from pathlib import Path


DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data"


def read_file(filename: str) -> str:
    allowed_directory = DATA_DIRECTORY.resolve()
    file_path = (allowed_directory / filename).resolve()

    if not file_path.is_relative_to(allowed_directory):
        return "不允许读取 data 目录外的文件。"

    if not file_path.exists():
        return "文件不存在。"

    return file_path.read_text(encoding="utf-8")

def search_files(query: str) -> str:
    results = []

    for file_path in DATA_DIRECTORY.glob("*.txt"):
        content = file_path.read_text(encoding="utf-8")

        if query in content:
            results.append(f"{file_path.name}: {content}")

    if not results:
        return "没有找到匹配内容。"

    return "\n".join(results)
