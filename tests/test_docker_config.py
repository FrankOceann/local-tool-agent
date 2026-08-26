from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_uses_expected_runtime_contract():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.11-slim" in dockerfile
    assert "WORKDIR /app" in dockerfile
    assert "COPY requirements.txt ./" in dockerfile
    assert "pip install --no-cache-dir -r requirements.txt" in dockerfile
    assert "COPY app ./app" in dockerfile
    assert "COPY tools ./tools" in dockerfile
    assert "COPY data ./data" in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert (
        'CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]'
        in dockerfile
    )
    assert ".env" not in dockerfile


def test_dockerignore_excludes_secrets_and_local_artifacts():
    ignored_paths = (PROJECT_ROOT / ".dockerignore").read_text(
        encoding="utf-8"
    ).splitlines()

    for required_path in (
        ".env",
        ".venv/",
        "__pycache__/",
        ".pytest_cache/",
        ".git/",
        ".worktrees/",
        "tests/",
        "docs/",
        "*.bak",
    ):
        assert required_path in ignored_paths