from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_eng_lab.config import get_settings


@pytest.fixture(autouse=True)
def clean_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEBUG", raising=False)
    get_settings.cache_clear()
    return tmp_path


def test_settings_read_from_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("DEBUG", "true")
    settings = get_settings()
    assert settings.debug is True
    assert settings.llm_model == "deepseek-chat"
    assert settings.llm_api_key.get_secret_value() == "test-key"


def test_settings_read_from_env_file(clean_env: Path) -> None:
    (clean_env / ".env").write_text("LLM_API_KEY=from-file\nLLM_TIMEOUT_S=5\n")
    settings = get_settings()
    assert settings.llm_api_key.get_secret_value() == "from-file"
    assert settings.llm_timeout_s == 5.0


def test_api_key_is_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    assert "test-key" not in repr(get_settings())


def test_missing_api_key_raises() -> None:
    with pytest.raises(ValidationError):
        get_settings()


def test_settings_are_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    assert get_settings() is get_settings()
