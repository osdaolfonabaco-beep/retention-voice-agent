"""Offline tests for the auditor settings loader and connectivity check."""

import pytest

from retention import audit, config
from retention.llm import LLMError

SECRETS = {
    "AUDITOR_BASE_URL": "https://secret-host.example/v1",
    "AUDITOR_API_KEY": "sk-secret-123",
    "AUDITOR_MODEL": "secret-model",
}


def write_env(tmp_path, **overrides):
    values = {**SECRETS, **overrides}
    lines = [f"{k}={v}" for k, v in values.items() if v is not None]
    path = tmp_path / ".env"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class FakeClient:
    def __init__(self, result="OK", models=()):
        self.result, self.models = result, list(models)
        self.calls = []

    def complete(self, model, messages):
        self.calls.append((model, messages))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def list_models(self):
        if isinstance(self.models, Exception):
            raise self.models
        return self.models


# ------------------------------------------------------------ settings loader

def test_loader_ok(tmp_path):
    path = write_env(tmp_path)
    assert config.load_auditor_settings(path) == tuple(SECRETS.values())


def test_loader_names_missing_variable_not_values(tmp_path):
    path = write_env(tmp_path, AUDITOR_MODEL=None)
    with pytest.raises(config.ConfigError) as err:
        config.load_auditor_settings(path)
    text = str(err.value)
    assert "AUDITOR_MODEL" in text
    for value in SECRETS.values():
        assert value not in text


def test_loader_empty_value_counts_as_missing(tmp_path):
    path = write_env(tmp_path, AUDITOR_API_KEY="")
    with pytest.raises(config.ConfigError, match="AUDITOR_API_KEY"):
        config.load_auditor_settings(path)


def test_loader_missing_file_error_has_no_values(tmp_path):
    with pytest.raises(config.ConfigError) as err:
        config.load_auditor_settings(tmp_path / ".env")
    for value in SECRETS.values():
        assert value not in str(err.value)


# ----------------------------------------------------------------------- ping

def test_ping_ok(tmp_path, capsys):
    client = FakeClient("OK")
    code = audit.main(["--ping"], client=client, env_path=write_env(tmp_path))
    assert code == 0
    assert capsys.readouterr().out == "secret-model: ok\n"
    assert client.calls == [
        ("secret-model", [{"role": "user", "content": "Reply with OK"}])
    ]


def test_ping_error(tmp_path, capsys):
    client = FakeClient(LLMError("BadRequestError", 400, "x" * 300))
    code = audit.main(["--ping"], client=client, env_path=write_env(tmp_path))
    assert code == 1
    out = capsys.readouterr().out
    assert out == f"secret-model: error BadRequestError 400 {'x' * 200}\n"


def test_ping_error_without_status(tmp_path, capsys):
    client = FakeClient(LLMError("APIConnectionError", None, "no\nroute"))
    code = audit.main(["--ping"], client=client, env_path=write_env(tmp_path))
    assert code == 1
    assert capsys.readouterr().out == "secret-model: error APIConnectionError no route\n"


# ---------------------------------------------------------------- list models

MODELS = ["zeta-1", "Alpha-Flash", "beta", "alpha-pro"]


def test_list_models_sorted(tmp_path, capsys):
    code = audit.main(["--list-models"], client=FakeClient(models=MODELS),
                      env_path=write_env(tmp_path))
    assert code == 0
    assert capsys.readouterr().out.splitlines() == sorted(MODELS)


def test_list_models_contains_case_insensitive(tmp_path, capsys):
    code = audit.main(["--list-models", "--contains", "ALPHA"],
                      client=FakeClient(models=MODELS), env_path=write_env(tmp_path))
    assert code == 0
    assert capsys.readouterr().out.splitlines() == ["Alpha-Flash", "alpha-pro"]


# ------------------------------------------------------------------ CLI usage

@pytest.mark.parametrize("argv", [[], ["--contains", "x"], ["--ping", "--list-models"]])
def test_no_valid_mode_exits_2(tmp_path, capsys, argv):
    client = FakeClient()
    code = audit.main(argv, client=client, env_path=write_env(tmp_path))
    assert code == 2
    assert "usage" in capsys.readouterr().out.lower()
    assert client.calls == []


def test_missing_settings_exit_message(tmp_path):
    path = write_env(tmp_path, AUDITOR_BASE_URL=None)
    with pytest.raises(SystemExit) as err:
        audit.main(["--ping"], client=FakeClient(), env_path=path)
    assert "AUDITOR_BASE_URL" in str(err.value)
    assert SECRETS["AUDITOR_API_KEY"] not in str(err.value)
