"""Read the LLM endpoint settings from a .env file (no extra dependency)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = ("LLM_BASE_URL", "LLM_API_KEY")
AUDITOR_REQUIRED = ("AUDITOR_BASE_URL", "AUDITOR_API_KEY", "AUDITOR_MODEL")
AGENT_REQUIRED = ("AGENT_MODEL", "CUSTOMER_MODEL")


class ConfigError(Exception):
    pass


def parse_env(text: str) -> dict[str, str]:
    """Parse KEY=VALUE lines. Blank lines and # comments are ignored."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_settings(path: Path | None = None) -> tuple[str, str]:
    """Return (base_url, api_key). Errors name the variable, never its value."""
    path = Path(path) if path else ROOT / ".env"
    if not path.is_file():
        raise ConfigError(f"{path.name} not found; copy .env.example and fill it in")
    values = parse_env(path.read_text(encoding="utf-8"))
    missing = [name for name in REQUIRED if not values.get(name)]
    if missing:
        raise ConfigError(f"missing in {path.name}: {', '.join(missing)}")
    return values["LLM_BASE_URL"], values["LLM_API_KEY"]


def load_auditor_settings(path: Path | None = None) -> tuple[str, str, str]:
    """Return (base_url, api_key, model). Errors name the variable, never its value."""
    path = Path(path) if path else ROOT / ".env"
    if not path.is_file():
        raise ConfigError(f"{path.name} not found; copy .env.example and fill it in")
    values = parse_env(path.read_text(encoding="utf-8"))
    missing = [name for name in AUDITOR_REQUIRED if not values.get(name)]
    if missing:
        raise ConfigError(f"missing in {path.name}: {', '.join(missing)}")
    return values["AUDITOR_BASE_URL"], values["AUDITOR_API_KEY"], values["AUDITOR_MODEL"]


def load_agent_settings(path: Path | None = None) -> tuple[str, str, str, str]:
    """Return (base_url, api_key, agent_model, customer_model).

    AGENT_BASE_URL and AGENT_API_KEY fall back to the AUDITOR_ ones. Errors name
    the variable, never its value.
    """
    path = Path(path) if path else ROOT / ".env"
    if not path.is_file():
        raise ConfigError(f"{path.name} not found; copy .env.example and fill it in")
    values = parse_env(path.read_text(encoding="utf-8"))
    missing = [name for name in AGENT_REQUIRED if not values.get(name)]
    base_url = values.get("AGENT_BASE_URL") or values.get("AUDITOR_BASE_URL")
    api_key = values.get("AGENT_API_KEY") or values.get("AUDITOR_API_KEY")
    if not base_url:
        missing.append("AGENT_BASE_URL (or AUDITOR_BASE_URL)")
    if not api_key:
        missing.append("AGENT_API_KEY (or AUDITOR_API_KEY)")
    if missing:
        raise ConfigError(f"missing in {path.name}: {', '.join(missing)}")
    return base_url, api_key, values["AGENT_MODEL"], values["CUSTOMER_MODEL"]
