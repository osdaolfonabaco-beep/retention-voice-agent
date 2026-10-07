"""Auditor entry point. For now only a connectivity check and a model listing."""

from __future__ import annotations

import argparse
import sys

from retention import config
from retention.llm import ChatClient, LLMError

PING_MESSAGE = "Reply with OK"


def _parse_args(argv):
    p = argparse.ArgumentParser(prog="python -m retention.audit")
    p.add_argument("--ping", action="store_true", help="one call to AUDITOR_MODEL")
    p.add_argument("--list-models", action="store_true", help="list model ids")
    p.add_argument("--contains", metavar="TEXT",
                   help="with --list-models: only ids containing TEXT (case-insensitive)")
    args = p.parse_args(argv)
    if args.ping == args.list_models or (args.contains is not None and not args.list_models):
        p.print_usage()
        return p, None
    return p, args


def ping(client, model: str) -> int:
    try:
        client.complete(model, [{"role": "user", "content": PING_MESSAGE}])
    except LLMError as exc:
        status = f" {exc.status}" if exc.status else ""
        detail = " ".join(exc.detail.split())[:200]
        print(f"{model}: error {exc.kind}{status} {detail}".rstrip())
        return 1
    print(f"{model}: ok")
    return 0


def list_models(client, contains: str | None) -> int:
    try:
        ids = client.list_models()
    except LLMError as exc:
        print(f"error: {exc}")
        return 1
    if contains is not None:
        ids = [m for m in ids if contains.lower() in m.lower()]
    for model_id in sorted(ids):
        print(model_id)
    return 0


def main(argv=None, client=None, env_path=None) -> int:
    _, args = _parse_args(argv)
    if args is None:
        return 2
    try:
        base_url, api_key, model = config.load_auditor_settings(env_path)
    except config.ConfigError as exc:
        raise SystemExit(f"error: {exc}") from None
    client = client or ChatClient(base_url, api_key)
    if args.ping:
        return ping(client, model)
    return list_models(client, args.contains)


if __name__ == "__main__":
    sys.exit(main())
