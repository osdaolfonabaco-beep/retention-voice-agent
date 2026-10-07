# Tolvane retention auditor

Portfolio project. An auditor reads a retention conversation and flags failures against `docs/policy.md`. Later, an agent is graded by that auditor. Tolvane is a fictional company and all data is synthetic.

## Working rules

- Present a short plan and wait for approval before creating or changing any file.
- Change only what the task asks. Do not change scope or design on your own; propose it instead.
- Never read, print, or log `.env` or any `data/*/scenarios.sealed` file. Never print API keys or decoded scenario cards.
- Do not run git commands. The owner commits.
- Do not call the model API unless the task says so. Tests run offline with a fake client.
- Keep answers short: what changed, how to run it, what to verify.

## Stack

- Python 3.13, virtual environment in `.venv`.
- Dependencies in `requirements.txt`. Ask before adding one.
- Tests with pytest in `tests/`. Run: `.venv/bin/python -m pytest -q`.
- Model access through an OpenAI-compatible endpoint. Config comes from `LLM_BASE_URL` and `LLM_API_KEY` in `.env`.
- The auditor's model settings come from `AUDITOR_BASE_URL`, `AUDITOR_API_KEY` and `AUDITOR_MODEL` in `.env`.
- Code, comments, and docs in English.

## Layout

- `docs/policy.md`: the retention policy. Its rule IDs (VR, FL, OF, PP, PR, ES, CL, TN) are the source of truth.
- `retention/`: Python package with the project code.
- `tests/`: unit tests.
- `data/`: synthetic conversations and labels.
