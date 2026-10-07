# Tolvane retention auditor

Portfolio project. An auditor reads a retention conversation and flags failures against `docs/policy.md`. Later, an agent is graded by that auditor. Tolvane is a fictional company and all data is synthetic.

## Working rules

- Present a short plan and wait for approval before creating or changing any file.
- Change only what the task asks. Do not change scope or design on your own; propose it instead.
- Never read, print, or log `.env` or any `data/*/scenarios.sealed` file. Never print API keys or decoded scenario cards.
- Do not run git commands. The owner commits.
- Do not call the model API unless the task says so. Tests run offline with a fake client.
- Only `retention/evaluate.py` reads labels (`labels_key.jsonl`, `second_reader.jsonl`). The auditor (`retention/audit.py`) never sees them.
- Keep answers short: what changed, how to run it, what to verify.

## Stack

- Python 3.13, virtual environment in `.venv`.
- Dependencies in `requirements.txt`. Ask before adding one.
- Tests with pytest in `tests/`. Run: `.venv/bin/python -m pytest -q`.
- Model access through an OpenAI-compatible endpoint. Config comes from `LLM_BASE_URL` and `LLM_API_KEY` in `.env`.
- The auditor's model settings come from `AUDITOR_BASE_URL`, `AUDITOR_API_KEY` and `AUDITOR_MODEL` in `.env`.
- Agent model calls send `reasoning_effort: "none"` (`AGENT_REASONING_EFFORT` in `retention/agent.py`), because gpt-6-luna rejects function tools with reasoning on `/v1/chat/completions`; it is recorded in `run.json`.
- The agent's opening turn of a contact is called without tools (`allow_tools=False`), because no tool can be needed before the customer speaks and a trial showed it inventing `verify_identity` values and locking the account.
- Two guards are on by default and are turned off together by `--no-guards` (`run.json` records `"guards"`). Ask-then-wait (`agent_turn`, `guards=True`): when a model message has text with "?" and tool calls, the turn ends with that text, the calls are not run or stored in the messages, and each is logged in the session's calls with result `{"blocked": "asked_a_question"}`. Grounding (`AccountSession(grounding=True)`, `ungrounded()` in `retention/tools.py`): the runner calls `session.hear(text)` with every customer turn; `verify_identity` and `take_payment` run before argument validation and refuse values the customer has not said (`not_said_by_customer`, listing parameter names in `missing`, no failed attempt counted, nothing compared with the account). A value is grounded when it appears in the customer's text so far: names, PINs and the address part before the first comma as lowercase alphanumerics; amounts among the numbers written with digits.
- `--agent-prompt PATH` picks the agent prompt file (default `prompts/agent_system.md`; relative paths resolve against the project root). `run.json` records `"agent_prompt"` as given. A `run.json` without `guards` and `agent_prompt` (such as `agent-v1`) counts as a mismatch, so that run cannot be resumed.
- The agent and the simulated customer use `AGENT_MODEL` and `CUSTOMER_MODEL` (required) in `.env`. `AGENT_BASE_URL` and `AGENT_API_KEY` are optional and default to the `AUDITOR_` ones.
- Code, comments, and docs in English.

## Layout

- `docs/policy.md`: the retention policy. Its rule IDs (VR, FL, OF, PP, PR, ES, CL, TN) are the source of truth.
- `retention/`: Python package with the project code.
- `prompts/audit_conversation.md`: the auditor's system prompt template (one placeholder, `{policy}`). Written by the owner; do not edit.
- `prompts/agent_system.md`, `prompts/customer_simulator.md`: the agent's and the simulated customer's prompt templates. Written by the owner; do not edit.
- `retention/tools.py`: the agent's tools over one synthetic account (`AccountSession`, `TOOL_SPECS`).
- `retention/agent.py`: the agent turn loop and the runner (`python -m retention.agent --run NAME`).
- `retention/report.py`: summary of an audit run (`python -m retention.report --run NAME`). Never reads labels.
- `retention/trace.py`: trace check of an agent run (`python -m retention.trace --run NAME [--audit AUDIT_RUN]`). Reads only `conversations.jsonl` and `tool_calls.jsonl` of the run, plus `audits.jsonl` of AUDIT_RUN; never labels. Violations: `ungrounded_call` (a processed `verify_identity` or `take_payment` call, error results included, with a value the customer had not said) and `ended_on_question` (a processed `end_contact` in an agent turn that contains "?"). Prints one line per contact, the totals and the calls blocked by each guard, and writes `trace.json`. With `--audit` it also lists the contacts with no failed category and no violation; it exits if the id sets differ.
- `tests/`: unit tests.
- `data/`: synthetic conversations and labels.
- `data/agent/scenarios.jsonl`: scenarios for the agent runs. `expected` is documentation for the owner and is never sent to a model.
- `data/agent_runs/`: agent runs, one folder per run (`conversations.jsonl`, `tool_calls.jsonl`, `log.jsonl`, `run.json`, `trace.json` from the trace check).
- `data/audits/`: audit runs, one folder per run (`audits.jsonl`, `log.jsonl`, `run.json`, `summary.json` from the report, and the evaluator's metrics and disagreements). `retention/audit.py` takes `--split` or `--conversations PATH`.
