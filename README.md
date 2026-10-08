# Retention QA auditor and agent, in text and voice

A portfolio project in three parts, all measured:

1. **An auditor** that reads a customer-retention conversation and flags failures against a written policy.
2. **An agent with tools** that handles retention contacts and is graded by that auditor, plus two deterministic checks.
3. **A voice layer** over the same agent, with latency per turn, interruption handling and escalation measured.

The company (Tolvane, a home internet provider), the policy, the accounts and every conversation are fictional and synthetic. Nothing here comes from a real employer: no recordings, policies, scripts, amounts or system names.

## Results in one page

**Auditor**, on 50 held-out conversations it was never tuned on (52 planted failures, policy v0.6):

| Measure | Result |
|---|---|
| Failures found (recall) | 52 of 52 (95% interval 93% to 100%) |
| Clean verdicts left alone (specificity) | 238 of 248 (96%) |
| Flags that were real (precision) | 52 of 62 (84%) |
| Conversations with all six verdicts right | 40 of 50 |

**Agent**, same model in every run (`gpt-6-luna`, reasoning off), 24 scenarios:

| Run | What changed | Auditor: no failure | Tool use: no violation | Right outcome in the system | Good on all three |
|---|---|---|---|---|---|
| `agent-v1` | baseline prompt, no guards | 9 | 8 | 13 | 2 (8.3%) |
| `agent-v1g` | two guards in code, same prompt | 13 | 24 | 23 | 13 (54.2%) |
| `agent-v2` | guards and a longer prompt | 16 | 24 | 20 | 14 (58.3%) |
| `agent-v3` | tools that answer before the customer's yes | 20 | 24 | 23 | 19 (79.2%) |

That is plus 71 percentage points without changing the model. Most of it came from code around the model, not from the prompt.

**Voice** (speech-to-text, the same agent, text-to-speech), 24 calls, 165 agent turns in the last run:

| Measure | `voice-v1` | `voice-v2` |
|---|---|---|
| Speech to text, median / 95th percentile | 0.65 s / 1.41 s | 0.57 s / 1.40 s |
| Model with tools | 1.94 s / 4.92 s | 1.77 s / 4.18 s |
| Text to speech, to first audio | 0.83 s / 1.42 s | 0.73 s / 1.64 s |
| **Customer stops talking to agent audio starts** | **3.66 s / 6.89 s** | **3.28 s / 6.33 s** |
| Simulated interruptions with no account change after | 13 of 13 | 18 of 18 |
| Required escalations made, to the right team | 8 of 8 | 8 of 8 |
| Account holders locked out by a misheard surname | 5 of 19 | 0 of 19 |
| Right outcome in the system | 17 of 24 | 22 of 24 |
| Good on all three measures | 12 (50%) | 16 (66.7%) |

`voice-v2` was audited with policy v0.7, which contains the voice verification rule; audited with v0.6 it would show 11 good contacts, for the reason given in finding 7.

A 3.3-second median is too slow for a natural phone call. It is reported as measured; the section on limits says where the time goes.

## How the evaluation works

One measure was not enough, and the project shows why twice.

- **What the agent says** is judged by the auditor, a model that reads the transcript against the policy and returns pass or fail in six categories: false resolution, missed escalation, unfulfillable promise, aggravating tone, premature closure, privacy breach. Every finding cites a rule and the turns.
- **What the agent does with its tools** is checked by fixed rules over the tool-call log, with no model: a call made with data the customer never gave, or a contact ended right after asking a question.
- **What ends up done in the system** is compared with the expected outcome of each scenario, also without a model: for example, a plan must be set in one scenario and must not be set in another.

A contact counts as good only when it passes all three.

A second reader from a different model family (Claude) audited every agent run blind, with the policy and the transcripts and nothing else. It agreed with the auditor on 94% to 96% of the 144 verdicts in each run, and its count of clean contacts rose the same way: 6, 12, 14, 18.

## What I found

**1. The first agent looked fine in two test calls and was broken in 16 of 24.**
With a policy in the prompt and well-described tools, the agent asked for the customer's name and, in the same turn, called the verification tool with empty fields or an invented person. The tool counted two failed attempts and locked the account, and the agent then turned away the real customer. In one contact it took a 35-dollar payment nobody had named.

**2. The auditor could not see three of those contacts.**
The agent had ended the contact through a tool, without a goodbye. The transcript just stopped, and a transcript reader marks that clean. For an agent with tools, auditing the words is not enough. That is why the tool-log check exists.

**3. Two guards in code did what the instructions had not.**
*Ask-then-wait*: if a message asks the customer a question, the turn ends there and any tool call in it is discarded. *Grounding*: the verification and payment tools refuse a name, address, PIN or amount that does not appear in what the customer said, and count no failed attempt. The baseline prompt already said "wait for the answer" and "never invent". The guards moved the result from 2 to 13 good contacts.

**4. A better prompt fixed the form and broke the substance.**
Prompt v2 fixed closings and escalations. It also made the agent say "no plan fits" on two accounts where a plan did fit. The auditor did not flag either one, because no rule says "offer the plan that fits" and it does not do arithmetic. Looking only at the auditor, v2 was a clean gain from 13 to 16. The outcome check showed four contacts ending wrong where one had before.

**5. The cause was a tool, not the model.**
The payment-plan tool checked the customer's yes before it checked the limits, so the agent had no way to ask "does this plan fit?" and did the sum itself. Once the tool answered that question before asking for the yes, the wrong refusals disappeared and the result went to 19 of 24.

**6. Voice broke verification in a way the average error rate hides.**
Speech-to-text got about 5% of words wrong, which sounds fine. But it wrote "Oyeleron" for Oyelaran, "Ibarra" for Abara and "Wroble" for Wrobel. Five of 19 account holders gave every detail correctly and were refused, because the name was compared letter by letter. The fix is a rule decided by the project owner from call-centre experience: a name that sounds like the one on file is accepted only after one more proof question, the ZIP code, asked after the name was given. The code enforces that the question is really asked. In the next run it was asked in five calls and all five customers were verified.

**7. When the rule changed, the policy document had to change too.**
The auditor, still reading policy v0.6, flagged those five correct verifications as privacy breaches. By the written policy it was right. The rule went into policy v0.7 as VR-7, and the same calls were audited again. A regression check on the 50 tuning conversations gave the same auditor numbers within one verdict (52 of 53 failures found before and after; 6 and 7 false alarms).

## Limits

- **Everything is synthetic.** The conversations were written by a model from scenario cards with planted failures, and the simulated customers are models too. Real customers are messier.
- **Small numbers.** 24 scenarios and one run per version. A difference of two or three contacts can be noise. The gain from the guards is far outside that; the gain from prompt v2 is not.
- **The improvements were designed while looking at the same 24 scenarios.** There is no held-out scenario set for the agent, so the gains will look better here than on new cases.
- **The answer key for the auditor was written and checked by Claude**, with a blind second reader, not by a human quality analyst.
- **Agent and auditor are the same model.** A model can be lenient with its own mistakes. The blind reader from another family is there to offset that, and its numbers are reported next to the auditor's.
- **The auditor makes about one false alarm in six flags** and missed the wrong plan refusals described above.
- **Latency.** The median of 3.3 s splits into roughly 0.6 s of speech-to-text, 1.8 s of model and 0.7 s to first audio. Turns with one model call take about 1.3 s of model time; turns that also call a tool take about 3 s. Only 44% of agent turns start within 3 seconds.
- **The interruption is simulated.** The agent's text is cut where the audio would have been cut, and the customer's line is fixed. There is no microphone detecting speech.
- **The voices are clean synthetic speech**, with no line noise or accents, so real transcription errors will be higher. One holder was still locked out in the last run by a misheard street name.
- **The simulated customer sometimes hangs up on a question** (0 to 3 contacts per run), which can cost the agent a closing step that was not its fault.
- **One known gap in the tools:** in one scenario the agent moved an agreed installment a week later although the account showed no extra time available, and the tool allowed it because the date was inside the stage limits.

## What I would do next

- Move the agent to an API path that allows reasoning together with tools, and measure the trade against latency.
- Stream the model's reply into text-to-speech sentence by sentence to cut the time to first audio.
- Extend the close-match rule to street names, with the same proof question.
- Write a second scenario set that nobody looks at until the final measurement.
- Run the voice layer on recorded speech with noise, and add a live mode with a microphone.

## What is in the repository

| Path | What it is |
|---|---|
| `docs/policy.md` | The retention policy (v0.7, 58 rules). Rule IDs are the source of truth. |
| `retention/audit.py`, `evaluate.py`, `report.py` | The auditor, its scoring against the answer key, and a summary of an audit run. Only `evaluate.py` reads labels. |
| `retention/tools.py` | The agent's tools over one synthetic account, with the grounding guard and the close-name rule. |
| `retention/agent.py` | The agent turn loop with the ask-then-wait guard, and the runner against a simulated customer. |
| `retention/trace.py` | The tool-log check, the escalation check and the outcome check. |
| `retention/voice.py` | The voice runner with simulated interruption, and the latency report. |
| `prompts/` | The auditor prompt, the agent prompts (baseline, v2, v3, v3 for voice) and the simulated customer prompt. |
| `data/batch2`, `data/batch3`, `data/dev_split.json` | 100 conversations with an answer key of 105 failures, split 50/50 into a tuning half and a held-out half. |
| `data/agent/` | The 24 agent scenarios and their expected escalations and outcomes. |
| `data/agent_runs/`, `data/audits/` | Every run reported above, with transcripts, tool calls, audits, traces and the blind reader's findings. `agent-v0-trial` is the first failed trial, kept as evidence. |
| `tests/` | 593 unit tests. They run offline with fake clients. |

## Run it

Requires Python 3.13 and an OpenAI-compatible API key.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # then fill in the keys and model names
.venv/bin/python -m pytest -q

# auditor on the held-out half, then its score
.venv/bin/python -m retention.audit --run my-check --split check
.venv/bin/python -m retention.evaluate --run my-check --split check

# agent on the 24 scenarios, audited and checked
.venv/bin/python -m retention.agent --run my-agent --agent-prompt prompts/agent_system_v3.md
.venv/bin/python -m retention.audit --run my-agent-audit --conversations data/agent_runs/my-agent/conversations.jsonl
.venv/bin/python -m retention.trace --run my-agent --audit my-agent-audit \
    --expect data/agent/expected_escalations.json --outcomes data/agent/expected_outcomes.json

# voice
.venv/bin/python -m retention.voice --run my-voice --agent-prompt prompts/agent_system_v3_voice.md
.venv/bin/python -m retention.voice --report my-voice
```

Every run records its models, prompt and policy hashes and settings in `run.json`, and refuses to resume with different ones. Runs have a call budget, and voice runs also have an audio budget.

Models used: `gpt-6-luna` for the auditor and the agent, `gpt-5.6-luna` for the simulated customer, `gpt-4o-mini-transcribe` and `gpt-4o-mini-tts` for voice. The whole project used less than 3 US dollars of API credit.

## How it was built

I directed the project: the scope, each go or no-go, and the policy rule decisions, which draw on several years of work in retention, collections and customer verification by phone. Claude (Anthropic) acted as architect and reviewer: it proposed designs, wrote the policy text, the scenarios and the prompts, read every run and reported what was measured and what was its own judgement. Claude Code wrote the Python code and the tests from task descriptions, with a plan approved before each change. I ran every command and made every commit.

Oscar David Olarte Forero · [LinkedIn](https://www.linkedin.com/in/oscardavidolarteforero)
