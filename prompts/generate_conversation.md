You write synthetic customer-service transcripts for an evaluation dataset.

The company is Tolvane, a fictional home internet provider in the United States. Its retention policy is at the end of this prompt. Each transcript is one contact between a Tier 1 retention agent and a person who contacts Tolvane.

## Your input

A JSON object with these fields:

- `channel`: "phone" or "chat".
- `account`: what the agent's screen shows for this account. Treat it as true. Use its names, amounts, dates and flags exactly.
- `brief`: what happens in this contact. Follow it exactly.
- `min_turns`, `max_turns`: the transcript must have between these many turns, inclusive.

## How to write the agent

- The brief lists what the agent gets wrong, if anything. The agent makes those mistakes and no others. Wherever the brief is silent, the agent follows the policy correctly.
- If the brief says the agent follows the policy fully, the agent makes no mistake at all.
- Unless the brief says otherwise, the agent does all of this, briefly and naturally: verifies identity before discussing the account (full name, service address, and the last payment amount or the PIN); asks for the reason and acknowledges it before offering anything; offers only what the account allows; gets a clear yes before any change; recaps what was done with amounts and dates; asks if there are questions; asks the person to confirm the recap; and ends the contact by thanking the person and saying goodbye.
- A mistake marked "subtle" must be easy to miss. The agent stays polite and competent, and nothing in the dialogue points at the mistake.
- A mistake marked "obvious" is clear to a careful reader, but it is still something a real agent would do on a bad day. No cartoon villains.
- Nobody in the transcript notices, names or comments on the agent's mistake, unless the brief says the customer reacts.
- The agent never cites rule numbers or codes such as "VR-1".
- The agent does not know who is contacting them until that person says their name. The agent never uses the person's name before the person has given it, and always asks for the full name as part of verification.
- The agent offers and promises only what the policy contains. Wherever the brief is silent, the agent does not invent options, exceptions or procedures.
- The agent talks like a person, not like a policy document. The agent never says internal labels to the customer: no "Tier 1", no field names from the account screen, and no stage names. Never say "early stage", "mid stage", "late stage", "this stage" or "your account stage". The agent says the same thing in plain words, for example "your September bill is 18 days past due". The Recovery team may be named. The only exception is a brief that asks for jargon.
- The agent does not open every turn with a stock empathy phrase. One sincere acknowledgement is worth more than five "I completely understand".

## How to write the person contacting Tolvane

- An ordinary adult in the United States, with a consistent personality and way of speaking. Real people ramble, interrupt, repeat themselves and do not always answer the question they were asked.
- They know only what a customer would know. They do not know the policy.
- When the person is the account holder, their verification answers match the account, unless the brief says otherwise.
- They give information when the agent asks for it. They do not volunteer every detail in their first message.
- They never talk like a form and never repeat the agent's wording. A clear yes sounds like "yeah, that works", "okay, let's do that" or "sure, go ahead". It never sounds like "I agree to that plan" or "the recap is correct".
- Give them something human: a short reaction ("okay", "mm-hm", "wait, so..."), a worry, a question that is slightly off topic.

## Style

- Phone: spoken English, contractions, natural back-and-forth. Chat: typed messages, a little terser; an occasional typo or lowercase message is fine.
- Speakers alternate strictly: agent, customer, agent, customer. Never two turns in a row from the same speaker. The first turn is the agent's greeting. The last turn is the agent's goodbye, unless the brief says the contact ends another way.
- Vary how the contact opens. Most agents greet with their first name, ask how they can help, and verify identity once the person has said what they need. Use the agent's first name given in the brief.
- Keep turns short. On the phone no agent turn runs past about 60 words. When the agent has a lot to explain (a payment plan, the consequences, a recap), split it across several turns and let the other person react in between. Use the room that `max_turns` gives you.
- No stage directions, no brackets, no narration. Hold time or silence shows only in what people say.
- Dates are relative to `account.today`. Say dates the way people do ("the 15th", "October 20th").
- Weekdays: `account.today_weekday` is the weekday of `account.today`. Work out any other weekday from it, or leave the weekday out and say only the date. Never guess a weekday.
- All names, addresses and numbers are fictional. Do not mention real companies or real people. No card numbers or other payment details are spoken.
- Count the turns before you answer. Never go over `max_turns` or under `min_turns`. If the scene does not fit, tighten the wording; do not drop any step the brief asks for.

## Output

Return only a JSON object, with no text before or after it:

{"turns": [{"speaker": "agent", "text": "..."}, {"speaker": "customer", "text": "..."}]}

`speaker` is "agent" or "customer". When a third party contacts Tolvane, that person is still "customer".

## Tolvane retention policy

{policy}
