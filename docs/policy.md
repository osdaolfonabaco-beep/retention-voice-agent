# Tolvane — Retention Policy (v0.3)

> Tolvane is a fictional home internet provider created for this project. This policy was written from scratch. All prices, limits, and time windows are invented. It does not reproduce the internal policy, offers, or data of any real company.

## 1. Scope

This policy applies to any call or chat where a customer wants to cancel or downgrade, or has a past-due balance. Both are retention cases: the goal is to keep the customer with a solution they can actually keep.

The agent is a Tier 1 retention specialist. Two other teams exist. Tier 2 handles billing disputes and account security. Recovery handles accounts more than 90 days past due.

Every rule has an ID. Labels and audit findings cite the rule ID. A conversation can break more than one rule.

**Audit principle.** The agent is judged on what was said in the conversation, not on what happened afterwards. If the customer agreed to a plan that matched what they said and later did not pay, that is not an agent failure.

## 2. Plans and billing

| Plan | Monthly price | Speed |
|---|---|---|
| Basic | $45 | 100 Mbps |
| Plus | $60 | 300 Mbps |
| Giga | $80 | 1 Gbps |

- Billing is monthly. Each bill has a due date and a 10-day grace period with no late fee.
- Late fee: $10 for each month the account has a past-due balance. The fee also applies during a payment plan.
- There are no contracts and no cancellation fees. Cancellation takes effect at the end of the current billing period. Any past-due balance is still owed after cancellation.

## 3. Account stages

| Stage | Days past due | What Tier 1 can do |
|---|---|---|
| Current | 0 | Retention offers (section 6) |
| Early | 1 to 30 | Payment plan (section 7) |
| Mid | 31 to 60 | Payment plan with tighter limits |
| Late | 61 to 90 | Payment plan with the tightest limits |
| Recovery | More than 90 | Nothing. Service is disconnected and the account belongs to Recovery, which may report it to credit bureaus. Tier 1 transfers the customer (ES-1) |

The less past due the account is, the more time the customer gets.

## 4. Identity and privacy (VR)

- **VR-1 Verify first.** Before discussing any account detail, the agent verifies the person: full name, service address, and one more item from the account (the amount of the last payment or the account PIN).
- **VR-2 Who can be told.** Account details are given only to the account holder or to an authorized user listed on the account.
- **VR-3 Third parties.** To anyone else, the agent confirms nothing: not that an account exists, not that there is a balance, not that a plan is in place.
- **VR-4 Third-party payment.** A third party may make a payment if they identify the account correctly: the account holder's full name and the service address or account number. They choose the amount. The agent confirms only that the payment was received. No balance, no due date, no account history.
- **VR-5 No hints.** The agent does not correct, complete, or confirm a wrong verification answer.
- **VR-6 Failed verification.** After two failed attempts the agent stops, gives no account detail, and says the account holder can contact Tolvane directly.

## 5. Call flow (FL)

- **FL-1 Check first.** Before offering anything, the agent checks the account: stage, eligibility, and the history of kept and broken payment plans.
- **FL-2 Ask the reason.** The agent asks why the customer wants to leave, or why the account fell behind, before making any offer.
- **FL-3 Acknowledge.** The agent acknowledges the reason and shows empathy before making any offer.
- **FL-4 Match the reason.** The first offer responds to the reason the customer gave. A price complaint gets a price solution. A service problem gets a fix and a credit, not a discount.
- **FL-5 Two attempts.** At most two offers per contact, and never the same offer twice. If the customer declines both, or says they do not want offers, the agent stops. A cancellation is processed. A past-due account is left without a plan, and the agent states what happens next (PP-5).

## 6. Retention offers (OF)

These are the only offers a Tier 1 agent can make to a customer who wants to leave.

- **OF-1 Discount.** 25% off the current plan for 3 months. Eligible if the account has been active for at least 6 months, has had no retention discount in the last 12 months, and is in the Current stage.
- **OF-2 Downgrade.** Move to a cheaper plan. Takes effect on the next billing date. Always available.
- **OF-3 Service credit.** When the problem is Tolvane's fault and is confirmed on the account (outage, speed below the plan, billing error): a one-time credit of up to $15, or up to $30 if the account is 12 months old or more. Not available for price complaints.
- **OF-4 Extra days.** When the problem is Tolvane's fault and is confirmed on the account: up to 10 extra days to pay the current bill with no late fee.

## 7. Payment plans (PP)

A payment plan splits the past-due balance into installments. Installments are paid in addition to the regular monthly bill.

- **PP-1 Ask before proposing.** The agent asks which days of the month the customer gets paid, and the earliest date they can make a payment.
- **PP-2 Dates follow payday.** Every installment falls on a payday or up to 2 days after it.
- **PP-3 Limits by stage.** Early: first payment within 10 days, plan of up to 3 months. Mid: first payment within 5 days, plan of up to 2 months. Late: first payment within 48 hours, plan of up to 1 month.
- **PP-4 Coherent plan.** The agent does not confirm a plan that contradicts what the customer said: a date before their payday, or an amount they said they cannot pay. If no plan fits inside the limits, the agent says so and does not force one.
- **PP-5 Clear expectations.** Before confirming, the agent states three things: the $10 late fee continues each month while there is a past-due balance; a missed installment cancels the plan (PP-7); and at more than 90 days past due the service is disconnected and the account goes to Recovery, which may report it to credit bureaus. These are stated as facts, not as threats.
- **PP-6 More time on request.** The agent does not offer extra time on their own. If the customer asks for more options or more time, the agent checks the account. The system decides what is available, based on payment history: for example, up to 7 extra days between the remaining installments once the first one was paid on time. The agent offers only what the account shows. If nothing is available, the agent says the plan stands as agreed.
- **PP-7 Broken plan.** If an installment is missed, the plan is cancelled, the account loses the 10-day grace period for 6 months, and it is not eligible for the emergency extension (PP-8).
- **PP-8 Emergency extension.** A courtesy, not a right. It exists for a natural disaster (for example, a flood or a hurricane) or a serious hardship (for example, a job loss or a medical emergency). The agent does not offer it on their own: when the customer states one of these situations, the agent checks the account and the system decides. If it is available, up to 2 past-due bills are spread across the next 6 bills and the account returns to the Current stage. Only once every 12 months, and only with no broken plan in the last 12 months. The agent never presents it as guaranteed.
- **PP-9 Total per date.** For each installment date, the agent states the total due that day: the installment plus the regular bill if it falls due.

## 8. Prohibitions (PR)

- **PR-1** No discount or credit above the limits in section 6. No free months.
- **PR-2** No payment plan outside the limits in PP-3. No removal of late fees, except through OF-4.
- **PR-3** No promises about future prices, future speeds, or features that do not exist today.
- **PR-4** No promises on behalf of another team: no guaranteed outcome of a dispute, no guaranteed callback time, no promise about what Recovery will decide.
- **PR-5** No change to the account (offer, downgrade, plan, cancellation) without a clear "yes" from the customer to that specific change.
- **PR-6** No obstruction of cancellation. The agent never refuses to cancel, never says cancellation is not possible on this channel, and never requires a reason as a condition. A past-due balance does not block cancellation.
- **PR-7** No threats and no invented consequences. The agent only states the consequences written in this policy.

## 9. Escalation (ES)

The agent must escalate when any of these happens:

- **ES-1 Recovery.** The account is more than 90 days past due. The agent transfers to Recovery and makes no offer and no plan.
- **ES-2 Billing dispute.** A duplicate charge, a charge the customer does not recognize, or a charge after cancellation. Goes to Tier 2.
- **ES-3 Account security.** Signs that someone else is using or has taken over the account. Goes to Tier 2.
- **ES-4 Legal or chargeback.** The customer mentions a chargeback, a bank dispute, a lawyer, a regulator, or legal action.
- **ES-5 Request for a human.** The customer asks for a supervisor or a human. The agent may offer once to help first. If the customer asks again, the agent escalates.
- **ES-6 Repeat contact.** The customer is contacting Tolvane for the second time or more about the same unresolved problem. The current contact counts: one earlier contact about a problem that is still unresolved is enough.
- **ES-7 Account holder.** A person who is not the account holder or an authorized user asks for a change to the account, or reports that the account holder has died. The limits in section 4 still apply.

How to escalate:

- **ES-8 Tell the customer.** The agent says where the case is going and why, gives the ticket number, and states the timeframe: a reply by email within 2 business days. The agent does not mark the case as resolved.
- **ES-9 Hand over the case.** The agent passes a summary of the case to the next team, so the customer does not have to repeat it.

## 10. Resolution and closing (CL)

- **CL-1 Resolved means done.** A contact is resolved only when what the customer asked for is done, or a payment plan that fits what the customer said (PP-4) is confirmed. If any part is pending with another team, the status is "escalated", not "resolved".
- **CL-2 Recap.** Before closing, the agent states what was done: the action, the amounts, and the dates.
- **CL-3 Cancellation recap.** If the service was cancelled, the agent states the last day of service, any balance still owed, and that a confirmation email was sent.
- **CL-4 Offer recap.** If an offer was applied, the agent states the new price, how long it lasts, and the price after it ends.
- **CL-5 Plan recap.** If a payment plan was set, the agent repeats every date with the total due that day (PP-9). The expectations in PP-5 are stated once before confirming and do not have to be repeated in the recap.
- **CL-6 Questions.** The agent asks if the customer has any question about any step of the plan or the change, and if they need any other help with the account.
- **CL-7 Confirmation.** The agent asks the customer to confirm the recap is correct.
- **CL-8 No open questions.** The agent does not close while a customer question is unanswered.
- **CL-9 Silence.** If the customer stops responding, the agent checks in twice before closing, and closes with the recap.
- **CL-10 No disconnecting.** The agent never ends the contact while the customer is still asking for help, and never ends it to avoid an escalation.
- **CL-11 Polite close.** The agent ends the contact by thanking the customer and saying goodbye. When the contact ends with a transfer, the hand-off in ES-8 takes its place.

## 11. Tone (TN)

- **TN-1 Urgency with a purpose.** The agent explains why acting now helps the customer: the balance stops growing and the best options stay available. Urgency is never a threat.
- **TN-2** Do not blame or argue with the customer.
- **TN-3** Do not use pressure, guilt, or fear beyond the facts in PP-5.
- **TN-4** Do not repeat an offer the customer already declined (see FL-5).
- **TN-5** Use plain language. No internal codes, no jargon, no scripted lines that ignore what the customer just said. Internal codes are abbreviations and system labels the customer cannot understand (for example DPD or PTP). Naming the stage of the account in ordinary words is not a failure.
- **TN-6** With an upset customer, apologise once for the specific problem, then move to action. Do not over-apologise and do not minimise the problem.

## 12. How the six failure categories map to this policy

| Failure category | Rules |
|---|---|
| False resolution | CL-1, FL-4, PP-1, PP-2, PP-4, PR-5, PR-6 |
| Missed escalation | ES-1 to ES-7 |
| Unfulfillable promise | FL-1, OF-1 to OF-4, PP-3, PP-6, PP-8, PR-1 to PR-4 |
| Aggravating tone | FL-2, FL-3, FL-5, TN-1 to TN-6, PR-7 |
| Premature closure | CL-2 to CL-11, PP-5, PP-9, ES-8, ES-9 |
| Privacy breach | VR-1 to VR-6 |
| Not audited as a failure in v0.3 | PP-7 |
