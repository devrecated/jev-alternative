# Customer support (synthetic sample)

Route tickets to a team, flag refund requests, score urgency.

**Every row here is synthetic.** Names, companies, policy numbers, amounts and texts are made up. Nothing comes from a real person, patient or customer.

- Rows: 32 train, 10 test (test scenarios are held out from train)
- Questions: team (choice), refund (noul), urgency (score)
- Source for the use case: TypeSafe's own use-case map lists "Customer support" ([docs.typesafe.ai](https://docs.typesafe.ai/concepts/use-case-map))

```bash
python validate_data.py samples/customer-support/*.jsonl
python train_laya.py --category customer-support --mode lora --max-steps 60
```

This set is small. Use it to check the format and run a smoke test, then swap in a few hundred of your own labelled rows. Forty rows can't fit calibration temperatures you should trust.

Example question block:

```json
{
 "team": {
  "type": "choice",
  "instructions": "Which team should handle this?",
  "criteria": {
   "billing": "invoices, payments, refunds, charges",
   "technical": "bugs, outages, errors, integrations",
   "sales": "pricing, new plans, upgrades, quotes",
   "account": "login, access, users, settings"
  }
 },
 "refund": {
  "type": "noul",
  "instructions": "Does the customer ask for a refund or credit?"
 },
 "urgency": {
  "type": "score",
  "instructions": "How urgent is this?",
  "criteria": [
   "not urgent",
   "soon",
   "today / blocking"
  ]
 }
}
```
