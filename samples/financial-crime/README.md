# Financial crime / AML (synthetic sample)

Label alert typology, decide whether an investigator should look, score risk.

**Every row here is synthetic.** Names, companies, policy numbers, amounts and texts are made up. Nothing comes from a real person, patient or customer.

- Rows: 33 train, 7 test (test scenarios are held out from train)
- Questions: typology, escalate, risk
- Source for the use case: TypeSafe's own use-case map lists "Financial crime" ([docs.typesafe.ai](https://docs.typesafe.ai/concepts/use-case-map))

```bash
python validate_data.py samples/financial-crime/*.jsonl
python train_laya.py --category financial-crime --mode lora --max-steps 60
```

This set is small. Use it to check the format and run a smoke test, then swap in a few hundred of your own labelled rows. Forty rows can't fit calibration temperatures you should trust.

Example question block:

```json
{
 "typology": {
  "type": "choice",
  "instructions": "Which typology best fits this alert?",
  "criteria": {
   "structuring": "many deposits just under reporting thresholds",
   "mule": "funds in then quickly out to unrelated parties",
   "fraud": "account takeover, scam or unauthorized payment",
   "benign": "explained, expected customer activity"
  }
 },
 "escalate": {
  "type": "noul",
  "instructions": "Should an investigator review this alert?"
 },
 "risk": {
  "type": "score",
  "instructions": "Overall risk level",
  "criteria": [
   "low",
   "medium",
   "high"
  ]
 }
}
```
