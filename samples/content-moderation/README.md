# Moderation and trust & safety (synthetic sample)

Name the broken policy, decide removal, score severity.

**Every row here is synthetic.** Names, companies, policy numbers, amounts and texts are made up. Nothing comes from a real person, patient or customer.

- Rows: 30 train, 10 test (test scenarios are held out from train)
- Questions: violation, remove, severity
- Source for the use case: TypeSafe's own use-case map lists "Moderation and trust and safety" ([docs.typesafe.ai](https://docs.typesafe.ai/concepts/use-case-map))

```bash
python validate_data.py samples/content-moderation/*.jsonl
python train_laya.py --category content-moderation --mode lora --max-steps 60
```

This set is small. Use it to check the format and run a smoke test, then swap in a few hundred of your own labelled rows. Forty rows can't fit calibration temperatures you should trust.

Example question block:

```json
{
 "violation": {
  "type": "choice",
  "instructions": "Which policy, if any, does this post break?",
  "criteria": {
   "none": "allowed content",
   "harassment": "insults or threats aimed at a person",
   "spam": "repetitive promotion, scams or link farming",
   "personal_data": "exposes someone's private contact or identity details"
  }
 },
 "remove": {
  "type": "noul",
  "instructions": "Should the post be removed or hidden pending review?"
 },
 "severity": {
  "type": "score",
  "instructions": "Severity",
  "criteria": [
   "none",
   "low",
   "high"
  ]
 }
}
```
