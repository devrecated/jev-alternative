# Recruiting (synthetic sample)

Match an application to a role, check must-haves, score job-related evidence.

**Every row here is synthetic.** Names, companies, policy numbers, amounts and texts are made up. Nothing comes from a real person, patient or customer.

- Rows: 30 train, 10 test (test scenarios are held out from train)
- Questions: route, meets_must_haves, evidence
- Source for the use case: TypeSafe's own use-case map lists "Recruiting" ([docs.typesafe.ai](https://docs.typesafe.ai/concepts/use-case-map))

```bash
python validate_data.py samples/recruiting/*.jsonl
python train_laya.py --category recruiting --mode lora --max-steps 60
```

This set is small. Use it to check the format and run a smoke test, then swap in a few hundred of your own labelled rows. Forty rows can't fit calibration temperatures you should trust.

Example question block:

```json
{
 "route": {
  "type": "choice",
  "instructions": "Which role is the best match for this application?",
  "criteria": {
   "backend": "backend / platform engineering",
   "data": "data engineering / analytics",
   "design": "product or UX design",
   "sales": "account executive / sales"
  }
 },
 "meets_must_haves": {
  "type": "noul",
  "instructions": "Does the candidate show evidence of every must-have (3+ years in the area, relevant shipped work)?"
 },
 "evidence": {
  "type": "score",
  "instructions": "Strength of job-related evidence",
  "criteria": [
   "weak",
   "some",
   "strong"
  ]
 }
}
```
