# Lead generation (synthetic sample)

Segment inbound leads, detect purchase intent, score ICP fit.

**Every row here is synthetic.** Names, companies, policy numbers, amounts and texts are made up. Nothing comes from a real person, patient or customer.

- Rows: 30 train, 10 test (test scenarios are held out from train)
- Questions: segment, purchase_intent, fit
- Source for the use case: TypeSafe's own use-case map lists "Lead generation" ([docs.typesafe.ai](https://docs.typesafe.ai/concepts/use-case-map))

```bash
python validate_data.py samples/lead-scoring/*.jsonl
python train_laya.py --category lead-scoring --mode lora --max-steps 60
```

This set is small. Use it to check the format and run a smoke test, then swap in a few hundred of your own labelled rows. Forty rows can't fit calibration temperatures you should trust.

Example question block:

```json
{
 "segment": {
  "type": "choice",
  "instructions": "Which segment is this lead?",
  "criteria": {
   "enterprise": "1,000+ employees",
   "midmarket": "100-999 employees",
   "smb": "under 100 employees",
   "not_a_fit": "student, job seeker, competitor or spam"
  }
 },
 "purchase_intent": {
  "type": "noul",
  "instructions": "Does the message show purchase intent (budget, timeline, pricing or demo request)?"
 },
 "fit": {
  "type": "score",
  "instructions": "Fit with the ideal customer profile",
  "criteria": [
   "poor",
   "partial",
   "strong"
  ]
 }
}
```
