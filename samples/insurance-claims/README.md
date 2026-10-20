# Insurance claims (synthetic sample)

Classify first-notice-of-loss reports, catch missing info, score settlement complexity.

**Every row here is synthetic.** Names, companies, policy numbers, amounts and texts are made up. Nothing comes from a real person, patient or customer.

- Rows: 32 train, 8 test (test scenarios are held out from train)
- Questions: claim_type, missing_info, complexity
- Source for the use case: TypeSafe's own use-case map lists "Insurance claims" ([docs.typesafe.ai](https://docs.typesafe.ai/concepts/use-case-map))

```bash
python validate_data.py samples/insurance-claims/*.jsonl
python train_laya.py --category insurance-claims --mode lora --max-steps 60
```

This set is small. Use it to check the format and run a smoke test, then swap in a few hundred of your own labelled rows. Forty rows can't fit calibration temperatures you should trust.

Example question block:

```json
{
 "claim_type": {
  "type": "choice",
  "instructions": "What kind of claim is this?",
  "criteria": {
   "auto": "vehicle collision, theft or glass",
   "property": "home, water, fire, storm damage",
   "liability": "injury or damage to a third party",
   "travel": "trip cancellation, lost baggage"
  }
 },
 "missing_info": {
  "type": "noul",
  "instructions": "Is key information (date, location, policy number or photos) missing?"
 },
 "complexity": {
  "type": "score",
  "instructions": "How complex is the claim to settle?",
  "criteria": [
   "straight-through",
   "standard review",
   "specialist / possible fraud"
  ]
 }
}
```

> **Medical / pharmaceutical disclaimer.** These rows are invented for testing the training pipeline. They are not clinical evidence and say nothing true about any drug, device or study. A model trained on them is not a medical device and must not be used for diagnosis, treatment, dosing or regulatory decisions. Keep PHI out of your own data unless you have the legal basis and safeguards to process it.
