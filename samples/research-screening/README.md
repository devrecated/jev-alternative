# Scientific / medical literature screening (synthetic sample)

Screen abstracts for a systematic review: study design, inclusion, methods completeness.

**Every row here is synthetic.** Names, companies, policy numbers, amounts and texts are made up. Nothing comes from a real person, patient or customer.

- Rows: 34 train, 6 test (test scenarios are held out from train)
- Questions: study_design, include, methods_complete
- Source for the use case: TypeSafe's own use-case map lists "Scientific discovery" ([docs.typesafe.ai](https://docs.typesafe.ai/concepts/use-case-map))

```bash
python validate_data.py samples/research-screening/*.jsonl
python train_laya.py --category research-screening --mode lora --max-steps 60
```

This set is small. Use it to check the format and run a smoke test, then swap in a few hundred of your own labelled rows. Forty rows can't fit calibration temperatures you should trust.

Example question block:

```json
{
 "study_design": {
  "type": "choice",
  "instructions": "What study design does this abstract describe?",
  "criteria": {
   "rct": "randomized controlled trial",
   "observational": "cohort, case-control or cross-sectional",
   "review": "systematic review or meta-analysis",
   "preclinical": "in vitro or animal study"
  }
 },
 "include": {
  "type": "noul",
  "instructions": "Does it meet inclusion criteria: human adults, an intervention vs comparator, and a reported primary outcome?"
 },
 "methods_complete": {
  "type": "score",
  "instructions": "How complete are the reported methods (controls, sample size, settings)?",
  "criteria": [
   "missing key details",
   "partial",
   "complete"
  ]
 }
}
```

> **Medical / pharmaceutical disclaimer.** These rows are invented for testing the training pipeline. They are not clinical evidence and say nothing true about any drug, device or study. A model trained on them is not a medical device and must not be used for diagnosis, treatment, dosing or regulatory decisions. Keep PHI out of your own data unless you have the legal basis and safeguards to process it.
