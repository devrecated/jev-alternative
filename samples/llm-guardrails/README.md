# LLM guardrails (synthetic sample)

Check LLM inputs, outputs and tool calls for prompt injection, data exposure and unsafe advice.

**Every row here is synthetic.** Names, companies, policy numbers, amounts and texts are made up. Nothing comes from a real person, patient or customer.

- Rows: 34 train, 6 test (test scenarios are held out from train)
- Questions: risk_type, block, confidence_needed
- Source for the use case: TypeSafe's own use-case map lists "LLM guardrails" ([docs.typesafe.ai](https://docs.typesafe.ai/concepts/use-case-map))

```bash
python validate_data.py samples/llm-guardrails/*.jsonl
python train_laya.py --category llm-guardrails --mode lora --max-steps 60
```

This set is small. Use it to check the format and run a smoke test, then swap in a few hundred of your own labelled rows. Forty rows can't fit calibration temperatures you should trust.

Example question block:

```json
{
 "risk_type": {
  "type": "choice",
  "instructions": "What risk does this LLM input or output carry?",
  "criteria": {
   "safe": "normal request or answer",
   "prompt_injection": "tries to override system instructions or tools",
   "data_exposure": "leaks secrets, credentials or personal data",
   "unsafe_advice": "dangerous or unqualified medical, legal or financial advice"
  }
 },
 "block": {
  "type": "noul",
  "instructions": "Should the harness block this before it reaches the model/user?"
 },
 "confidence_needed": {
  "type": "score",
  "instructions": "How much review does this need?",
  "criteria": [
   "auto-allow",
   "log + sample",
   "human review"
  ]
 }
}
```

> **Medical / pharmaceutical disclaimer.** These rows are invented for testing the training pipeline. They are not clinical evidence and say nothing true about any drug, device or study. A model trained on them is not a medical device and must not be used for diagnosis, treatment, dosing or regulatory decisions. Keep PHI out of your own data unless you have the legal basis and safeguards to process it.
