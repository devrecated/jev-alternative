# Legal and compliance (synthetic sample)

Tell contracts, policies, ad copy and filings apart, flag prohibited claims or missing clauses, score severity.

**Every row here is synthetic.** Names, companies, policy numbers, amounts and texts are made up. Nothing comes from a real person, patient or customer.

- Rows: 30 train, 10 test (test scenarios are held out from train)
- Questions: doc_type, violation, severity
- Source for the use case: TypeSafe's own use-case map lists "Legal and compliance" ([docs.typesafe.ai](https://docs.typesafe.ai/concepts/use-case-map))

```bash
python validate_data.py samples/legal-compliance/*.jsonl
python train_laya.py --category legal-compliance --mode lora --max-steps 60
```

This set is small. Use it to check the format and run a smoke test, then swap in a few hundred of your own labelled rows. Forty rows can't fit calibration temperatures you should trust.

Example question block:

```json
{
 "doc_type": {
  "type": "choice",
  "instructions": "What kind of document is this excerpt from?",
  "criteria": {
   "contract": "agreement between parties",
   "policy": "internal or public policy",
   "marketing": "ad copy, landing page or claim",
   "filing": "regulatory or court filing"
  }
 },
 "violation": {
  "type": "noul",
  "instructions": "Does the excerpt contain a prohibited claim, missing required clause, or policy violation?"
 },
 "severity": {
  "type": "score",
  "instructions": "How serious is the issue for counsel?",
  "criteria": [
   "none / cosmetic",
   "needs edit",
   "escalate to counsel"
  ]
 }
}
```
