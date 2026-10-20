# E-commerce marketplaces (synthetic sample)

Categorize listings, flag counterfeit or prohibited claims, score listing quality.

**Every row here is synthetic.** Names, companies, policy numbers, amounts and texts are made up. Nothing comes from a real person, patient or customer.

- Rows: 30 train, 10 test (test scenarios are held out from train)
- Questions: category, policy_flag, quality
- Source for the use case: TypeSafe's own use-case map lists "E-commerce marketplaces" ([docs.typesafe.ai](https://docs.typesafe.ai/concepts/use-case-map))

```bash
python validate_data.py samples/ecommerce-listings/*.jsonl
python train_laya.py --category ecommerce-listings --mode lora --max-steps 60
```

This set is small. Use it to check the format and run a smoke test, then swap in a few hundred of your own labelled rows. Forty rows can't fit calibration temperatures you should trust.

Example question block:

```json
{
 "category": {
  "type": "choice",
  "instructions": "Which catalog category fits this listing?",
  "criteria": {
   "electronics": "phones, audio, computers, accessories",
   "apparel": "clothing, shoes, bags",
   "home": "kitchen, furniture, decor",
   "beauty": "skin care, cosmetics, fragrance"
  }
 },
 "policy_flag": {
  "type": "noul",
  "instructions": "Does the listing show counterfeit signals or a prohibited claim?"
 },
 "quality": {
  "type": "score",
  "instructions": "Listing quality (title, attributes, description)",
  "criteria": [
   "poor",
   "ok",
   "good"
  ]
 }
}
```
