# laya-unsloth

Fine-tune the [Laya](https://huggingface.co/convaiinnovations/laya) decision model on our own labelled
decisions with [Unsloth](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth),
calibrate it on a holdout, and compare base vs fine-tuned on a separate test set.

Spec: the internal Devrecated "Laya x Unsloth" training guide (2026-10-19), §5. The PDF isn't in this
repo because it's marked confidential. Ask engineering for a copy.

> **License warning (AGPL-3.0).** Laya's weights are Apache-2.0. **Unsloth's decision-model source files
> (`unsloth/models/decision*.py`, `_decision_common.py`) carry an `AGPL-3.0-only` header.** This repo
> imports them. Get counsel to review before any Unsloth code ships inside a client product or a hosted
> service.

| File | What |
|---|---|
| `train_laya.py` | Loads Laya in 16-bit, builds the dataset, splits a calibration holdout, evaluates the **base** checkpoint, trains (LoRA / full / head-only), calibrates on the holdout, evaluates **fine-tuned** on the test set (accuracy, ECE, loss, record accuracy from Unsloth, plus Brier computed by this script), then saves a merged 16-bit Laya checkpoint and `devrecated_eval.json`. |
| `validate_data.py` | Checks JSONL rows against the Unsloth / Laya dataset rules and prints per-label counts. |
| `data/sample_train.jsonl`, `data/sample_test.jsonl` | Illustrative support-ticket rows (32 train / 10 test rows, 3 questions each). Use them as a format reference and smoke test. They're too small to fit calibration temperatures. |
| `scripts/runpod_train.py` | Runs `train_laya.py` once on a Runpod GPU pod, pulls back `train.log` and `devrecated_eval.json`, and always terminates the pod (with a spend cap). |

## Laya-specific settings (they differ from the LLM recipe in the Unsloth docs)

- **16-bit only.** `load_in_4bit=True` raises `NotImplementedError` for Laya checkpoints.
- **Defaults = Unsloth Studio's Laya defaults:** LoRA `r=64`, `alpha=64`, encoder LR `8e-4`, **4 epochs**,
  batch 8 × grad-accum 8, cosine, weight decay 0.01. Head LR is `1e-4`. A full fine-tune starts at `2.5e-5`.
- `calibrate()` fits one temperature per question type (choice / score / noul) and needs ≥10 held-out
  decisions of a type.
- `evaluate()` doesn't return Brier, so `train_laya.py` computes the multi-class Brier score itself from
  `FastDecisionModel.predict` (reported as `brier_predict`).

## Data format

One JSON object per line with `state`, `questions`, `gold`:

```json
{"state": "Hi, I was charged twice for invoice #4411. Please refund the duplicate today.",
 "questions": {
   "team":    {"type": "choice", "instructions": "Which team should handle this?",
               "criteria": {"billing": "invoices, payments, refunds, charges", "technical": "bugs, outages, errors, integrations",
                            "sales": "pricing, new plans, upgrades, quotes", "account": "login, access, users, settings"}},
   "refund":  {"type": "noul", "instructions": "Does the customer ask for a refund or credit?"},
   "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "soon", "today / blocking"]}},
 "gold": {"team": "billing", "refund": true, "urgency": 2}}
```

Gold rules: choice → a key of `criteria`; noul → `true`/`false`; score → 0-based level index. Don't use
boolean-word keys (yes/no, true/false) in choice questions. Keep test rows from a different source,
customer or time window than train.

## Run

Needs one CUDA GPU (Unsloth reports ~2.5 GB VRAM for its Laya fine-tune). bf16 is used when supported,
otherwise fp16.

```bash
pip install -r requirements.txt
python -c "from unsloth import FastDecisionModel, DecisionTrainer"     # smoke test; set USE_TF=0 if it hangs
python validate_data.py data/*.jsonl

# Quick smoke run on the sample rows
python train_laya.py --train data/sample_train.jsonl --test data/sample_test.jsonl --mode lora --max-steps 60

# Our data
python train_laya.py --train data/train.jsonl --test data/test.jsonl --checkpoint english --mode lora

# Reproduce the public benchmark (1,200 train / 400 test cases)
python train_laya.py --hf-dataset LocalLLaMA/typed-decisions --hf-config all --mode lora
```

Checkpoints: `--checkpoint english` (ModernBERT-large, 512 ctx), `multilingual` (mmBERT-base; use it for any
non-English input), and `typed-decisions`. Modes: `--mode lora` (default), `full`, `head`.

### On a Runpod pod (no local GPU)

```bash
export RUNPOD_API_KEY=...            # never commit it
python scripts/runpod_train.py --gpu "NVIDIA RTX A5000" --max-usd 5 -- \
    --hf-dataset LocalLLaMA/typed-decisions --hf-config all --mode lora
```

The script terminates the pod when training finishes, when the spend cap is hit, or on any error, and
verifies it's gone. Results land in `results/<pod id>/`, which is git-ignored.

## Serve

```bash
UNSLOTH_SYSTEMONE_MODEL=outputs/laya-finetuned unsloth studio -H 0.0.0.0 -p 8888   # Jev-compatible /v1/systemone
```

## Honest evaluation checklist

Report test-set numbers only, base vs fine-tuned on the same rows, next to majority-class and random
baselines. Slice by question type, label and option count. Re-ask with shuffled option order. Adopt Laya
only if it beats the best zero-shot option (Jev, Clef or an LLM) on our test set at an acceptable ECE.

## Status

The Unsloth API calls were checked against the `unsloth==2026.10.6` wheel (`unsloth/models/decision.py`)
but haven't run on a GPU yet.
