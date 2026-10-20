# jev-alternative

<p align="center">
  <img src="assets/hero.svg" alt="jev-alternative: fine-tune an open Jev-style decision model on your own labels" width="100%"/>
</p>

**A free, open-source Jev alternative you can fine-tune on your own data.**

[Jev](https://madewithjev.com/jev-fine-tune) is TypeSafe's hosted decision model. You send it a piece of text and some typed questions, and it returns probabilities for each answer. The weights aren't released, so you can't fine-tune it or run it yourself. This repo takes [Laya](https://huggingface.co/convaiinnovations/laya), an open Jev-style decision model with Apache-2.0 weights, and fine-tunes it on your labelled decisions with [Unsloth](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth). It then calibrates the result on a holdout and compares base vs fine-tuned on a separate test set.

Built by [Devrecated](https://devrecated.com). Pairs with [runhug](https://github.com/adamsiwiec1/runhug) for RunPod setup.

## Get started in 3 commands

```bash
# 1) Install (needs one CUDA GPU; no GPU? see "Run on RunPod with runhug" below)
git clone https://github.com/devrecated/jev-alternative && cd jev-alternative && pip install -r requirements.txt

# 2) Check your data
python validate_data.py data/*.jsonl

# 3) Smoke-test a LoRA fine-tune on the sample rows
python train_laya.py --train data/sample_train.jsonl --test data/sample_test.jsonl --mode lora --max-steps 60
```

## Features

- Fine-tunes Laya with LoRA (default), a full fine-tune, or head-only training.
- Answers the same three question types as Jev: `choice`, `score` and `noul` (yes/no).
- Calibrates one temperature per question type on a holdout, so probabilities can drive automation thresholds.
- Reports test-set accuracy, ECE, loss and record accuracy from Unsloth, plus a multi-class Brier score this script computes itself.
- Saves a merged 16-bit Laya checkpoint and a `devrecated_eval.json` with base and fine-tuned numbers side by side.
- Includes a RunPod launcher with a hard spend cap that always terminates the pod.
- Serves the fine-tuned model through Unsloth Studio's Jev-compatible `/v1/systemone` endpoint.

## How it works

| File | What it does |
|---|---|
| `train_laya.py` | Loads Laya in 16-bit and builds the dataset. Splits off a calibration holdout and evaluates the **base** checkpoint. Trains, calibrates, then evaluates the **fine-tuned** model on the test set. Saves the merged checkpoint and `devrecated_eval.json`. |
| `validate_data.py` | Checks JSONL rows against the Unsloth / Laya dataset rules and prints counts per label. |
| `data/sample_train.jsonl`, `data/sample_test.jsonl` | Made-up support-ticket rows (32 train, 10 test, 3 questions each). Good for checking the format and a smoke test, but too small to fit calibration temperatures. |
| `scripts/runpod_train.py` | Runs `train_laya.py` once on a RunPod GPU pod, pulls back `train.log` and `devrecated_eval.json`, and always terminates the pod. |
| `scripts/serve_results.py` | Runs on the pod and serves results only to requests that carry the run's random bearer token. |

### Data format

One JSON object per line with `state`, `questions` and `gold`:

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

Gold rules: for `choice`, use a key of `criteria`. For `noul`, use `true` or `false`. For `score`, use the 0-based level index. Don't use yes/no or true/false as keys in choice questions. Take test rows from a different source, customer or time window than the training rows.

## Fine-tune with Unsloth

Unsloth reports about 2.5 GB of VRAM for its Laya fine-tune. Training uses bf16 when the GPU supports it and fp16 otherwise.

```bash
python -c "from unsloth import FastDecisionModel, DecisionTrainer"     # smoke test; set USE_TF=0 if it hangs

# Your data
python train_laya.py --train data/train.jsonl --test data/test.jsonl --checkpoint english --mode lora

# Reproduce the public benchmark (1,200 train / 400 test cases)
python train_laya.py --hf-dataset LocalLLaMA/typed-decisions --hf-config all --mode lora
```

Checkpoints: `--checkpoint english` (ModernBERT-large, 512 context), `multilingual` (mmBERT-base, for any non-English input) and `typed-decisions`. Modes: `--mode lora` (default), `full` and `head`.

Laya's settings differ from the LLM recipe in the Unsloth docs:

- Laya only loads in 16-bit. `load_in_4bit=True` raises `NotImplementedError` for Laya checkpoints.
- The defaults match Unsloth Studio's Laya defaults: LoRA `r=64`, `alpha=64`, encoder LR `8e-4`, 4 epochs, batch 8 × grad-accum 8, cosine schedule, weight decay 0.01. Head LR is `1e-4`. Start a full fine-tune at `2.5e-5`.
- `calibrate()` fits one temperature per question type and needs at least 10 held-out decisions of that type.
- `evaluate()` doesn't return Brier, so `train_laya.py` computes it from `FastDecisionModel.predict` and reports it as `brier_predict`.

### Serve it

```bash
UNSLOTH_SYSTEMONE_MODEL=outputs/laya-finetuned unsloth studio -H 0.0.0.0 -p 8888   # Jev-compatible /v1/systemone
```

## Run on RunPod with runhug

No local GPU? Rent one on [RunPod](https://www.runpod.io). [runhug](https://github.com/adamsiwiec1/runhug) is our CLI for RunPod and Hugging Face. Use it to store your keys and pick a GPU, then launch training with `scripts/runpod_train.py`.

```bash
# Install runhug (macOS / Linux; apt, dnf, Scoop, winget and npm are in the runhug README)
brew install --cask openhat-security/tap/runhug
# or: npm install -g runhug

runhug connect          # store your RunPod key (and `runhug connect hf` for Hugging Face)
runhug gpu list         # see available GPU types

# Launch one training run with a $5 cap
export RUNPOD_API_KEY=...            # the launcher reads it from the environment; never commit it
python scripts/runpod_train.py --gpu "NVIDIA RTX A5000" --max-usd 5 -- \
    --hf-dataset LocalLLaMA/typed-decisions --hf-config all --mode lora
```

The launcher terminates the pod when training finishes, when the spend cap is hit, on any error and on Ctrl-C, then checks that the pod is gone. Results land in `results/<pod id>/`, which git ignores. While the pod runs, its result server on port 8000 answers only requests carrying a random token generated for that run. The token is printed once to your terminal and never written to `train.log`.

## FAQ

**Is this Jev?** No. It fine-tunes Laya, a separate open model built to answer the same kind of typed questions. It isn't affiliated with TypeSafe.

**Can I fine-tune Jev itself?** No. Jev runs behind TypeSafe's API and its weights aren't public. To use your own labels, you need an open model like Laya.

**Will it beat Jev?** Maybe, on your own task. Laya's published results show tasks where Jev still wins, so measure on your test set. Report test-set numbers only: base vs fine-tuned on the same rows, next to majority-class and random baselines. Slice by question type, label and number of options, and re-ask with the options shuffled. Switch only if Laya beats the best zero-shot option (Jev, Clef or an LLM) at an acceptable ECE.

**Do I need a big GPU?** No. Unsloth reports about 2.5 GB of VRAM for a Laya fine-tune.

**Has this run end to end?** The Unsloth API calls were checked against the `unsloth==2026.10.6` wheel (`unsloth/models/decision.py`), but they haven't run on a GPU yet. The RunPod launcher hasn't run against a live pod either. Treat your first run as a smoke test and watch the RunPod console.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). Report security problems privately as described in [SECURITY.md](SECURITY.md).

## License

[AGPL-3.0](LICENSE). Laya's weights are Apache-2.0. Unsloth's decision-model source files (`unsloth/models/decision*.py`, `_decision_common.py`) carry an `AGPL-3.0-only` header, and this repo imports them. Get legal advice before shipping this code inside a product or a hosted service.
