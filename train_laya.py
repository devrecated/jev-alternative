#!/usr/bin/env python3
"""Fine-tune the Laya decision model on your own data with Unsloth.

Devrecated Solutions - engineering reference, 2026-10-19.

Every Unsloth call in this file was checked against the unslothai/unsloth source
(unsloth/models/decision.py, last changed in commit a00ef5f56e, 2026-10-13; PyPI unsloth 2026.10.6):
  FastDecisionModel.from_pretrained / get_peft_model / build_dataset / split_holdout /
  evaluate / calibrate / predict / for_inference, DecisionTrainer, model.save_pretrained_merged.

Not run on a GPU by Devrecated yet. Treat the first run as a smoke test.

Laya-specific facts taken from that source (they differ from the LLM recipe in the Unsloth docs):
  * Laya trains in 16-bit. load_in_4bit=True raises NotImplementedError for Laya checkpoints.
  * Without get_peft_model and without full_finetuning, the encoder is frozen and only the decision
    head trains (head-only). get_peft_model adds LoRA to the encoder. full_finetuning=True trains all.
  * The head always trains at head_learning_rate (default 1e-4); TrainingArguments.learning_rate
    applies to the encoder (LoRA or full) only.
  * Unsloth Studio's Laya defaults (studio/backend/assets/configs/model_defaults/decision/
    convaiinnovations_laya.yaml): 4 epochs, LoRA lr 8e-4, r=64, alpha=64, batch 8, grad-accum 8,
    cosine, weight_decay 0.01; full fine-tuning starts at lr 2.5e-5.
  * calibrate() fits one temperature per question type (choice / score / noul) and needs at least
    10 held-out decisions of a type to fit it. evaluate() reports accuracy, ece, loss,
    record_accuracy. It does not report Brier, so this script computes Brier itself.

Usage:
  pip install --upgrade unsloth
  python train_laya.py --train data/sample_train.jsonl --test data/sample_test.jsonl --mode lora
  python train_laya.py --hf-dataset LocalLLaMA/typed-decisions --hf-config all --mode lora
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Import Unsloth before transformers so its patches apply.
from unsloth import FastDecisionModel, DecisionTrainer, is_bfloat16_supported
from transformers import TrainingArguments

CHECKPOINTS = {
    # name: (repo, subfolder). All three live in one HF repo; only the requested one downloads.
    "english": ("convaiinnovations/laya", None),                       # ModernBERT-large, 421M, 512 ctx
    "multilingual": ("convaiinnovations/laya", "multilingual"),        # mmBERT-base, 322M, 1024 ctx
    "typed-decisions": ("convaiinnovations/laya", "typed-decisions"),  # ModernBERT-large, 421M, 1024 ctx
}

# Starting points. LoRA / full values = Unsloth Studio's Laya defaults (see module docstring).
DEFAULT_LR = {"lora": 8e-4, "full": 2.5e-5, "head": 1e-4}  # head-only: no encoder params train; head_lr rules


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{n}: invalid JSON ({exc})")
    return rows


def load_rows(args) -> tuple[list, list | None]:
    if args.hf_dataset:
        from datasets import load_dataset
        train = load_dataset(args.hf_dataset, args.hf_config, split="train")
        test = load_dataset(args.hf_dataset, args.hf_config, split="test")
        return list(train), list(test)
    # Plain JSON lists, not datasets.load_dataset("json"): Arrow would merge differing question
    # schemas across rows into one struct and fill the gaps with nulls.
    return read_jsonl(args.train), (read_jsonl(args.test) if args.test else None)


def _parsed(value):
    return json.loads(value) if isinstance(value, str) else value


def _gold_label(gold):
    # Hard gold ("billing", True, 2) or typed-decisions style soft gold ({"label": ..., "probabilities": ...}).
    if isinstance(gold, dict):
        return gold.get("label")
    return gold


def _norm(key) -> str:
    return str(key).strip().lower()


def brier_and_accuracy(model, tokenizer, rows: list[dict]) -> dict:
    """Multi-class Brier score (sum over options of (p - y)^2, averaged over decisions) and argmax
    accuracy, from FastDecisionModel.predict at the model's current (calibrated) temperatures."""
    FastDecisionModel.for_inference(model)
    total, correct, brier = 0, 0, 0.0
    for row in rows:
        questions, gold = _parsed(row["questions"]), _parsed(row["gold"])
        answers = FastDecisionModel.predict(model, tokenizer, _parsed(row["state"]), questions)
        for name, result in answers.items():
            if name not in gold:
                continue
            label = _norm(_gold_label(_parsed(gold[name])))
            probs = {_norm(k): float(v) for k, v in result["probabilities"].items()}
            if label not in probs:
                continue  # gold not among the options; build_dataset would have skipped it too
            brier += sum((p - (1.0 if k == label else 0.0)) ** 2 for k, p in probs.items())
            correct += int(_norm(result["answer"]) == label)
            total += 1
    if not total:
        return {"decisions": 0}
    return {"decisions": total, "accuracy": correct / total, "brier": brier / total}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--train", help="JSONL with state / questions / gold per line")
    src.add_argument("--hf-dataset", help="Hugging Face dataset id, e.g. LocalLLaMA/typed-decisions")
    ap.add_argument("--hf-config", default="all")
    ap.add_argument("--test", help="separate held-out JSONL, never trained or calibrated on")
    ap.add_argument("--checkpoint", choices=CHECKPOINTS, default="english")
    ap.add_argument("--mode", choices=("lora", "full", "head"), default="lora")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=None, help="encoder learning rate (LoRA / full)")
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-length", type=int, default=None, help="default: checkpoint's served length")
    ap.add_argument("--max-steps", type=int, default=-1, help="set e.g. 60 for a quick smoke test")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--out", default="outputs/laya-finetuned")
    ap.add_argument("--push-to-hub", default=None, help="optional HF repo id for push_to_hub_merged")
    args = ap.parse_args()

    repo, subfolder = CHECKPOINTS[args.checkpoint]
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=repo,
        subfolder=subfolder,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,                      # Laya: 16-bit only (4-bit raises)
        full_finetuning=(args.mode == "full"),
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    print("decision_config max_len / head_max_len:",
          model.decision_config.get("max_len"), model.decision_config.get("head_max_len"))

    train_rows, test_rows = load_rows(args)
    items, report = FastDecisionModel.build_dataset(train_rows, tokenizer, model)
    print(f"Skipped {report['skipped']} of {report['total']} decisions"
          + (f" - most common: {report['reason']}" if report.get("reason") else ""))
    # Calibration holdout: 10% of decisions, capped at 400, whole rows only.
    train_items, calib_items = FastDecisionModel.split_holdout(items, seed=args.seed)
    print(f"{len(train_items)} training decisions, {len(calib_items)} held out for calibration")

    test_items = None
    if test_rows:
        test_items, test_report = FastDecisionModel.build_dataset(test_rows, tokenizer, model)
        print(f"Test: {len(test_items)} decisions ({test_report['skipped']} skipped)")

    # 1) Baseline: the untouched checkpoint on the test set (or the calibration holdout if no test set).
    eval_set, eval_rows = (test_items, test_rows) if test_items else (calib_items, None)
    results = {"checkpoint": args.checkpoint, "mode": args.mode}
    results["base"] = FastDecisionModel.evaluate(model, tokenizer, eval_set)
    if eval_rows:
        results["base"].update({f"{k}_predict": v for k, v in brier_and_accuracy(model, tokenizer, eval_rows).items()})
    print("BASE:", json.dumps(results["base"], indent=2))

    # 2) Pick what trains.
    if args.mode == "lora":
        model = FastDecisionModel.get_peft_model(
            model, r=args.r, lora_alpha=args.alpha, lora_dropout=0.0,
            target_modules="all-linear", use_gradient_checkpointing="unsloth", random_state=args.seed,
        )
    # mode == "full": everything trains (full_finetuning=True above).
    # mode == "head": nothing else to do; from_pretrained froze the encoder, the head trains.

    lr = args.lr if args.lr is not None else DEFAULT_LR[args.mode]
    trainer = DecisionTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_items,
        eval_dataset=calib_items,
        head_learning_rate=args.head_lr,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_steps=0,
            weight_decay=0.01,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            eval_strategy="epoch",
            logging_steps=1,
            output_dir=str(Path(args.out).with_suffix("")) + "-trainer",
            report_to="none",
            seed=args.seed,
        ),
    )
    trainer.train()

    # 3) Calibrate on the holdout only (never on the test set). Writes temperatures into decision_config.
    results["calibration"] = FastDecisionModel.calibrate(model, tokenizer, calib_items)
    if not results["calibration"].get("fitted_types"):
        print("WARNING: no temperature fitted - need >= 10 held-out decisions per question type.")
    print("CALIBRATION (holdout, 2-fold cross-fitted):", json.dumps(results["calibration"], indent=2))

    # 4) Honest evaluation on the untouched test set, same metrics as the baseline.
    if test_items:
        results["finetuned"] = FastDecisionModel.evaluate(model, tokenizer, test_items)
        results["finetuned"].update({f"{k}_predict": v for k, v in brier_and_accuracy(model, tokenizer, test_rows).items()})
        print("FINE-TUNED (test):", json.dumps(results["finetuned"], indent=2))
    else:
        print("No --test set: baseline was measured on the calibration holdout; add a separate test set.")

    # 5) Save a complete merged 16-bit Laya-layout checkpoint (rl_agent_config.json, encoder/, tokenizer/,
    #    model.safetensors) with the fitted temperatures. Serve it with Unsloth's Decision API:
    #    UNSLOTH_SYSTEMONE_MODEL=<out> unsloth studio -H 0.0.0.0 -p 8888
    model.save_pretrained_merged(args.out)
    if args.push_to_hub:
        model.push_to_hub_merged(args.push_to_hub)
    Path(args.out, "devrecated_eval.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"Saved {args.out} and {args.out}/devrecated_eval.json")


if __name__ == "__main__":
    main()
