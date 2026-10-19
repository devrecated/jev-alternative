#!/usr/bin/env python3
"""Run train_laya.py once on a Runpod GPU pod, pull the results back, and always terminate the pod.

Reads RUNPOD_API_KEY from the environment (never printed). The repo's code and data are shipped to the
pod as a base64 tarball in an env var, so the pod needs no git access. When training finishes the pod
serves its output folder on port 8000 through the Runpod proxy; this script downloads train.log and
devrecated_eval.json into results/<pod id>/ and then terminates the pod.

Spend guard: the pod is terminated once (cost per hour x elapsed hours) reaches --max-usd, on any
error, and on Ctrl-C. Termination is verified by polling until the pod no longer exists.

  python scripts/runpod_train.py --gpu "NVIDIA RTX A5000" --max-usd 5 -- \
      --train data/sample_train.jsonl --test data/sample_test.jsonl --mode lora
  python scripts/runpod_train.py --gpu "NVIDIA RTX A5000" --max-usd 5 -- \
      --hf-dataset LocalLLaMA/typed-decisions --hf-config all --mode lora

Untested against a live pod so far: treat the first run as a smoke test and watch the Runpod console.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shlex
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.runpod.io/graphql"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
SHIP = ["train_laya.py", "validate_data.py", "requirements.txt", "data"]


def gql(query: str, variables: dict | None = None) -> dict:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY is not set")
    req = urllib.request.Request(
        API,
        data=json.dumps({"query": query, "variables": variables or {}}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                 "User-Agent": "laya-unsloth/0.1"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        out = json.load(resp)
    if out.get("errors"):
        raise RuntimeError(json.dumps(out["errors"]))
    return out["data"]


def payload() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in SHIP:
            tar.add(ROOT / name, arcname=name)
    return base64.b64encode(buf.getvalue()).decode()


def start_command(train_args: list[str]) -> str:
    args = " ".join(shlex.quote(a) for a in train_args)
    script = (
        "set -o pipefail; mkdir -p /work/out && cd /work && "
        "echo \"$LAYA_PAYLOAD\" | base64 -d | tar xz && "
        "(pip install --upgrade pip && pip install --upgrade -r requirements.txt && "
        "python validate_data.py data/*.jsonl && nvidia-smi && "
        f"USE_TF=0 python train_laya.py {args} --out /work/out/laya-finetuned; "
        "echo EXIT_CODE=$? ) 2>&1 | tee /work/out/train.log; "
        "cp /work/out/laya-finetuned/devrecated_eval.json /work/out/ 2>/dev/null; "
        "touch /work/out/DONE; cd /work/out && python -m http.server 8000"
    )
    return "bash -c " + shlex.quote(script)


def pod_status(pod_id: str) -> dict | None:
    return gql("query($id:String!){ pod(input:{podId:$id}){ id desiredStatus costPerHr runtime{ uptimeInSeconds } } }",
               {"id": pod_id})["pod"]


def fetch(pod_id: str, name: str) -> bytes | None:
    try:
        with urllib.request.urlopen(f"https://{pod_id}-8000.proxy.runpod.net/{name}", timeout=30) as r:
            return r.read()
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return None


def terminate(pod_id: str) -> None:
    for _ in range(3):
        try:
            gql("mutation($id:String!){ podTerminate(input:{podId:$id}) }", {"id": pod_id})
        except Exception as exc:  # retry, then verify below
            print(f"terminate error: {exc}", file=sys.stderr)
        for _ in range(12):
            try:
                if pod_status(pod_id) is None:
                    print(f"Pod {pod_id} terminated (verified: pod no longer exists).")
                    return
            except Exception:
                pass
            time.sleep(5)
    print(f"WARNING: could not verify pod {pod_id} is terminated. Check the Runpod console NOW.", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="NVIDIA RTX A5000", help="Runpod gpuTypeId")
    ap.add_argument("--cloud", default="ALL", choices=("ALL", "COMMUNITY", "SECURE"))
    ap.add_argument("--max-usd", type=float, default=5.0, help="hard spend cap for this pod")
    ap.add_argument("--disk-gb", type=int, default=40)
    ap.add_argument("train_args", nargs=argparse.REMAINDER, help="arguments after -- go to train_laya.py")
    args = ap.parse_args()
    train_args = [a for a in args.train_args if a != "--"] or [
        "--train", "data/sample_train.jsonl", "--test", "data/sample_test.jsonl", "--mode", "lora"]

    pod = gql(
        "mutation($in:PodFindAndDeployOnDemandInput){ podFindAndDeployOnDemand(input:$in){ id costPerHr machine{ gpuDisplayName } } }",
        {"in": {
            "cloudType": args.cloud, "gpuCount": 1, "gpuTypeId": args.gpu, "name": "laya-unsloth-train",
            "imageName": IMAGE, "containerDiskInGb": args.disk_gb, "volumeInGb": 0, "ports": "8000/http",
            "dockerArgs": start_command(train_args),
            "env": [{"key": "LAYA_PAYLOAD", "value": payload()}, {"key": "USE_TF", "value": "0"}],
        }},
    )["podFindAndDeployOnDemand"]
    pod_id, rate = pod["id"], float(pod["costPerHr"])
    print(f"Pod {pod_id} on {pod['machine']['gpuDisplayName']} at ${rate:.3f}/hr; cap ${args.max_usd:.2f}")
    started = time.time()
    out_dir = ROOT / "results" / pod_id
    try:
        while True:
            spent = rate * (time.time() - started) / 3600
            if spent >= args.max_usd:
                print(f"Spend cap reached (${spent:.2f}); stopping.")
                break
            if fetch(pod_id, "DONE") is not None:
                print("Training finished on the pod.")
                break
            time.sleep(30)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("train.log", "devrecated_eval.json"):
            data = fetch(pod_id, name)
            if data is not None:
                (out_dir / name).write_bytes(data)
                print(f"Saved {out_dir / name}")
    finally:
        elapsed = time.time() - started
        terminate(pod_id)
        print(f"Wall time {elapsed / 60:.1f} min; estimated cost ${rate * elapsed / 3600:.3f}")


if __name__ == "__main__":
    main()
