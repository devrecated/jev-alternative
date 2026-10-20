#!/usr/bin/env bash
# Serve your fine-tuned Laya checkpoint on RunPod with runhug (https://runhug.devrecated.com).
#
# runhug is the deploy path; RunPod is the GPU backend it drives. The script:
#   1. installs a PINNED runhug release (checksum-verified binary from GitHub Releases),
#   2. prints a plan + cost estimate (dry run, nothing created),
#   3. deploys a RunPod serverless endpoint only if you pass --yes.
#
# Usage:
#   python train_laya.py --category insurance-claims --push-to-hub YOUR_ORG/laya-insurance
#   scripts/serve_with_runhug.sh YOUR_ORG/laya-insurance          # plan + estimate only
#   scripts/serve_with_runhug.sh YOUR_ORG/laya-insurance --yes    # actually deploy (spends money)
#
# Credentials: run `runhug connect` once (hidden prompt for the RunPod key) and, for private
# Hub repos, `runhug connect hf`. This script never reads or prints keys.
set -euo pipefail

RUNHUG_VERSION="v0.4.4"   # pinned; bump deliberately after reading the release notes
MODEL="${1:?usage: $0 <hf-org/model> [--yes]}"; shift || true
CONFIRM="${1:-}"
BIN_DIR="${RUNHUG_BIN_DIR:-$HOME/.local/bin}"

install_runhug() {
  local ver="${RUNHUG_VERSION#v}" os arch asset base tmp
  os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  case "$(uname -m)" in x86_64|amd64) arch=amd64 ;; arm64|aarch64) arch=arm64 ;; *) echo "unsupported arch" >&2; exit 1 ;; esac
  asset="runhug_${ver}_${os}_${arch}"
  base="https://github.com/adamsiwiec1/runhug/releases/download/${RUNHUG_VERSION}"
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' RETURN
  curl -fsSL -o "$tmp/$asset" "$base/$asset"
  curl -fsSL -o "$tmp/SHA256SUMS.txt" "$base/SHA256SUMS.txt"
  (cd "$tmp" && grep " ${asset}\$" SHA256SUMS.txt | sha256sum -c -)
  mkdir -p "$BIN_DIR" && install -m 0755 "$tmp/$asset" "$BIN_DIR/runhug"
}

if ! command -v runhug >/dev/null || [ "$(runhug --version 2>/dev/null | awk '{print $2}')" != "${RUNHUG_VERSION#v}" ]; then
  echo "Installing runhug ${RUNHUG_VERSION} into ${BIN_DIR}"
  install_runhug
  export PATH="$BIN_DIR:$PATH"
fi
runhug --version

runhug inspect "$MODEL"
runhug deploy "$MODEL" --dry-run --estimate

if [ "$CONFIRM" = "--yes" ]; then
  runhug deploy "$MODEL" --yes
  runhug list
else
  echo "Dry run only. Re-run with --yes to create the RunPod endpoint."
fi
