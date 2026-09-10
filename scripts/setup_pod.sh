#!/usr/bin/env bash
# =============================================================================
# One-command bring-up for a rented Linux GPU pod (RunPod et al.)
# =============================================================================
#
# Every minute spent debugging pip on a pod is billed. Run this once, straight
# after the pod boots, and it either finishes clean or tells you exactly what is
# wrong before you have spent anything meaningful.
#
#   bash scripts/setup_pod.sh              # full setup
#   bash scripts/setup_pod.sh --gdc-only   # just fetch the Linux gdc-client
#   bash scripts/setup_pod.sh --check      # verify an existing env, install nothing
#
# Expects, in the environment:
#   HF_TOKEN   Hugging Face token for the gated MahmoodLab/UNI model.
#              Set it on the pod BEFORE running:  export HF_TOKEN=hf_...
#              Your HF account must also have accepted the UNI licence.
#
# Sets HF_HOME to <repo>/.hf so the 1.3 GB UNI download lands on the persistent
# volume and survives the pod being destroyed and recreated.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${REPO_ROOT}/tools"

# GDC Data Transfer Tool v2.3.0, Ubuntu x64 build.
# Verified against gdc.cancer.gov/access-data/gdc-data-transfer-tool on
# 2026-08-25. If GDC bumps the version this URL 404s — check that page and
# update both lines. Override the checksum with GDC_MD5=... if it has changed.
GDC_URL="${GDC_URL:-https://gdc.cancer.gov/system/files/public/file/gdc-client_2.3_Ubuntu_x64-py3.8-ubuntu-20.04.zip}"
GDC_MD5="${GDC_MD5:-18591d74de07cdcd396dab71c52663da}"

CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"

MODE="full"
[[ "${1:-}" == "--gdc-only" ]] && MODE="gdc"
[[ "${1:-}" == "--check" ]]    && MODE="check"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
install_gdc_client() {
  say "Fetching the Linux gdc-client"
  local dest="${TOOLS_DIR}/gdc-client-linux"

  if [[ -x "$dest" ]] && "$dest" --version >/dev/null 2>&1; then
    echo "already present and runnable: $dest"
    return
  fi

  mkdir -p "$TOOLS_DIR"
  local tmp; tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN

  curl -fSL --retry 3 -o "${tmp}/gdc.zip" "$GDC_URL" \
    || die "download failed: $GDC_URL
       Check gdc.cancer.gov/access-data/gdc-data-transfer-tool for a new version
       and re-run with:  GDC_URL=<new url> GDC_MD5=<new md5> bash $0 --gdc-only"

  local got; got="$(md5sum "${tmp}/gdc.zip" | awk '{print $1}')"
  if [[ "$got" != "$GDC_MD5" ]]; then
    die "gdc-client checksum mismatch
       expected $GDC_MD5
       got      $got
     GDC probably published a new build. Verify the checksum on the GDC page,
     then re-run with GDC_MD5=<the published md5>."
  fi
  echo "md5 ok"

  # GDC ships a zip INSIDE the zip: the outer archive contains
  # gdc-client_2.3_Ubuntu_x64.zip, and the binary is inside that. Unzip
  # repeatedly until no new zips appear, then look for the binary anywhere.
  unzip -qo "${tmp}/gdc.zip" -d "$tmp"
  local round=0
  while [[ $round -lt 3 ]]; do
    local inner; inner="$(find "$tmp" -name '*.zip' ! -name 'gdc.zip' -type f | head -1)"
    [[ -n "$inner" ]] || break
    unzip -qo "$inner" -d "$(dirname "$inner")/unpacked_${round}"
    rm -f "$inner"
    round=$((round + 1))
  done

  local bin; bin="$(find "$tmp" -name 'gdc-client' -type f | head -1)"
  [[ -n "$bin" ]] || die "no gdc-client binary found after unpacking. Contents were:
$(find "$tmp" -type f | head -20)"
  mv "$bin" "$dest"
  chmod +x "$dest"

  "$dest" --version >/dev/null 2>&1 || die "$dest will not execute on this machine"
  echo "installed: $dest  ($("$dest" --version 2>&1 | head -1))"
}

# -----------------------------------------------------------------------------
install_python_deps() {
  say "Installing Python dependencies"
  python -m pip install --upgrade pip >/dev/null

  if python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "torch with working CUDA already present — leaving it alone"
  else
    echo "installing torch for CUDA (index: ${CUDA_INDEX})"
    python -m pip install torch torchvision --index-url "$CUDA_INDEX"
  fi

  python -m pip install -r "${REPO_ROOT}/requirements.txt"
}

# -----------------------------------------------------------------------------
verify() {
  say "Verifying the environment"
  local fail=0

  python - <<'PY' || fail=1
import os, sys, shutil

def ok(msg):   print(f"  [ ok ] {msg}")
def bad(msg):  print(f"  [FAIL] {msg}"); globals().__setitem__("BAD", True)
BAD = False

try:
    import torch
    ok(f"torch {torch.__version__}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        ok(f"CUDA visible: {n} x {torch.cuda.get_device_name(0)}, "
           f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB")
    else:
        bad("torch sees no CUDA device — you are paying for a GPU you cannot use")
except Exception as e:
    bad(f"torch: {e}")

try:
    import openslide
    ok(f"openslide {openslide.__library_version__} (python wrapper "
       f"{openslide.__version__})")
except Exception as e:
    bad(f"openslide: {e}\n         -> pip install openslide-bin")

for mod in ("timm", "skimage", "numpy", "PIL", "tqdm", "huggingface_hub"):
    try:
        __import__(mod); ok(mod)
    except Exception as e:
        bad(f"{mod}: {e}")

cpus = os.cpu_count() or 0
if cpus >= 8:
    ok(f"{cpus} vCPU — enough decode workers to keep the GPU fed")
else:
    print(f"  [warn] only {cpus} vCPU. Tile decoding is the bottleneck in this "
          f"pipeline, so the GPU will idle. Consider a pod with more vCPUs.")

if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
    ok("HF_TOKEN is set")
else:
    bad("HF_TOKEN is not set — UNI is a gated model and step 01 will fail.\n"
        "         -> export HF_TOKEN=hf_...  (and accept the licence at\n"
        "            huggingface.co/MahmoodLab/UNI)")

print(f"  [info] HF_HOME = {os.environ.get('HF_HOME', '(default ~/.cache)')}")
sys.exit(1 if BAD else 0)
PY

  if [[ -x "${TOOLS_DIR}/gdc-client-linux" ]] \
     && "${TOOLS_DIR}/gdc-client-linux" --version >/dev/null 2>&1; then
    echo "  [ ok ] gdc-client-linux runs"
  else
    echo "  [FAIL] gdc-client-linux missing or not runnable"
    fail=1
  fi

  [[ $fail -eq 0 ]] || die "environment is not ready — fix the FAIL lines above BEFORE starting a long run"
  say "Environment OK"
}

# -----------------------------------------------------------------------------
say "Repo: ${REPO_ROOT}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf}"
mkdir -p "$HF_HOME"
echo "HF_HOME=${HF_HOME}  (keep this on the persistent volume)"

case "$MODE" in
  gdc)   install_gdc_client ;;
  check) verify ;;
  full)
    if command -v apt-get >/dev/null 2>&1; then
      say "System packages"
      apt-get update -qq && apt-get install -y -qq unzip curl >/dev/null || \
        warn "apt-get failed — continuing; unzip/curl may already be present"
    fi
    install_python_deps
    install_gdc_client
    verify
    cat <<EOF

Next:
  export HF_HOME="${HF_HOME}"
  export HF_TOKEN=hf_...

  # Gate 1 — one slide, end to end, inside tmux so a dropped SSH does not kill it
  tmux new -s embed
  python scripts/00_fetch_embed_batches.py --batch-size 1 --max-batches 1

  Read the MEASURED per-slide download/embed seconds it prints at the end.
  That is the number every cost estimate in docs/execution_roadmap.md is
  currently guessing at. Decide on the full run only after seeing it.
EOF
    ;;
esac
