#!/usr/bin/env bash
set -e

# ---- USAGE ----
usage() {
  cat <<EOF
Usage: $0 --hf-token <TOKEN> [--model <ltx2|ltx2.3|all>] [--output <PATH>]

Download LTX model weights for Latent Space Jam via the Hugging Face CLI.

Options:
  --hf-token <TOKEN>                Hugging Face token (required)
  --model    <ltx2|ltx2.3|all>      Which model set to download (default: all)
  --output   <PATH>                 Output directory for downloaded models (default: <repo_root>/models)
  -h, --help                        Show this help message

Examples:
  $0 --hf-token hf_XXXX
  $0 --hf-token hf_XXXX --model ltx2.3
  $0 --hf-token hf_XXXX --model ltx2
  $0 --hf-token hf_XXXX --output /data/models
EOF
  exit 1
}

# ---- PARSE ARGS ----
HF_TOKEN=""
MODEL="all"
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-token) HF_TOKEN="$2";   shift 2 ;;
    --model)    MODEL="$2";      shift 2 ;;
    --output)   OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help)  usage ;;
    *)          echo "Unknown option: $1"; usage ;;
  esac
done

if [[ -z "$HF_TOKEN" ]]; then
  echo "Error: --hf-token is required."
  usage
fi

if [[ "$MODEL" != "ltx2" && "$MODEL" != "ltx2.3" && "$MODEL" != "all" ]]; then
  echo "Error: --model must be 'ltx2', 'ltx2.3', or 'all'."
  usage
fi

# ---- OUTPUT DIR ----
if [ -z "$OUTPUT_DIR" ]; then
  ROOT_DIR=$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")
  OUTPUT_DIR="${ROOT_DIR}/models"
fi
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# ---- HF LOGIN ----
echo "Logging into Hugging Face..."
hf auth login --token "$HF_TOKEN"

# ---- DOWNLOAD HELPER ----
download_hf_file() {
  local repo="$1"
  local filename="$2"
  local dest="${3:-.}"
  echo "  [hf] ${repo}/${filename}"
  hf download "$repo" "$filename" --local-dir "$dest"
}

download_hf_repo() {
  local repo="$1"
  local dest="$2"
  echo "  [hf] ${repo} (full repo)"
  hf download "$repo" --local-dir "$dest"
}

# ---- MODEL DEFINITIONS ----
LTX23_HF_REPO="Lightricks/LTX-2.3"
LTX23_FILES=(
  "ltx-2.3-22b-dev.safetensors"
  "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors"
  "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
  "ltx-2.3-22b-distilled-lora-384.safetensors"
)

LTX2_HF_REPO="Lightricks/LTX-2"
LTX2_FILES=(
  "ltx-2-19b-dev.safetensors"
  "ltx-2-spatial-upscaler-x2-1.0.safetensors"
  "ltx-2-19b-distilled-lora-384.safetensors"
)

GEMMA_HF_REPO="google/gemma-3-12b-it-qat-q4_0-unquantized"
GEMMA_DIR="gemma-3-12b-it-qat-q4_0-unquantized"

# ---- DOWNLOAD FUNCTIONS ----
download_ltx23() {
  echo ""
  echo "=== Downloading LTX-2.3 models ==="
  for file in "${LTX23_FILES[@]}"; do
    download_hf_file "$LTX23_HF_REPO" "$file"
  done
}

download_ltx2() {
  echo ""
  echo "=== Downloading LTX-2 models ==="
  for file in "${LTX2_FILES[@]}"; do
    download_hf_file "$LTX2_HF_REPO" "$file"
  done
}

download_gemma() {
  echo ""
  echo "=== Downloading Gemma 3 text encoder ==="
  download_hf_repo "$GEMMA_HF_REPO" "./${GEMMA_DIR}"
}

# ---- EXECUTE DOWNLOADS ----
case "$MODEL" in
  ltx2.3)
    download_ltx23
    download_gemma
    ;;
  ltx2)
    download_ltx2
    download_gemma
    ;;
  all)
    download_ltx23
    download_ltx2
    download_gemma
    ;;
esac

echo ""
echo "Done! Models saved to: ${OUTPUT_DIR}/"
