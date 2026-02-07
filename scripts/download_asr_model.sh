#!/usr/bin/env bash
# Download whisper.cpp tiny.en model (INT8 quantized, ~75 MB).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../models/asr"
MODEL_FILE="${MODEL_DIR}/ggml-tiny.en.bin"
URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin"

mkdir -p "${MODEL_DIR}"

if [ -f "${MODEL_FILE}" ]; then
    echo "ASR model already exists: ${MODEL_FILE}"
    exit 0
fi

echo "Downloading whisper.cpp tiny.en model (~75 MB)..."
curl -L -o "${MODEL_FILE}" "${URL}"
echo "Saved to: ${MODEL_FILE}"
