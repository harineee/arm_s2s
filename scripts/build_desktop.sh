#!/bin/bash
# Build the desktop translation pipeline.
# Requires: cmake, g++/clang++, ONNX Runtime, SentencePiece
# Optional: PortAudio (for real-time microphone mode)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build-desktop"

echo "=== Building Desktop Translation Pipeline ==="
echo "Project root: $PROJECT_ROOT"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$PROJECT_ROOT/desktop" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build . -j"$(nproc)"

echo ""
echo "Build complete: $BUILD_DIR/translation_desktop"
echo ""
echo "Usage:"
echo "  $BUILD_DIR/translation_desktop \\"
echo "    --asr-model models/asr/ggml-tiny.en.bin \\"
echo "    --mt-model models/mt/onnx/encoder_model.onnx \\"
echo "    --tts-model models/tts/hi_IN-rohan-medium.onnx \\"
echo "    --input test.wav --output hindi.wav"
