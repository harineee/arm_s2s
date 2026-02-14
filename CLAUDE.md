# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ARM S2S is an on-device, CPU-only, fully offline real-time English-to-Hindi speech-to-speech translation system. It targets ARM Android (arm64-v8a) and x86/Linux (host dev builds). Pure C++17 at runtime — no Python, no cloud APIs. Features NMT-accelerated speculative decoding where Marian NMT drafts and Qwen3-0.6B (ExecuTorch) verifies in a single forward pass.

## Build Commands

### Host (Desktop) Build
```bash
# Without ExecuTorch (NMT only):
./scripts/build_host.sh

# With ExecuTorch (LLM translation):
mkdir -p build-host && cd build-host
cmake ../host -DCMAKE_BUILD_TYPE=Release -DUSE_EXECUTORCH=ON -DET_ROOT=/path/to/executorch
make -j$(nproc)
```

### Android Build
```bash
./scripts/setup_android_deps.sh          # ONNX Runtime AAR + SentencePiece
./scripts/prepare_models_android.sh      # INT8/INT4 quantize, copy to assets (incl. LLM)
cd android && ./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

### Model Setup
```bash
git submodule update --init --recursive  # whisper.cpp submodule
./scripts/download_asr_model.sh
./scripts/download_mt_model.sh
./scripts/download_tts_model.sh
./scripts/download_llm_model.sh          # Qwen3-0.6B from HuggingFace
./scripts/export_llm_model.sh            # Export to .pte (ExecuTorch format)
```

### Conda Environment
```bash
conda activate deer-arm                  # Python 3.10 + ExecuTorch + torch + transformers
```

### Running (Host)
```bash
# Balanced mode (speculative decoding — default):
./build-host/translation_host \
  --asr-model models/asr/ggml-tiny.en.bin \
  --mt-model models/mt/onnx/encoder_model.onnx \
  --tts-model models/tts/hi_IN-rohan-medium.onnx \
  --llm-model models/llm/qwen3_0.6B.pte \
  --llm-tokenizer models/llm/Qwen3-0.6B/tokenizer.json \
  --translation-mode balanced \
  --input test.wav --output hindi.wav

# Translation modes: speed | balanced | quality
```

### Running Tests
```bash
# Speculative decoding test (requires ExecuTorch build):
g++ -std=c++17 -DUSE_EXECUTORCH -I/path/to/executorch/include \
    tests/test_speculative.cpp mt/llm_translator.cpp \
    -lexecutorch -lextension_module -lextension_tensor -o test_speculative
./test_speculative --model models/llm/qwen3_0.6B.pte --tokenizer models/llm/Qwen3-0.6B/tokenizer.json
```

## Architecture

**Data flow:** Audio In (16kHz PCM) → ASR → English text → Phrase Detector → MT → Hindi text → TTS → Audio Out

**Three-thread parallel pipeline** connected by lock-free SPSC queues:
- **Thread 1 (ASR):** whisper.cpp streaming, 40-80ms chunks, greedy decoding
- **Thread 2 (MT):** Three-mode routing via `TranslationMode` enum
- **Thread 3 (TTS):** Piper VITS via ONNX Runtime, Devanagari→IPA conversion, chunked synthesis

**Translation modes** (`mt/translation_mode.h`):
- **SPEED:** Marian NMT only (~20ms)
- **BALANCED:** NMT draft + LLM speculative verification (~150ms) — default
- **QUALITY:** Full LLM autoregressive generation (~500ms)

**Speculative decoding** (`LLMTranslator::speculative_translate()`): NMT generates Hindi draft → tokenize prompt+draft → ONE LLM forward pass → verify each draft token via argmax → accept matching prefix → regenerate from rejection point. Uses `Module::forward()` for raw logits access (not TextLlmRunner).

**Phrase boundary detection** triggers translation on: pause >150ms, 2-5 words accumulated, punctuation, or max word count.

## Key Source Directories

- `asr/` — whisper.cpp wrapper; submodule at `asr/whisper_cpp`
- `mt/` — Translation module:
  - `translation_mode.h` — SPEED/BALANCED/QUALITY enum
  - `llm_translator.h/cpp` — ExecuTorch/Qwen3 with speculative decoding (behind `#ifdef USE_EXECUTORCH`)
  - `marian_mt_wrapper.h/cpp` — ONNX Runtime + SentencePiece (behind `#ifdef USE_ONNXRUNTIME`)
  - `mt_wrapper.h/cpp` — Three-mode routing layer with mode control and metrics
- `tts/` — Piper VITS wrapper
- `pipeline/` — Orchestrator, `lockfree_queue.h` (wait-free SPSC), `phrase_detector`, `performance_tracker`
- `host/` — Desktop CLI with `--translation-mode` arg
- `android/native-lib/` — JNI bridge with `nativeSetTranslationMode`/`nativeGetTranslationMode`
- `tests/test_speculative.cpp` — Standalone speculative decoding test
- `models/llm/` — Qwen3 model files (downloaded separately)

## CMake Structure

`USE_EXECUTORCH` is the master compile switch — defined in root, host, and android CMakeLists. When ON:
- `mt/CMakeLists.txt` adds `llm_translator.cpp` + `translation_mode.h` and links `extension_module extension_tensor`
- Desktop: set `ET_ROOT` to ExecuTorch install dir
- Android: prebuilt `libexecutorch_jni.so` from `libs/executorch/` or Maven AAR

The project **must** still compile and work without ExecuTorch (`USE_EXECUTORCH=OFF`).

Root CMakeLists adds subdirs: mt → asr → tts → pipeline (order matters). `-ffast-math` intentionally avoided (breaks whisper.cpp).

## Android Integration

- JNI `nativeInit` accepts LLM model paths + translation_mode (0=SPEED, 1=BALANCED, 2=QUALITY)
- `nativeSetTranslationMode`/`nativeGetTranslationMode` for runtime mode switching
- `nativeGetAcceptanceRate` returns speculative decoding token acceptance rate
- `.pte` files have compression disabled in Gradle (`noCompress`)
- ExecuTorch AAR: `org.pytorch:executorch-android:0.7.0` + soloader + fbjni

## Concurrency Pattern

Lock-free SPSC queues (`lockfree_queue.h`) use power-of-2 capacity with cache-line aligned (64-byte) atomic positions. Memory ordering: `relaxed` for local reads, `acquire`/`release` at synchronization points. Four queues: Audio→ASR, ASR→MT, MT→TTS, TTS→Playback.
