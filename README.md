# ARM-- Real-Time English → Hindi Speech-to-Speech Translation

**On-device, CPU-only, fully offline** translation from spoken English to spoken Hindi. Runs on **ARM Android** (arm64-v8a, NEON) and **x86/Linux** (host build for development). No cloud; no Python at runtime.

---

## What This Project Does

1. **Captures** English speech (microphone or WAV file).
2. **Transcribes** it with **whisper.cpp** (tiny.en, INT8).
3. **Translates** to Hindi with **Marian NMT** (encoder–decoder ONNX, INT8, SentencePiece).
4. **Synthesizes** Hindi speech with **Piper VITS** (ONNX, INT8).
5. **Plays** the result (speaker or WAV file).

**Target:** &lt; 500 ms time-to-first-audio where feasible; phrase-level streaming to keep latency low.

---

## Architecture (High Level)

```
Audio In (16 kHz) → [ASR] → English text → [Phrase detect] → [MT] → Hindi text → [TTS] → Audio Out
                    whisper.cpp            Marian ONNX              Piper VITS
```

- **Three pipeline threads** (ASR, MT, TTS) with **lock-free SPSC queues** between them.
- **Phrase-level translation**: boundaries on pause (&gt;150 ms), word count (2–5), or punctuation so we don’t wait for full sentences.
- **C++ only** at runtime: whisper.cpp, ONNX Runtime, SentencePiece (statically linked on Android).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

---

## Repository Layout

```
armm/
├── asr/              # ASR: whisper.cpp wrapper (submodule: asr/whisper_cpp)
├── mt/               # MT: Marian ONNX + SentencePiece
├── tts/              # TTS: Piper VITS ONNX + Hindi phonemizer
├── pipeline/         # Orchestrator, phrase detector, lock-free queues
├── host/             # Host (desktop) CLI app — PortAudio or file I/O
├── android/          # Android app: JNI, Gradle, UI
├── models/           # Model configs & tokenizers (weights downloaded separately)
├── scripts/          # Build, model download, quantization, Android prep
├── utils/            # environment.yml (Conda env for model prep)
├── third_party/      # SentencePiece (for Android build)
└── docs/             # ARCHITECTURE.md, BUILD.md
```

**Models (not in repo):** ASR ~75 MB, MT ~216 MB INT8, TTS ~17 MB INT8. Download and quantize via `scripts/`; see below.

---

## Requirements

| Context | Requirements |
|--------|---------------|
| **Host build** | CMake ≥3.18, C++17, ONNX Runtime C++, SentencePiece; optional PortAudio |
| **Android build** | Android SDK 34, NDK r25+, CMake 3.22; run `scripts/setup_android_deps.sh` for ONNX Runtime + SentencePiece |
| **Android device** | arm64-v8a, Android 8+, 4–6 GB RAM, ~400 MB free storage, microphone permission |

---

## Quick Start (Host)

```bash
git clone <repo-url> armm && cd armm
git submodule update --init --recursive

# Conda env (model download/quantization)
conda env create -f utils/environment.yml
conda activate armhack
conda install onnxruntime onnxruntime-tools onnx -c conda-forge

# Download models
./scripts/download_asr_model.sh
./scripts/download_mt_model.sh
./scripts/download_tts_model.sh

# Quantize for smaller size (optional for host, required for mobile)
pip install onnxruntime
python -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
base = 'models/mt/onnx'
for name in ['encoder_model', 'decoder_model', 'decoder_with_past_model']:
    quantize_dynamic(base + f'/{name}.onnx', base + f'/{name}_int8.onnx', weight_type=QuantType.QInt8)
"

# Build host binary
./scripts/build_host.sh

# Run (file mode)
./build-host/translation_host \
  --asr-model models/asr/ggml-tiny.en.bin \
  --mt-model models/mt/onnx/encoder_model.onnx \
  --tts-model models/tts/hi_IN-rohan-medium.onnx \
  --input your_audio.wav --output hindi.wav
```

---

## Android Build & Deploy

```bash
# 1. Android deps (ONNX Runtime + SentencePiece for arm64-v8a)
./scripts/setup_android_deps.sh

# 2. Prepare models (INT8) and copy into app assets
./scripts/prepare_models_android.sh

# 3. Build APK
cd android && ./gradlew assembleDebug

# 4. Install on device
adb install app/build/outputs/apk/debug/app-debug.apk
```

Open the app → allow microphone → wait for “Ready” (first run extracts ~312 MB from APK) → tap **Start** and speak English.

---

## Model Sizes (INT8, Mobile)

| Component | Size |
|-----------|------|
| ASR (ggml-tiny.en.bin) | 74 MB |
| MT (encoder + decoder + decoder_with_past) | ~217 MB |
| TTS (Piper Hindi) | ~17 MB |
| **Total in APK** | ~312 MB |

Peak RAM on device is estimated ~765 MB; suitable for 4–6 GB phones.

---

## Tech Choices (for Evaluators)

| Component | Choice | Rationale |
|-----------|--------|-----------|
| ASR | whisper.cpp (tiny.en) | C++, NEON-friendly, no ONNX; low latency |
| MT | Marian NMT (ONNX) | Dedicated NMT, 10–30 ms per phrase; INT8 for mobile |
| TTS | Piper VITS (ONNX) | Single-stage, Hindi support; INT8 quantized |
| Runtime | ONNX Runtime (C++) | One stack for MT + TTS on Android |
| Tokenization | SentencePiece (C++) | Same as Marian; built from source or prebuilt for Android |
| Concurrency | Lock-free SPSC queues | Avoids mutex latency spikes on mobile |

---

## License & Third-Party

- whisper.cpp: MIT  
- ONNX Runtime: MIT  
- SentencePiece: Apache 2.0  
- Piper / OPUS-MT models: see their repos (e.g. MIT, CC-BY-4.0)


## Troubleshooting

| Issue | Action |
|-------|--------|
| `whisper.cpp not found` | `git submodule update --init --recursive` |
| ONNX Runtime / SentencePiece not found (host) | `conda install onnxruntime-cpp sentencepiece -c conda-forge` |
| Android build fails (CMake / NDK) | Check `android/local.properties`, NDK path; run `scripts/setup_android_build.sh` if needed |
| App OOM on device | Use INT8 models only; run `prepare_models_android.sh` |
| No sound / placeholder text | Check model paths and that RECORD_AUDIO is granted |

For more detail, see [docs/BUILD.md](docs/BUILD.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
