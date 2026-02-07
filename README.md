# ARMM — Real-Time English → Hindi Speech-to-Speech Translation

On-device, CPU-only, fully offline speech-to-speech translation for ARM Android and x86 desktop.

**Pipeline:** Microphone → ASR (whisper.cpp) → MT (Marian ONNX) → TTS (Piper VITS ONNX) → Speaker

**Target latency:** < 500 ms time-to-first-audio (after quantization + KV cache).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Audio In (16 kHz mono)                                              │
│       ↓                                                              │
│  ┌─────────────────┐    Lock-Free     ┌──────────────────┐           │
│  │  ASR Thread      │───  Queue  ────→│  MT Thread        │          │
│  │  whisper.cpp     │    (SPSC)       │  Marian ONNX      │          │
│  │  tiny.en INT8    │                 │  encoder-decoder   │          │
│  │  2 threads       │                 │  SentencePiece     │          │
│  └─────────────────┘                  │  2 ONNX threads    │          │
│                                       └────────┬───────────┘          │
│                                                │                      │
│                           Lock-Free            ↓                      │
│  ┌─────────────────┐←───  Queue  ────┌──────────────────┐            │
│  │  Audio Out       │    (SPSC)      │  TTS Thread       │            │
│  │  16 kHz mono     │               │  Piper VITS ONNX   │            │
│  └─────────────────┘                │  Hindi phonemizer   │            │
│                                     │  1 ONNX thread      │            │
│                                     └──────────────────┘              │
└──────────────────────────────────────────────────────────────────────┘
```

**Key design:**
- 3 pipeline threads + system audio threads
- Lock-free SPSC queues (cache-line aligned, power-of-2 capacity)
- Phrase-level translation (pause detection, 1-8 word boundaries)
- No Python runtime dependencies — pure C++

---

## Folder Structure

```
armm/
├── asr/                        # ASR module (whisper.cpp wrapper)
│   ├── asr_wrapper.h/cpp
│   ├── CMakeLists.txt
│   └── whisper_cpp/            # Git submodule
├── mt/                         # Machine Translation (Marian ONNX)
│   ├── mt_wrapper.h/cpp        # Public interface
│   ├── marian_mt_wrapper.h/cpp # ONNX encoder-decoder + SentencePiece
│   └── CMakeLists.txt
├── tts/                        # Text-to-Speech (Piper VITS ONNX)
│   ├── tts_wrapper.h/cpp       # ONNX inference + Hindi phonemizer
│   └── CMakeLists.txt
├── pipeline/                   # Pipeline orchestrator
│   ├── pipeline.h/cpp          # Multi-threaded pipeline
│   ├── phrase_detector.h/cpp   # Phrase boundary detection
│   ├── lockfree_queue.h        # SPSC lock-free queue
│   ├── performance_tracker.h/cpp
│   └── CMakeLists.txt
├── desktop/                    # Desktop entry point
│   ├── main.cpp                # CLI app (PortAudio + file mode)
│   └── CMakeLists.txt
├── android/                    # Android application
│   ├── app/                    # Java/Kotlin UI + Gradle
│   ├── native-lib/             # JNI bridge + CMake
│   └── scripts/
├── models/                     # Model files (download separately)
│   ├── asr/ggml-tiny.en.bin    # 75 MB
│   ├── mt/onnx/                # ~863 MB FP32 / ~215 MB INT8
│   └── tts/                    # ~61 MB FP32 / ~15 MB INT8
├── scripts/                    # Build + utility scripts
│   ├── build_desktop.sh
│   ├── download_asr_model.sh
│   ├── download_mt_model.sh
│   ├── download_tts_model.sh
│   ├── quantize_models.py
│   └── install_portaudio.sh
├── CMakeLists.txt              # Root CMake (component libs)
├── environment.yml             # Conda env (model conversion only)
└── README.md
```

---

## Dependencies

### Laptop / Build Machine

| Dependency | Version | Purpose | Install |
|---|---|---|---|
| CMake | ≥ 3.18 | Build system | `apt install cmake` |
| GCC/Clang | C++17 | Compiler | `apt install g++` |
| Git | any | Submodules | `apt install git` |
| ONNX Runtime C++ | ≥ 1.16 | MT + TTS inference | `conda install onnxruntime-cpp -c conda-forge` |
| SentencePiece C++ | ≥ 0.1.99 | MT tokenization | `conda install sentencepiece -c conda-forge` |
| PortAudio | ≥ 19 | Real-time mic (optional) | `apt install libportaudio2 portaudio19-dev` |
| Android SDK | 34 | Android build | Android Studio |
| Android NDK | r25+ | Native compilation | Android Studio SDK Manager |
| Python 3 | ≥ 3.9 | Model quantization only | System Python |
| onnxruntime (pip) | ≥ 1.16 | Quantization script | `pip install onnxruntime` |

### Android Phone

| Requirement | Detail |
|---|---|
| Architecture | ARMv8-A (arm64-v8a) |
| Android | API 26+ (Android 8.0+) |
| RAM | ≥ 4 GB (6 GB recommended) |
| Storage | ~400 MB for all models |
| Permissions | Microphone (RECORD_AUDIO) |
| Network | Not required (fully offline) |

---

## Installation — Step by Step

### 1. Clone Repository

```bash
git clone <repo-url> armm
cd armm
git submodule update --init --recursive
```

### 2. Set Up Build Environment

```bash
# Option A: Conda (recommended)
conda env create -f environment.yml
conda activate armhack
conda install onnxruntime-cpp sentencepiece -c conda-forge

# Option B: System packages (Ubuntu/Debian)
sudo apt install cmake g++ git libportaudio2 portaudio19-dev
# Then install ONNX Runtime + SentencePiece from source or conda
```

### 3. Download Models

```bash
# ASR model (75 MB)
./scripts/download_asr_model.sh

# MT model — download OPUS-MT EN→HI and export to ONNX
./scripts/download_mt_model.sh

# TTS model — download Piper Hindi voice
./scripts/download_tts_model.sh
```

After download, verify:
```bash
ls -lh models/asr/ggml-tiny.en.bin          # ~75 MB
ls -lh models/mt/onnx/encoder_model.onnx    # ~195 MB
ls -lh models/mt/onnx/decoder_model.onnx    # ~340 MB
ls -lh models/tts/hi_IN-rohan-medium.onnx   # ~61 MB
```

---

## Model Quantization (Required for Mobile)

Quantize FP32 ONNX models to INT8:

```bash
pip install onnxruntime
python scripts/quantize_models.py
```

Result:
```
MT models:
  encoder_model.onnx      195 MB → ~49 MB (25%)
  decoder_model.onnx      340 MB → ~85 MB (25%)
  decoder_with_past.onnx  328 MB → ~82 MB (25%)
TTS model:
  hi_IN-rohan-medium.onnx  61 MB → ~15 MB (25%)
```

Total: **~1 GB → ~231 MB** (fits in mobile RAM).

---

## Build — Desktop

```bash
./scripts/build_desktop.sh
```

Or manually:
```bash
mkdir -p build-desktop && cd build-desktop
cmake ../desktop -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

---

## Build — Android APK

### 1. Prepare Models for APK

```bash
# Quantize first (see above), then:
./android/scripts/prepare_models.sh
```

### 2. Build APK

```bash
cd android
./gradlew assembleDebug
```

APK location: `android/app/build/outputs/apk/debug/app-debug.apk`

### 3. Install on Device

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

---

## Running — Desktop

### File Mode (no microphone)

```bash
./build-desktop/translation_desktop \
    --asr-model models/asr/ggml-tiny.en.bin \
    --mt-model models/mt/onnx/encoder_model.onnx \
    --tts-model models/tts/hi_IN-rohan-medium.onnx \
    --input test_audio.wav \
    --output hindi_output.wav
```

### Real-Time Mode (requires PortAudio)

```bash
./build-desktop/translation_desktop \
    --asr-model models/asr/ggml-tiny.en.bin \
    --mt-model models/mt/onnx/encoder_model.onnx \
    --tts-model models/tts/hi_IN-rohan-medium.onnx
```

Press Ctrl+C to stop.

---

## Running — Android

1. Install APK (see above)
2. Open "ARMM Translation" app
3. Grant microphone permission
4. Tap **Start** — speak English into the mic
5. Hindi text appears on screen, Hindi audio plays through speaker
6. Tap **Stop** to end

### Monitor via ADB

```bash
# Live logs
adb logcat -s NativePipeline:V ASR:V MT:V TTS:V

# Memory usage
adb shell dumpsys meminfo com.armm.translation

# CPU per thread
adb shell top -H -p $(adb shell pidof com.armm.translation)

# Thermal zones
adb shell cat /sys/class/thermal/thermal_zone*/temp
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `whisper.cpp not found` | Submodule not initialized | `git submodule update --init --recursive` |
| `ONNX Runtime not found` | Library not installed | `conda install onnxruntime-cpp -c conda-forge` |
| `SentencePiece not found` | Library not installed | `conda install sentencepiece -c conda-forge` |
| `Failed to load source.spm` | Missing tokenizer file | Re-run `./scripts/download_mt_model.sh` |
| `TTS ONNX model not found` | Model not downloaded | Re-run `./scripts/download_tts_model.sh` |
| Test tone instead of speech | TTS model missing or init failed | Check model path and ONNX Runtime availability |
| `[HI: ...]` placeholder text | MT model not loaded | Verify ONNX models exist in `models/mt/onnx/` |
| App crash on Android | OOM — models too large | Quantize models to INT8 first |
| Silence on Android | Audio permission denied | Grant RECORD_AUDIO in settings |
| Very slow translation | No KV cache / FP32 models | Use INT8 models; KV cache is auto-attempted |
| Build error: `-ffast-math` | whisper.cpp incompatibility | Already fixed — ensure CMake uses `-funroll-loops` not `-ffast-math` |

---

## Performance Expectations

### Desktop (x86_64, 4-core, 3.5 GHz)

| Component | Latency |
|---|---|
| ASR (1s audio chunk) | ~100-200 ms |
| MT (5-word phrase) | ~200-500 ms |
| TTS (short phrase) | ~300-600 ms |
| **End-to-end** | **~600-1300 ms** |

### Android (Snapdragon 8-series, INT8 quantized)

| Component | Latency (estimated) |
|---|---|
| ASR | ~200-400 ms |
| MT (with KV cache) | ~200-500 ms |
| TTS | ~300-700 ms |
| **End-to-end** | **~700-1600 ms** |

### Model Sizes

| Model | FP32 | INT8 |
|---|---|---|
| ASR (whisper tiny.en) | 75 MB | 75 MB (already quantized) |
| MT (encoder + decoder + decoder_past) | 863 MB | ~216 MB |
| TTS (Piper VITS Hindi) | 61 MB | ~15 MB |
| **Total** | **999 MB** | **~306 MB** |

### RAM Usage (INT8, Android)

| Component | Estimated |
|---|---|
| ASR | ~130 MB |
| MT (all sessions) | ~275 MB |
| TTS | ~25 MB |
| Pipeline + Android | ~70 MB |
| **Total** | **~500 MB** |

Fits within 4-6 GB device memory budget.

---

## Mobile Setup — Laptop Requirements

**Install all of these on your laptop/build machine:**

```bash
# 1. System packages
sudo apt update
sudo apt install -y cmake g++ git pkg-config

# 2. Android SDK + NDK (via Android Studio or command-line)
# Download: https://developer.android.com/studio
# In SDK Manager, install:
#   - Android SDK Platform 34
#   - Android NDK (r25 or later)
#   - CMake 3.18.1
#   - Android SDK Build-Tools 34

# 3. Set environment variables
export ANDROID_HOME=$HOME/Android/Sdk
export ANDROID_NDK_HOME=$ANDROID_HOME/ndk/<version>
export PATH=$PATH:$ANDROID_HOME/platform-tools

# 4. Conda environment (for model prep + desktop build)
conda env create -f environment.yml
conda activate armhack
conda install onnxruntime-cpp sentencepiece -c conda-forge
pip install onnxruntime  # for quantization script

# 5. Verify
cmake --version     # ≥ 3.18
adb version         # installed
ndk-build --version # NDK accessible
```

## Mobile Setup — Android Phone Requirements

**On the phone itself:**

1. **Enable Developer Options:** Settings → About Phone → tap Build Number 7 times
2. **Enable USB Debugging:** Settings → Developer Options → USB Debugging → ON
3. **Connect via USB** and authorize the laptop
4. **Verify:** `adb devices` shows your device
5. **Storage:** Ensure ≥ 500 MB free internal storage
6. **No additional apps** required — everything runs as a native APK

---

## License

Components:
- whisper.cpp: MIT
- ONNX Runtime: MIT
- SentencePiece: Apache 2.0
- Piper TTS models: MIT
- OPUS-MT models: CC-BY-4.0
