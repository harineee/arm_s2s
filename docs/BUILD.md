# Build Instructions

## Prerequisites

### Host Machine Requirements
- **OS**: Linux (Ubuntu 20.04+) or macOS
- **CMake**: 3.18 or higher
- **Android NDK**: r25 or higher
- **Git**: For cloning submodules
- **Python 3**: For model download scripts (optional)

### Android Device Requirements
- **Architecture**: Arm64-v8a (aarch64)
- **Android Version**: 8.0+ (API 26+)
- **RAM**: Minimum 2 GB
- **Storage**: ~500 MB for models

## Step 1: Clone Repository and Submodules

```bash
git clone <repository-url>
cd armm
git submodule update --init --recursive
```

### Required Submodules

1. **whisper.cpp** (ASR)
   ```bash
   git submodule add https://github.com/ggerganov/whisper.cpp.git asr/whisper_cpp
   ```

2. **Marian NMT** (MT)
   ```bash
   git submodule add https://github.com/marian-nmt/marian.git mt/marian
   ```

3. **VITS** (TTS) - Optional, may use ONNX Runtime instead
   ```bash
   git submodule add https://github.com/jaywalnut310/vits.git tts/vits
   ```

## Step 2: Download Models

### ASR Model (whisper.cpp)

```bash
cd asr/whisper_cpp
# Download tiny.en INT8 quantized model
./models/download-ggml-model.sh tiny.en
# Or manually:
# wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin
```

### MT Model (OPUS-MT EN→HI)

Download from OPUS-MT repository:
```bash
mkdir -p models
cd models
# Download OPUS-MT EN→HI model
# See: https://github.com/Helsinki-NLP/OPUS-MT-train
# Convert to Marian format or ONNX
```

**Note**: Marian models are typically in `.npz` format. For Android, you may need to:
1. Convert to ONNX format
2. Use ONNX Runtime for inference
3. Or build Marian with Android NDK support

### TTS Model (Hindi VITS)

Download from Indic-TTS:
```bash
# See: https://github.com/AI4Bharat/Indic-TTS
# Download Hindi VITS checkpoint
# Convert PyTorch model to ONNX for inference
```

## Step 3: Build Native Libraries

### Option A: Build with Android Studio

1. Open `android/` directory in Android Studio
2. Sync Gradle
3. Build → Make Project
4. Native libraries will be built automatically

### Option B: Build with Command Line

```bash
cd android
./gradlew assembleDebug
```

### Option C: Build Individual Components

#### Build whisper.cpp

```bash
cd asr/whisper_cpp
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
         -DANDROID_ABI=arm64-v8a \
         -DANDROID_PLATFORM=android-26 \
         -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_SHARED_LIBS=OFF
make -j4
```

#### Build Marian NMT

**Note**: Marian has complex dependencies. For Android, consider:
1. Using ONNX Runtime with converted models
2. Building Marian statically with Android NDK
3. Using a pre-built inference library

```bash
cd mt/marian
# See Marian documentation for Android build
# This is complex and may require custom toolchain
```

**Recommended Alternative**: Use ONNX Runtime
```bash
# Download ONNX Runtime Mobile for Android
# See: https://onnxruntime.ai/docs/build/inferencing.html#mobile
```

#### Build VITS (or use ONNX Runtime)

Similar to Marian, VITS models can be converted to ONNX and run with ONNX Runtime.

## Step 4: Configure Model Paths

Update `MainActivity.java` with actual model paths:

```java
String modelsDir = getFilesDir().getAbsolutePath() + "/models";
String asrModel = modelsDir + "/ggml-tiny.en.bin";
String mtModel = modelsDir + "/opus-mt-en-hi.onnx";  // If using ONNX
String ttsModel = modelsDir + "/hindi-vits.onnx";    // If using ONNX
```

## Step 5: Install and Run

```bash
# Build APK
cd android
./gradlew assembleDebug

# Install on device
adb install app/build/outputs/apk/debug/app-debug.apk

# Copy models to device
adb push models/ /data/data/com.armm.translation/files/models/

# Run
adb shell am start -n com.armm.translation/.MainActivity
```

## Troubleshooting

### Build Errors

1. **NDK not found**: Set `ANDROID_NDK` environment variable
   ```bash
   export ANDROID_NDK=/path/to/android-ndk-r25
   ```

2. **CMake version**: Ensure CMake 3.18+
   ```bash
   cmake --version
   ```

3. **Submodule issues**: Reinitialize submodules
   ```bash
   git submodule update --init --recursive --force
   ```

### Runtime Errors

1. **Model not found**: Check model paths in `MainActivity.java`
2. **Permission denied**: Ensure RECORD_AUDIO permission is granted
3. **Library not found**: Check that native libraries are included in APK

### Performance Issues

1. **High latency**: 
   - Reduce chunk sizes
   - Enable NEON optimizations
   - Use quantized models (INT8/INT4)

2. **Memory issues**:
   - Use smaller models
   - Reduce buffer sizes
   - Enable model quantization

## Model Conversion Notes

### Converting Marian to ONNX

```python
# Example script (requires transformers, onnx)
from transformers import MarianMTModel, MarianTokenizer
import torch

model_name = "Helsinki-NLP/opus-mt-en-hi"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Export to ONNX
dummy_input = tokenizer("Hello", return_tensors="pt")
torch.onnx.export(model, dummy_input, "opus-mt-en-hi.onnx")
```

### Converting VITS to ONNX

```python
# Load VITS checkpoint
import torch

checkpoint = torch.load("hindi-vits.pth", map_location="cpu")
model = checkpoint["model"]

# Export to ONNX
dummy_text = torch.randint(0, 100, (1, 50))
torch.onnx.export(model, dummy_text, "hindi-vits.onnx")
```

## Next Steps

See [ARCHITECTURE.md](ARCHITECTURE.md) for architecture details and [OPTIMIZATION.md](OPTIMIZATION.md) for optimization strategies.
