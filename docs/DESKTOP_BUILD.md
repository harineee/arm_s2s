# Desktop CPU Build Instructions

This guide explains how to build and run the speech-to-speech translation system on a desktop CPU (Linux/macOS/Windows) instead of Android.

## Prerequisites

### Required
- **CMake**: 3.18 or higher
- **C++ Compiler**: GCC 7+ or Clang 8+ with C++17 support
- **Make** or **Ninja**: Build system

### Optional (for real-time audio)
- **PortAudio**: For microphone/speaker I/O
  - Linux: `sudo apt-get install libportaudio2 libportaudio-dev`
  - macOS: `brew install portaudio`
  - Or use the install script: `./scripts/install_portaudio.sh`

## Quick Start

### 1. Build the Project

```bash
# Using the build script (recommended)
./scripts/build_desktop.sh

# Or manually:
mkdir build-desktop && cd build-desktop
cmake ../desktop -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### 2. Download Models

```bash
# Download ASR model
./scripts/download_asr_model.sh

# For MT and TTS models, see docs/BUILD.md
# They need to be converted to ONNX format
```

### 3. Run the Application

#### Real-time Mode (with PortAudio)

```bash
./build-desktop/translation_desktop \
    --asr-model models/asr/ggml-tiny.en.bin \
    --mt-model models/mt/opus-mt-en-hi.onnx \
    --tts-model models/tts/hindi-vits.onnx
```

#### File Mode (without PortAudio)

```bash
./build-desktop/translation_desktop \
    --asr-model models/asr/ggml-tiny.en.bin \
    --mt-model models/mt/opus-mt-en-hi.onnx \
    --tts-model models/tts/hindi-vits.onnx \
    --input input.wav \
    --output output.wav
```

## Architecture Differences

### Desktop vs Android

| Component | Android | Desktop |
|-----------|---------|---------|
| Audio Input | AudioRecord (JNI) | PortAudio or WAV file |
| Audio Output | AudioTrack (JNI) | PortAudio or WAV file |
| Build System | Gradle + CMake | CMake only |
| Entry Point | MainActivity.java | main.cpp |
| Threading | Android threads | std::thread |

### Audio I/O Options

1. **PortAudio** (recommended for real-time)
   - Cross-platform audio library
   - Supports microphone and speaker
   - Real-time processing

2. **File-based** (fallback)
   - Read from WAV file
   - Write to WAV file
   - Good for testing and batch processing

## Building Components

### Individual Component Builds

```bash
# Build only ASR module
cd build-desktop
cmake --build . --target asr_wrapper

# Build only MT module
cmake --build . --target mt_wrapper

# Build only TTS module
cmake --build . --target tts_wrapper

# Build only pipeline
cmake --build . --target pipeline
```

### Debug Build

```bash
mkdir build-desktop-debug && cd build-desktop-debug
cmake ../desktop -DCMAKE_BUILD_TYPE=Debug
cmake --build . -j$(nproc)
```

## Usage Examples

### Basic Usage

```bash
# Real-time translation (requires PortAudio)
./translation_desktop \
    --asr-model models/asr/ggml-tiny.en.bin \
    --mt-model models/mt/opus-mt-en-hi.onnx \
    --tts-model models/tts/hindi-vits.onnx
```

### File Processing

```bash
# Process a WAV file
./translation_desktop \
    --asr-model models/asr/ggml-tiny.en.bin \
    --mt-model models/mt/opus-mt-en-hi.onnx \
    --tts-model models/tts/hindi-vits.onnx \
    --input english_speech.wav \
    --output hindi_speech.wav
```

### Help

```bash
./translation_desktop --help
```

## Troubleshooting

### PortAudio Not Found

**Error**: `PortAudio not found. Building without real-time audio`

**Solution**:
```bash
# Install PortAudio
./scripts/install_portaudio.sh

# Rebuild
./scripts/build_desktop.sh
```

### Model Not Found

**Error**: `Failed to initialize pipeline`

**Solution**:
- Check model paths are correct
- Verify model files exist
- Ensure models are in correct format (ONNX for MT/TTS)

### Audio Device Issues

**Error**: `Failed to open input/output stream`

**Solution**:
- Check microphone/speaker permissions
- Verify audio devices are available
- Try file mode instead: `--input file.wav --output out.wav`

### Build Errors

**CMake version too old**:
```bash
# Install newer CMake or use package manager
sudo apt-get install cmake  # Linux
brew install cmake          # macOS
```

**Compiler not found**:
```bash
# Install build essentials
sudo apt-get install build-essential  # Linux
xcode-select --install                # macOS
```

## Performance Tuning

### CPU Optimization

The build automatically enables:
- **Arm64**: NEON SIMD instructions
- **x86_64**: AVX2 and FMA instructions

### Thread Configuration

Edit `desktop/main.cpp` or use environment variables to configure:
- Number of threads per component
- Thread priorities
- CPU affinity

### Memory Optimization

- Reduce buffer sizes in `pipeline/pipeline.cpp`
- Use smaller models (tiny.en for ASR)
- Enable quantization (INT8)

## Testing

### Unit Tests (if available)

```bash
cd build-desktop
ctest
```

### Manual Testing

1. **Test ASR only**: Use a simple English WAV file
2. **Test MT only**: Provide English text, verify Hindi output
3. **Test TTS only**: Provide Hindi text, verify audio output
4. **Test full pipeline**: Use real-time or file mode

## Development

### Debugging

```bash
# Build debug version
mkdir build-desktop-debug
cd build-desktop-debug
cmake ../desktop -DCMAKE_BUILD_TYPE=Debug
cmake --build . -j$(nproc)

# Run with GDB
gdb ./translation_desktop
```

### Profiling

```bash
# Using perf (Linux)
perf record ./translation_desktop --asr-model ... --mt-model ... --tts-model ...
perf report

# Using valgrind (Linux)
valgrind --tool=callgrind ./translation_desktop --asr-model ... --mt-model ... --tts-model ...
```

## Platform-Specific Notes

### Linux

- PortAudio uses ALSA by default
- May need to install `libasound2-dev`
- Check audio permissions: `groups` (should include `audio`)

### macOS

- PortAudio uses CoreAudio
- May need to grant microphone permission in System Preferences
- Xcode command-line tools required: `xcode-select --install`

### Windows

- PortAudio uses DirectSound or WASAPI
- Requires Visual Studio or MinGW
- CMake can generate Visual Studio project files

## Next Steps

- See [BUILD.md](BUILD.md) for model download/conversion
- See [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- See [OPTIMIZATION.md](OPTIMIZATION.md) for performance tuning
