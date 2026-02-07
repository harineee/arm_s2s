# Optimization Documentation

## Arm NEON Optimization

### Overview

NEON (Advanced SIMD) is Arm's SIMD (Single Instruction Multiple Data) extension that enables parallel processing of multiple data elements in a single instruction. This is critical for ML inference performance on mobile CPUs.

### NEON Usage in Components

#### 1. ASR (whisper.cpp)

whisper.cpp already includes NEON optimizations for:
- Matrix multiplications (convolution layers)
- Activation functions (ReLU, GELU)
- Attention mechanisms
- Audio preprocessing (FFT operations)

**Verification**:
```bash
# Check if NEON is enabled
grep -r "NEON\|neon" asr/whisper_cpp/
```

#### 2. MT (Marian NMT)

Marian uses NEON for:
- Embedding lookups
- Matrix-vector multiplications
- Softmax computations
- Layer normalization

**Manual Optimization** (if needed):
```cpp
#include <arm_neon.h>

// Example: Vectorized addition
void add_vectors_neon(float* a, float* b, float* result, size_t len) {
    size_t i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t vresult = vaddq_f32(va, vb);
        vst1q_f32(&result[i], vresult);
    }
    // Handle remainder
    for (; i < len; i++) {
        result[i] = a[i] + b[i];
    }
}
```

#### 3. TTS (VITS)

VITS benefits from NEON for:
- Convolution operations
- Upsampling/downsampling
- Vocoder operations (HiFiGAN)

### Compiler Flags

**CMake Configuration**:
```cmake
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a+simd")
endif()
```

**GCC/Clang Flags**:
- `-march=armv8-a+simd`: Enable NEON
- `-O3`: Maximum optimization
- `-ffast-math`: Fast math operations (may affect precision)
- `-funroll-loops`: Loop unrolling

### Verification

**Check NEON Support**:
```cpp
#include <arm_neon.h>
#include <iostream>

bool check_neon_support() {
    // NEON is always available on arm64-v8a
    #ifdef __aarch64__
        return true;
    #else
        return false;
    #endif
}
```

## Quantization

### INT8 Quantization

**Benefits**:
- 4x reduction in model size
- 2-4x faster inference
- Lower memory bandwidth

**Implementation**:
1. **Post-training Quantization**: Convert FP32 → INT8
2. **Quantization-aware Training**: Train with quantization (better accuracy)

**Tools**:
- ONNX Runtime quantization
- TensorFlow Lite quantization
- Custom quantization (per-channel or per-tensor)

### INT4 Quantization (Future)

**Benefits**:
- 8x reduction in model size
- Further speedup (with custom kernels)

**Challenges**:
- Requires custom inference kernels
- May need mixed precision (INT4 weights, INT8 activations)

## Memory Optimization

### Buffer Reuse

```cpp
// Pre-allocate buffers
class BufferPool {
    std::vector<std::vector<float>> audio_buffers;
    std::vector<std::vector<int>> text_buffers;
    
public:
    std::vector<float>& get_audio_buffer(size_t size) {
        // Reuse or allocate
        for (auto& buf : audio_buffers) {
            if (buf.size() >= size) {
                buf.resize(size);
                return buf;
            }
        }
        audio_buffers.emplace_back(size);
        return audio_buffers.back();
    }
};
```

### Cache-Friendly Data Layout

```cpp
// Structure of Arrays (SoA) for better cache usage
struct AudioBuffer {
    std::vector<float> samples;  // Contiguous memory
    // Instead of Array of Structures (AoS)
};
```

### Memory Alignment

```cpp
// Align to cache line (64 bytes)
alignas(64) float audio_buffer[16000];
```

## Threading Optimization

### Thread Affinity

**Android-specific**:
```cpp
#include <sys/syscall.h>
#include <unistd.h>

void set_thread_affinity(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}
```

### Thread Priorities

```cpp
// Set high priority for ASR thread
#include <sys/resource.h>

void set_thread_priority(int priority) {
    setpriority(PRIO_PROCESS, 0, priority);
}
```

**Android Java**:
```java
Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO);
```

### Lock-Free Queues

See `pipeline/lockfree_queue.h` for SPSC queue implementation.

**Benefits**:
- No mutex contention
- Lower latency
- Better cache behavior

## Pipeline Overlap

### Parallel Execution

```
Time →
ASR:  [====]
MT:        [==]
TTS:          [====]
Play:            [====]
```

**Key**: Start TTS as soon as MT produces text, don't wait for full sentence.

### Chunked Processing

- **ASR**: Process 40-80 ms chunks
- **MT**: Translate phrases (2-5 words)
- **TTS**: Synthesize 0.5-1.0 s chunks

## Model-Specific Optimizations

### whisper.cpp

**Configuration**:
```cpp
whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
params.n_threads = 2;  // Optimize for mobile (2-4 cores)
params.temperature = 0.0f;  // Deterministic
params.no_context = false;  // Use context for better accuracy
```

**Model Selection**:
- `tiny.en`: Smallest, fastest (~39M parameters)
- `base.en`: Better accuracy (~74M parameters)
- Use INT8 quantized versions

### Marian NMT

**Optimization Strategies**:
1. **Beam Size**: Use beam=1 (greedy) for lowest latency
2. **Length Penalty**: Disable or reduce for faster decoding
3. **Batch Size**: Process single phrases (batch=1)

### VITS

**Chunked Synthesis**:
- Synthesize in 0.5-1.0 s chunks
- Start playback immediately
- Overlap synthesis and playback

## Profiling and Measurement

### Latency Measurement

```cpp
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();
// ... inference ...
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
```

### Android Profiling

```bash
# Use Android Studio Profiler
# Or command line:
adb shell dumpsys gfxinfo com.armm.translation
```

### Perf Tools

```bash
# On device (requires root)
perf record -g ./your_app
perf report
```

## SME2 Considerations (Future)

SME2 (Scalable Matrix Extension) is Arm's next-generation SIMD extension for matrix operations.

**When Available**:
- Use for large matrix multiplications
- Optimize transformer attention mechanisms
- Accelerate convolution operations

**Detection**:
```cpp
#include <sys/auxv.h>

bool has_sme2() {
    unsigned long hwcap = getauxval(AT_HWCAP);
    // Check for SME2 capability
    return (hwcap & HWCAP2_SME2) != 0;
}
```

## Best Practices Summary

1. **Enable NEON**: Always use `-march=armv8-a+simd`
2. **Use INT8**: Quantize models for 2-4x speedup
3. **Reuse Buffers**: Avoid allocations in hot paths
4. **Lock-Free**: Use SPSC queues for inter-thread communication
5. **Chunked Processing**: Process in small chunks for lower latency
6. **Overlap Stages**: Don't wait for full completion before next stage
7. **Profile First**: Measure before optimizing
8. **Cache Awareness**: Structure data for cache-friendly access

## References

- [Arm NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/simd-isas/neon)
- [whisper.cpp Optimization](https://github.com/ggerganov/whisper.cpp#optimization)
- [Android NDK Optimization Guide](https://developer.android.com/ndk/guides/cpp-support)
