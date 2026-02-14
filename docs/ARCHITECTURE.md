# Architecture Documentation

## System Overview

The speech-to-speech translation system is designed as a modular, parallel pipeline optimized for low-latency on-device inference on Arm-based Android devices.

## Pipeline Architecture

```
┌─────────────┐
│ Microphone  │ (16 kHz PCM, mono)
│   Capture   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Thread 1: Audio Capture + ASR      │
│  ┌──────────┐      ┌──────────────┐ │
│  │ Audio    │─────▶│ whisper.cpp  │ │
│  │ Buffer   │      │ (Streaming)  │ │
│  └──────────┘      └──────┬───────┘ │
└───────────────────────────┼─────────┘
                            │
                            ▼ English Text
┌─────────────────────────────────────┐
│  Phrase Boundary Detection          │
│  ┌──────────────────────────────┐   │
│  │ • Pause > 150 ms             │   │
│  │ • Word count ≥ 2-5           │   │
│  │ • Punctuation detection       │   │
│  └──────────┬───────────────────┘   │
└─────────────┼───────────────────────┘
              │
              ▼ Phrase Boundary
┌─────────────────────────────────────┐
│  Thread 2: Machine Translation      │
│  ┌──────────┐      ┌──────────────┐ │
│  │ English  │─────▶│ Qwen3-0.6B   │ │
│  │ Phrase   │      │ (ExecuTorch) │ │
│  │          │      │ + KleidiAI   │ │
│  │          │      ├──────────────┤ │
│  │          │      │ Marian NMT   │ │
│  │          │      │ (Fallback)   │ │
│  └──────────┘      └──────┬───────┘ │
└───────────────────────────┼─────────┘
                            │
                            ▼ Hindi Text
┌─────────────────────────────────────┐
│  Thread 3: Text-to-Speech           │
│  ┌──────────┐      ┌──────────────┐ │
│  │ Hindi    │─────▶│ VITS         │ │
│  │ Text     │      │ (Chunked)    │ │
│  └──────────┘      └──────┬───────┘ │
└───────────────────────────┼─────────┘
                            │
                            ▼ Audio (16 kHz PCM)
┌─────────────────────────────────────┐
│  Thread 4: Audio Playback           │
│  ┌──────────┐      ┌──────────────┐ │
│  │ Audio    │─────▶│ AudioTrack   │ │
│  │ Buffer   │      │ (Android)    │ │
│  └──────────┘      └──────────────┘ │
└─────────────────────────────────────┘
```

## Component Details

### 1. ASR Module (whisper.cpp)

**Purpose**: Convert English speech to text in real-time

**Key Features**:
- Streaming inference with chunk-based processing
- Greedy decoding (beam=1) for low latency
- INT8 quantization for reduced memory and faster inference
- NEON-optimized operations

**Implementation**:
- Uses whisper.cpp's streaming API
- Processes 40-80 ms audio chunks
- Maintains context buffer for continuity
- Emits partial results continuously

**Latency Target**: 120-250 ms per chunk

### 2. Phrase Boundary Detection

**Purpose**: Determine when to trigger translation (not waiting for full sentences)

**Detection Criteria**:
1. **Pause Detection**: No text update for > 150 ms
2. **Word Count**: Accumulated text has 2-5 words
3. **Punctuation**: Presence of sentence-ending punctuation
4. **Forced Boundary**: Max word count reached (prevents long delays)

**Rationale**: 
- Full sentences can be 2-5 seconds long
- Phrase-level translation reduces latency
- Natural speech has pauses that indicate phrase boundaries

### 3. MT Module — Three Adaptive Translation Modes

The MT module supports three translation modes, selectable at runtime:

| Mode | Backend | Latency | Quality | Method |
|------|---------|---------|---------|--------|
| **SPEED** | Marian NMT only | ~20ms | Good | Encoder-decoder ONNX |
| **BALANCED** | NMT draft + LLM verify | ~150ms | Very good | Speculative decoding |
| **QUALITY** | Full LLM autoregressive | ~500ms | Best | Token-by-token generation |

#### 3a. Marian NMT (SPEED mode / draft generator)

**Purpose**: Fast phrase-level translation, also generates draft for speculative mode

**Key Features**:
- OPUS-MT EN→HI from Helsinki-NLP (ONNX Runtime)
- INT8 quantization, CPU-only
- ~20ms per phrase

#### 3b. LLM Translation (Qwen3-0.6B via ExecuTorch)

**Purpose**: High-quality LLM translation with two inference modes

**Key Features**:
- ExecuTorch runtime with XNNPACK + KleidiAI (SME2/NEON acceleration)
- `/no_think` mode for fast, direct translation without reasoning chains
- Deterministic output (argmax decoding)
- INT4 quantized (8da4w, group size 128, 4-bit embeddings)

**Model**: Qwen3-0.6B-Instruct (~388 MB as .pte with INT4)
**Framework**: ExecuTorch 0.7+ with XNNPACK + KleidiAI

#### 3c. NMT-Accelerated Speculative Decoding (BALANCED mode)

**Purpose**: Near-LLM quality at a fraction of the latency

This is the novel core of the system — a cross-architecture speculative decoding approach where an encoder-decoder NMT model drafts for a decoder-only LLM:

1. **Marian NMT generates draft** (~20ms): Fast encoder-decoder produces Hindi translation
2. **Tokenize and concatenate**: Prompt tokens + draft tokens into one sequence
3. **Single LLM forward pass** (~80ms): LLM processes the entire sequence in parallel
4. **Token-by-token verification**: Compare LLM's argmax prediction at each position against the draft token
5. **Accept up to first rejection**: All matching prefix tokens are kept
6. **Regenerate remainder**: LLM generates replacement tokens autoregressively from the rejection point

**Key insight**: Unlike standard speculative decoding (small LLM drafts for large LLM), we use a fundamentally different architecture (encoder-decoder NMT) as the drafter. The NMT's ~70% token acceptance rate means most phrases need only 1 LLM forward pass instead of N.

**Optimization stack**:
- ExecuTorch XNNPACK backend with KleidiAI
- SME2/NEON acceleration on Arm
- INT4 quantization (8da4w, group size 128)
- `/no_think` to disable Qwen3 reasoning chain

### 4. TTS Module (VITS)

**Purpose**: Synthesize Hindi text to speech

**Key Features**:
- Chunked synthesis (0.5-1.0 s audio chunks)
- Streaming playback (start before full synthesis)
- NEON-optimized inference
- 16 kHz output sample rate

**Implementation Options**:
1. **VITS Native**: Convert PyTorch model to ONNX
2. **FastSpeech2 + HiFiGAN**: Alternative pipeline
3. **ONNX Runtime**: Run converted models

**Model**: Hindi VITS from Indic-TTS

**Latency Target**: 80-150 ms per chunk

### 5. Pipeline Orchestration

**Threading Model**:
- **Thread 1**: Audio capture + ASR (highest priority)
- **Thread 2**: MT (medium priority)
- **Thread 3**: TTS (medium priority)
- **Thread 4**: Audio playback (system-managed)

**Communication**:
- Lock-free SPSC (Single Producer Single Consumer) queues
- Cache-line aligned atomic operations
- Minimal memory copies

**Synchronization**:
- No mutexes (lock-free design)
- Atomic flags for start/stop
- Queue-based message passing

## Data Flow

### Audio Flow
1. **Capture**: Android AudioRecord → float32[] (16 kHz)
2. **ASR Processing**: float32[] → whisper.cpp → text
3. **Translation**: text → Marian NMT → Hindi text
4. **Synthesis**: Hindi text → VITS → float32[] (16 kHz)
5. **Playback**: float32[] → Android AudioTrack → Speaker

### Text Flow
1. **ASR Output**: Partial English text (continuously updated)
2. **Phrase Detection**: Extract phrases on boundaries
3. **MT Input**: English phrase → Hindi phrase
4. **TTS Input**: Hindi phrase → Audio chunks

## Memory Management

### Buffer Sizes
- **Audio Queue**: ~1 second buffer (16,000 samples)
- **Text Queues**: 64 entries (phrases)
- **Audio Output Queue**: 32 chunks (~16 seconds total)

### Memory Optimization
- Reuse buffers where possible
- Pre-allocate fixed-size buffers
- Avoid dynamic allocation in hot paths
- Use stack allocation for small buffers

## Latency Breakdown (Target)

| Stage | Target Latency | Notes |
|-------|---------------|-------|
| Audio Capture | 0 ms | Hardware buffering |
| ASR Processing | 120-250 ms | Chunk-based, streaming |
| Phrase Detection | < 1 ms | Simple logic |
| MT Translation (SPEED) | 10-30 ms | Marian NMT only |
| MT Translation (BALANCED) | 80-150 ms | NMT draft + LLM speculative verify |
| MT Translation (QUALITY) | 300-800 ms | Full LLM autoregressive |
| TTS Synthesis | 80-150 ms | Chunked, streaming |
| Audio Playback | 0 ms | Hardware buffering |
| **Total (first audio)** | **< 500 ms** | Overlapped pipeline |

## Error Handling

### Component Failures
- **ASR Failure**: Pipeline stops, error reported to UI
- **MT Failure**: Skip translation, show English text
- **TTS Failure**: Skip synthesis, show Hindi text only

### Recovery Strategies
- Model reload on failure
- Graceful degradation (text-only mode)
- User notification of errors

## Extension Points

### Adding New Languages
1. Add MT model for new language pair
2. Add TTS model for target language
3. Update configuration

### Model Swapping
- Runtime model loading
- A/B testing support
- Model versioning

### Performance Monitoring
- Latency metrics per stage
- Queue depth monitoring
- Memory usage tracking

## Security Considerations

### Model Protection
- Models stored in app's private directory
- Optional encryption for sensitive models
- Integrity verification

### Privacy
- All processing on-device (no cloud)
- No data logging (unless explicitly enabled)
- User data never leaves device

## Future Optimizations

1. **SME2 Support**: Scalable Matrix Extension for Arm
2. **INT4 Quantization**: Further reduce model size
3. **Model Pruning**: Remove unnecessary parameters
4. **Custom Kernels**: Hand-optimized NEON code
5. **Hardware Acceleration**: DSP/NPU if available
