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

| Mode | Backend | Measured Latency | Quality | Method |
|------|---------|---------|---------|--------|
| **SPEED** | Marian NMT only | **49ms** | Correct Hindi | Encoder-decoder ONNX |
| **BALANCED** | NMT draft + LLM verify | **6.5s*** | Correct (with minor corrections) | Refinement-prompt speculative decoding |
| **QUALITY** | NMT draft + LLM verify | **6.5s*** | Same as Balanced | Same as Balanced (standalone LLM produces incorrect Hindi) |

*\*Full-prefill O(n²) approach; KV-cache decode would reduce to ~200-400ms.*

**Note**: Balanced and Quality mode use the same code path. Qwen3-0.6B (600M params) cannot translate Hindi from scratch — it produces incorrect output in all prompt formats (confirmed in Python float32). Both modes use the refinement prompt approach where NMT provides the draft and LLM verifies/corrects it.

#### 3a. Marian NMT (SPEED mode / draft generator)

**Purpose**: Fast phrase-level translation + draft generation for speculative decoding

**Key Features**:
- OPUS-MT EN→HI from Helsinki-NLP (ONNX Runtime)
- INT8 quantization, CPU-only
- SentencePiece tokenization with vocab.json ID mapping
- Measured: **49ms** per phrase, produces correct Hindi

#### 3b. LLM Verifier (Qwen3-0.6B via ExecuTorch)

**Purpose**: Verify and refine NMT draft translations via speculative decoding

**Key Features**:
- ExecuTorch runtime with XNNPACK + KleidiAI (SME2/NEON acceleration)
- Qwen3 chat template with pre-filled empty `<think>` block to skip reasoning chain
- HFTokenizer with PCRE2 fallback (re2 cannot compile Qwen3's lookahead regex)
- Deterministic output (argmax decoding with 1.3x repetition penalty)
- INT4 quantized (8da4w, group size 128, 4-bit embeddings)

**Model**: Qwen3-0.6B (~388 MB as .pte with INT4)
**Framework**: ExecuTorch 0.7+ with XNNPACK + KleidiAI
**Limitation**: Too small for standalone Hindi translation; serves as refinement/verification layer

#### 3c. Refinement-Prompt Speculative Decoding (BALANCED/QUALITY mode)

**Purpose**: Preserve NMT translation quality while allowing LLM-based corrections

This is the novel core of the system — a cross-architecture speculative decoding approach using a **refinement prompt** that gives the LLM the NMT draft as context:

1. **Marian NMT generates draft** (~49ms): Encoder-decoder produces correct Hindi
2. **Build refinement prompt**: "English: {text}\nDraft: {nmt_hindi}" — the LLM sees the NMT output
3. **Verify each draft token**: LLM predicts next token; if it matches the NMT draft token, accept
4. **Accept matching prefix**: The LLM tends to echo correct NMT tokens (65% acceptance)
5. **Diverge on corrections**: Where LLM disagrees, it applies minor corrections (e.g., gender agreement)
6. **Continue autoregressively**: Generate remaining tokens from the rejection point

**Key insight — the refinement prompt**: Without the NMT draft in the prompt, the LLM translates independently and produces completely different (incorrect) Hindi, yielding only 4% token acceptance. By giving the LLM the NMT draft as context, acceptance jumps to **65%** because the LLM echoes correct tokens and only diverges for genuine corrections.

**Measured acceptance rates**: 65% average across test sentences (15/23 tokens accepted for "I will go to the market")

**Optimization stack**:
- ExecuTorch XNNPACK backend with KleidiAI
- SME2/NEON acceleration on Arm
- INT4 quantization (8da4w, group size 128)
- Full-prefill (start_pos=0) for quantized model reliability
- KV-cache decode is the primary optimization target (~15x latency reduction)

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
3. **MT Input**: English phrase → NMT draft → LLM refinement → Hindi phrase
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

## Latency Breakdown (Measured, Desktop Apple M4)

| Stage | Measured Latency | Notes |
|-------|---------------|-------|
| Audio Capture | 0 ms | Hardware buffering |
| ASR Processing | 40 ms avg | Chunk-based, streaming |
| Phrase Detection | < 1 ms | Simple logic |
| MT Translation (SPEED) | **49 ms** | Marian NMT only — correct Hindi |
| MT Translation (BALANCED) | **6.5s** (full-prefill) | NMT draft + LLM speculative verify, 65% acceptance |
| MT Translation (BALANCED, projected) | **200-400 ms** | With KV-cache decode optimization |
| TTS Synthesis | 150-210 ms | Chunked, streaming |
| Audio Playback | 0 ms | Hardware buffering |
| **Total — Speed mode** | **~250 ms** | Best for real-time conversation |
| **Total — Balanced (projected)** | **~500 ms** | With KV-cache optimization |

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

1. **KV-cache decode**: Fix INT4 quantized model reliability with KV-cache to enable O(n) speculative verification (~15x latency reduction, targeting ~200-400ms)
2. **Larger LLM**: Qwen3-1.5B+ or Hindi-specialized model for standalone translation and higher acceptance rates
3. **SME2 Support**: Scalable Matrix Extension for Arm Cortex-A
4. **Model Pruning**: Remove unnecessary parameters
5. **Hardware Acceleration**: DSP/NPU offload via Android NNAPI or Qualcomm QNN
