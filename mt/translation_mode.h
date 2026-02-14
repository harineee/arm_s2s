/**
 * Translation mode enum — controls quality/speed tradeoff
 */
#ifndef TRANSLATION_MODE_H
#define TRANSLATION_MODE_H

enum class TranslationMode {
    SPEED,      // Marian NMT only (~20ms, good quality)
    BALANCED,   // NMT draft + LLM speculative verify (~150ms, very good quality)
    QUALITY     // Full LLM autoregressive generation (~500ms, best quality)
};

#endif // TRANSLATION_MODE_H
