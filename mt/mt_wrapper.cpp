/**
 * MT wrapper — three-mode translation routing with speculative decoding.
 *
 * SPEED:    Marian NMT only (~49ms, correct Hindi)
 * BALANCED: NMT draft + LLM speculative verification (65% acceptance)
 * QUALITY:  NMT draft + LLM refinement (same as Balanced — standalone
 *           LLM produces incorrect Hindi at 0.6B param scale)
 */

#include "mt_wrapper.h"
#include "performance_tracker.h"
#include <iostream>
#include <fstream>

MTWrapper::MTWrapper()
    : use_placeholder_(true)
{
#ifdef USE_ONNXRUNTIME
    marian_ = std::make_unique<MarianMTWrapper>();
#endif
}

MTWrapper::~MTWrapper() = default;

bool MTWrapper::init(const std::string& model_path) {
    std::ifstream f(model_path);
    if (!f.good()) {
        std::cerr << "MT model not found: " << model_path << " (placeholder mode)" << std::endl;
        use_placeholder_ = true;
        return true;
    }

#ifdef USE_ONNXRUNTIME
    if (marian_ && marian_->init(model_path)) {
        use_placeholder_ = false;
        use_nmt_ = true;
        return true;
    }
#endif
    use_placeholder_ = true;
    return true;
}

bool MTWrapper::init_llm(const std::string& llm_model_path,
                         const std::string& tokenizer_path) {
#ifdef USE_EXECUTORCH
    llm_ = std::make_unique<LLMTranslator>();
    if (llm_->init(llm_model_path, tokenizer_path)) {
        use_llm_ = true;
        std::cout << "MT: LLM translation enabled (Qwen3-0.6B)" << std::endl;
        return true;
    }
    std::cerr << "MT: LLM init failed, will use NMT fallback" << std::endl;
    llm_.reset();
#else
    (void)llm_model_path;
    (void)tokenizer_path;
    std::cerr << "MT: ExecuTorch not compiled in, LLM unavailable" << std::endl;
#endif
    return false;
}

// ============================================================
// Mode control
// ============================================================

void MTWrapper::set_translation_mode(TranslationMode mode) {
    mode_ = mode;
    std::cout << "MT: Translation mode set to " << get_mode_name() << std::endl;
}

std::string MTWrapper::get_mode_name() const {
    switch (mode_) {
        case TranslationMode::SPEED:    return "Speed (NMT only)";
        case TranslationMode::BALANCED: return "Balanced (NMT + LLM speculative)";
        case TranslationMode::QUALITY:  return "Quality (NMT + LLM refinement)";
    }
    return "Unknown";
}

bool MTWrapper::is_llm_active() const {
#ifdef USE_EXECUTORCH
    return use_llm_ && llm_ && llm_->is_initialized();
#else
    return false;
#endif
}

bool MTWrapper::is_nmt_active() const {
#ifdef USE_ONNXRUNTIME
    return use_nmt_ && marian_ && !use_placeholder_;
#else
    return false;
#endif
}

// ============================================================
// Metrics (forwarded from LLM translator)
// ============================================================

double MTWrapper::get_last_latency_ms() const {
#ifdef USE_EXECUTORCH
    if (llm_) return llm_->get_last_latency_ms();
#endif
    return 0.0;
}

double MTWrapper::get_acceptance_rate() const {
#ifdef USE_EXECUTORCH
    if (llm_) return llm_->get_acceptance_rate();
#endif
    return 0.0;
}

int MTWrapper::get_last_accepted_tokens() const {
#ifdef USE_EXECUTORCH
    if (llm_) return llm_->get_last_accepted_tokens();
#endif
    return 0;
}

int MTWrapper::get_last_total_draft_tokens() const {
#ifdef USE_EXECUTORCH
    if (llm_) return llm_->get_last_total_draft_tokens();
#endif
    return 0;
}

// ============================================================
// Main translate — routes based on current mode
// ============================================================

std::string MTWrapper::translate(const std::string& english_text) {
    if (english_text.empty()) return "";

    switch (mode_) {
        case TranslationMode::SPEED:
            return translate_nmt(english_text);

        case TranslationMode::BALANCED:
            // Speculative requires both NMT and LLM
            if (is_llm_active() && (is_nmt_active() || !use_placeholder_)) {
                return translate_speculative(english_text);
            }
            // Fall through to best available
            if (is_llm_active()) return translate_llm(english_text);
            return translate_nmt(english_text);

        case TranslationMode::QUALITY:
            // Quality uses the same refinement approach as Balanced.
            // Standalone LLM (0.6B) produces incorrect Hindi; the
            // refinement prompt with NMT draft gives correct output.
            if (is_llm_active() && (is_nmt_active() || !use_placeholder_)) {
                return translate_speculative(english_text);
            }
            if (is_llm_active()) return translate_llm(english_text);
            return translate_nmt(english_text);
    }

    return translate_nmt(english_text);
}

// ============================================================
// SPEED mode — NMT only
// ============================================================

std::string MTWrapper::translate_nmt(const std::string& english_text) {
#ifdef USE_ONNXRUNTIME
    if (use_nmt_ && marian_) {
        PERF_START("nmt_translate");
        std::string result = marian_->translate(english_text);
        if (!result.empty()) return result;
    }
#endif
    if (use_placeholder_) return "[HI: " + english_text + "]";
    return "[HI: " + english_text + "]";
}

// ============================================================
// BALANCED mode — NMT draft + LLM speculative verification
// ============================================================

std::string MTWrapper::translate_speculative(const std::string& english_text) {
    // Step 1: Get NMT draft (~20ms)
    std::string nmt_draft;
#ifdef USE_ONNXRUNTIME
    if (use_nmt_ && marian_) {
        PERF_START("nmt_draft");
        nmt_draft = marian_->translate(english_text);
    }
#endif
    if (nmt_draft.empty()) {
        // No NMT available — fall back to full LLM
        return translate_llm(english_text);
    }

    // Step 2: LLM speculative verification (~80-150ms)
#ifdef USE_EXECUTORCH
    if (use_llm_ && llm_ && llm_->is_initialized()) {
        PERF_START("llm_speculative");
        std::string result = llm_->speculative_translate(english_text, nmt_draft);
        if (!result.empty()) return result;
        // Speculative failed — return NMT draft as-is
        std::cerr << "MT: Speculative verification failed, using NMT draft" << std::endl;
    }
#endif

    return nmt_draft;
}

// ============================================================
// QUALITY mode — Full LLM autoregressive generation
// ============================================================

std::string MTWrapper::translate_llm(const std::string& english_text) {
#ifdef USE_EXECUTORCH
    if (use_llm_ && llm_ && llm_->is_initialized()) {
        PERF_START("llm_translate");
        std::string result = llm_->translate(english_text);
        if (!result.empty()) return result;
        std::cerr << "MT: LLM translate failed, falling back to NMT" << std::endl;
    }
#endif
    // Fall back to NMT
    return translate_nmt(english_text);
}

// ============================================================
// Batch translation
// ============================================================

std::vector<std::string> MTWrapper::translate_batch(
    const std::vector<std::string>& texts) {
    std::vector<std::string> results;
    results.reserve(texts.size());
    for (const auto& t : texts) results.push_back(translate(t));
    return results;
}
