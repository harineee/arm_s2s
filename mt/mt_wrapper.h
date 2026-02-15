/**
 * MT wrapper — three-mode translation routing:
 *   SPEED:    Marian NMT only (~49ms, correct Hindi)
 *   BALANCED: NMT draft + LLM speculative verify (65% acceptance)
 *   QUALITY:  NMT draft + LLM refinement (same path as Balanced)
 */

#ifndef MT_WRAPPER_H
#define MT_WRAPPER_H

#include <string>
#include <vector>
#include <memory>
#include "translation_mode.h"

#ifdef USE_ONNXRUNTIME
#include "marian_mt_wrapper.h"
#endif

#ifdef USE_EXECUTORCH
#include "llm_translator.h"
#endif

class MTWrapper {
public:
    MTWrapper();
    ~MTWrapper();

    // Initialize Marian NMT backend
    bool init(const std::string& model_path);

    // Initialize LLM backend (Qwen3-0.6B via ExecuTorch)
    bool init_llm(const std::string& llm_model_path,
                  const std::string& tokenizer_path);

    // Translate using the current translation mode
    std::string translate(const std::string& english_text);
    std::vector<std::string> translate_batch(const std::vector<std::string>& texts);

    // Translation mode control
    void set_translation_mode(TranslationMode mode);
    TranslationMode get_translation_mode() const { return mode_; }
    std::string get_mode_name() const;

    // Query backend availability
    bool is_llm_active() const;
    bool is_nmt_active() const;

    // Metrics (forwarded from LLM translator)
    double get_last_latency_ms() const;
    double get_acceptance_rate() const;
    int get_last_accepted_tokens() const;
    int get_last_total_draft_tokens() const;

private:
    // NMT-only translation (SPEED mode)
    std::string translate_nmt(const std::string& english_text);

    // Speculative: NMT draft + LLM verify (BALANCED mode)
    std::string translate_speculative(const std::string& english_text);

    // Full LLM autoregressive (QUALITY mode)
    std::string translate_llm(const std::string& english_text);

#ifdef USE_ONNXRUNTIME
    std::unique_ptr<MarianMTWrapper> marian_;
#endif
#ifdef USE_EXECUTORCH
    std::unique_ptr<LLMTranslator> llm_;
#endif
    bool use_placeholder_;
    bool use_llm_ = false;
    bool use_nmt_ = false;
    TranslationMode mode_ = TranslationMode::SPEED;
};

#endif // MT_WRAPPER_H
