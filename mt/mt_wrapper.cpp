/**
 * MT wrapper implementation — Marian ONNX backend only.
 * No LLM path. No Python.
 */

#include "mt_wrapper.h"
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
        return true;
    }
#endif
    use_placeholder_ = true;
    return true;
}

std::string MTWrapper::translate(const std::string& english_text) {
    if (english_text.empty()) return "";
    if (use_placeholder_) return "[HI: " + english_text + "]";

#ifdef USE_ONNXRUNTIME
    if (marian_) {
        std::string r = marian_->translate(english_text);
        if (!r.empty()) return r;
    }
#endif
    return "[HI: " + english_text + "]";
}

std::vector<std::string> MTWrapper::translate_batch(
    const std::vector<std::string>& texts) {
    std::vector<std::string> results;
    results.reserve(texts.size());
    for (const auto& t : texts) results.push_back(translate(t));
    return results;
}
