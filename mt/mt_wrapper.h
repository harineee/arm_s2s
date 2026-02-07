/**
 * MT wrapper — routes to Marian ONNX backend.
 * Clean interface: init() + translate().
 */

#ifndef MT_WRAPPER_H
#define MT_WRAPPER_H

#include <string>
#include <vector>
#include <memory>

#ifdef USE_ONNXRUNTIME
#include "marian_mt_wrapper.h"
#endif

class MTWrapper {
public:
    MTWrapper();
    ~MTWrapper();

    bool init(const std::string& model_path);
    std::string translate(const std::string& english_text);
    std::vector<std::string> translate_batch(const std::vector<std::string>& texts);

private:
#ifdef USE_ONNXRUNTIME
    std::unique_ptr<MarianMTWrapper> marian_;
#endif
    bool use_placeholder_;
};

#endif // MT_WRAPPER_H
