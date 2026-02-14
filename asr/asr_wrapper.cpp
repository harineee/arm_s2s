#include "asr_wrapper.h"
#include "whisper.h" // whisper.cpp header

#include <cstring>
#include <algorithm>
#include <iostream>

ASRWrapper::ASRWrapper() : chunk_size_ms_(80) {
    ctx_ = nullptr;
}

ASRWrapper::~ASRWrapper() {
    if (ctx_) {
        whisper_free(ctx_);
    }
}

bool ASRWrapper::init(const std::string& model_path) {
    // Load whisper model
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false; // CPU only
    
    whisper_context* ctx = whisper_init_from_file_with_params(
        model_path.c_str(), cparams);
    
    if (!ctx) {
        return false;
    }
    
    ctx_ = ctx;
    return true;
}

std::string ASRWrapper::process_chunk(const float* audio_samples, 
                                      size_t num_samples) {
    if (!ctx_) {
        return "";
    }
    
    // Accumulate samples
    size_t old_size = audio_buffer_.size();
    audio_buffer_.resize(old_size + num_samples);
    std::memcpy(audio_buffer_.data() + old_size, 
                audio_samples, 
                num_samples * sizeof(float));
    
    // Process when we have enough samples for a chunk
    // Minimum 1 second of audio for whisper to work properly
    size_t min_samples = SAMPLE_RATE * 1; // 1 second minimum
    size_t chunk_samples = std::max(static_cast<size_t>((chunk_size_ms_ * SAMPLE_RATE) / 1000), min_samples);
    
    // Cap buffer to avoid internal limits / OOM (e.g. 30 s at 16 kHz)
    static const size_t kMaxSamples = SAMPLE_RATE * 30;
    if (audio_buffer_.size() > kMaxSamples) {
        audio_buffer_.erase(audio_buffer_.begin(),
                            audio_buffer_.begin() + (audio_buffer_.size() - kMaxSamples));
    }

    if (audio_buffer_.size() >= chunk_samples) {
        whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        params.print_progress = false;
        params.print_special = false;
        params.print_realtime = false;
        params.translate = false; // English ASR, not translation
        params.language = "en";
        params.n_threads = 2; // Optimize for mobile CPU
        params.offset_ms = 0;
        params.no_context = false;
        params.single_segment = false;
        params.suppress_blank = true;
        params.temperature = 0.0f; // Deterministic
        
        size_t n = audio_buffer_.size();
        if (n == 0) return partial_text_;
        // Run inference
        int result = whisper_full(ctx_, params, 
                                 audio_buffer_.data(), 
                                 static_cast<int>(n));
        
        if (result == 0) {
            // Extract text from segments
            int n_segments = whisper_full_n_segments(ctx_);
            std::string text;
            
            for (int i = 0; i < n_segments; i++) {
                const char* segment_text = whisper_full_get_segment_text(ctx_, i);
                if (segment_text) {
                    std::string seg_str(segment_text);
                    // Trim whitespace
                    while (!seg_str.empty() && (std::isspace(seg_str[0]) || seg_str[0] == '\0')) {
                        seg_str.erase(0, 1);
                    }
                    while (!seg_str.empty() && (std::isspace(seg_str.back()) || seg_str.back() == '\0')) {
                        seg_str.pop_back();
                    }
                    if (!seg_str.empty() && seg_str != "[BLANK_AUDIO]") {
                        if (!text.empty()) text += " ";
                        text += seg_str;
                    }
                }
            }
            
            if (!text.empty()) {
                partial_text_ = text;
            } else {
                // If no text but we processed audio, keep previous text
            }
            
            // Keep last portion of buffer for context (overlap)
            size_t keep_samples = chunk_samples / 2;
            if (audio_buffer_.size() > keep_samples) {
                std::vector<float> keep_buffer(
                    audio_buffer_.end() - keep_samples,
                    audio_buffer_.end());
                audio_buffer_ = std::move(keep_buffer);
            } else {
                audio_buffer_.clear();
            }
        }
    }
    
    return partial_text_;
}

std::string ASRWrapper::flush() {
    // Force process remaining audio buffer even if it's smaller than chunk size
    if (!ctx_ || audio_buffer_.empty()) {
        return partial_text_;
    }
    size_t n = audio_buffer_.size();
    static const size_t kMaxSamples = SAMPLE_RATE * 30;
    if (n > kMaxSamples) n = kMaxSamples;
    
    // Process remaining audio (even if less than minimum)
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_progress = false;
    params.print_special = false;
    params.print_realtime = false;
    params.translate = false;
    params.language = "en";
    params.n_threads = 2;
    params.offset_ms = 0;
    params.no_context = false;
    params.single_segment = false;
    params.suppress_blank = true;
    params.temperature = 0.0f;
    
    int result = whisper_full(ctx_, params, 
                             audio_buffer_.data(), 
                             static_cast<int>(n));
    
    if (result == 0) {
        int n_segments = whisper_full_n_segments(ctx_);
        std::string text;
        
        for (int i = 0; i < n_segments; i++) {
            const char* segment_text = whisper_full_get_segment_text(ctx_, i);
            if (segment_text) {
                std::string seg_str(segment_text);
                while (!seg_str.empty() && (std::isspace(seg_str[0]) || seg_str[0] == '\0')) {
                    seg_str.erase(0, 1);
                }
                while (!seg_str.empty() && (std::isspace(seg_str.back()) || seg_str.back() == '\0')) {
                    seg_str.pop_back();
                }
                if (!seg_str.empty() && seg_str != "[BLANK_AUDIO]") {
                    if (!text.empty()) text += " ";
                    text += seg_str;
                }
            }
        }
        
        if (!text.empty()) {
            partial_text_ = text;
        }
    }
    
    audio_buffer_.clear();
    return partial_text_;
}

void ASRWrapper::reset() {
    if (ctx_) {
        whisper_reset_timings(ctx_);
    }
    partial_text_.clear();
    audio_buffer_.clear();
}
