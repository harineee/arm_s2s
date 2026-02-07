#include "pipeline.h"
#include "phrase_detector.h"
#include "performance_tracker.h"
#include <iostream>
#include <chrono>
#include <thread>

#ifdef __ANDROID__
#include <sys/syscall.h>
#include <unistd.h>
#endif

Pipeline::Pipeline()
    : audio_queue_(1024)      // Buffer ~1 second at 16 kHz
    , asr_text_queue_(64)     // Text phrases
    , mt_text_queue_(64)      // Translated phrases
    , tts_audio_queue_(32)    // Audio chunks
    , running_(false)
{
}

Pipeline::~Pipeline() {
    stop();
}

bool Pipeline::init(const PipelineConfig& config) {
    config_ = config;
    
    // Initialize performance tracker
    perf_tracker_ = std::make_unique<PerformanceTracker>();
    if (!g_performance_tracker) {
        g_performance_tracker = perf_tracker_.get();
    }
    
    // Initialize ASR
    {
        PERF_START("asr_init");
        asr_ = std::make_unique<ASRWrapper>();
        if (!asr_->init(config.asr_model_path)) {
            std::cerr << "Failed to initialize ASR" << std::endl;
            return false;
        }
        asr_->set_chunk_size_ms(config.chunk_size_ms);
    }
    
    // Initialize MT (LLM)
    {
        PERF_START("mt_init");
        mt_ = std::make_unique<MTWrapper>();
        if (!mt_->init(config.mt_model_path)) {
            std::cerr << "Failed to initialize MT (LLM)" << std::endl;
            // Don't fail, will use placeholder
        }
    }
    
    // Initialize TTS
    {
        PERF_START("tts_init");
        tts_ = std::make_unique<TTSWrapper>();
        if (!tts_->init(config.tts_model_path)) {
            std::cerr << "Failed to initialize TTS" << std::endl;
            // Don't fail, will use placeholder
        }
        tts_->set_sample_rate(config.sample_rate);
    }
    
    // Initialize phrase detector
    phrase_detector_ = std::make_unique<PhraseDetector>();
    
    return true;
}

bool Pipeline::start() {
    if (running_.load()) {
        return false;
    }
    
    running_.store(true);
    
    // Start ASR thread
    asr_thread_ = std::thread([this]() { this->asr_thread_func(); });
    
    // Start MT thread
    mt_thread_ = std::thread([this]() { this->mt_thread_func(); });
    
    // Start TTS thread
    tts_thread_ = std::thread([this]() { this->tts_thread_func(); });
    
    // Set thread priorities (Android-specific)
#ifdef __ANDROID__
    if (config_.pin_threads) {
        // Note: Requires root or appropriate permissions
        // For production, use Android's native thread priority APIs
    }
#endif
    
    return true;
}

void Pipeline::stop() {
    if (!running_.load()) {
        return;
    }
    
    running_.store(false);
    
    // Join threads
    if (asr_thread_.joinable()) {
        asr_thread_.join();
    }
    if (mt_thread_.joinable()) {
        mt_thread_.join();
    }
    if (tts_thread_.joinable()) {
        tts_thread_.join();
    }
    
    // Print performance summary
    if (perf_tracker_) {
        perf_tracker_->print_summary();
    }
}

void Pipeline::enable_performance_tracking(bool enable) {
    if (enable && !perf_tracker_) {
        perf_tracker_ = std::make_unique<PerformanceTracker>();
        g_performance_tracker = perf_tracker_.get();
    } else if (!enable) {
        perf_tracker_.reset();
        if (g_performance_tracker == perf_tracker_.get()) {
            g_performance_tracker = nullptr;
        }
    }
}

void Pipeline::push_audio(const float* samples, size_t num_samples) {
    std::vector<float> audio_chunk(samples, samples + num_samples);
    audio_queue_.push(audio_chunk);
}

bool Pipeline::pop_audio(std::vector<float>& audio_samples) {
    return tts_audio_queue_.pop(audio_samples);
}

void Pipeline::asr_thread_func() {
    std::vector<float> audio_chunk;
    
    while (running_.load()) {
        if (audio_queue_.pop(audio_chunk)) {
            PERF_START("asr_process");
            
            // Process audio with ASR
            std::string partial_text = asr_->process_chunk(
                audio_chunk.data(), 
                audio_chunk.size());
            
            if (!partial_text.empty() && partial_text != "[BLANK_AUDIO]") {
                current_english_ = partial_text;
                
                // Check for phrase boundaries
                std::vector<PhraseBoundary> boundaries;
                if (phrase_detector_->process_text(partial_text, boundaries)) {
                    // Push complete phrases to MT queue
                    for (const auto& boundary : boundaries) {
                        if (!boundary.text.empty() && boundary.text != "[BLANK_AUDIO]") {
                            asr_text_queue_.push(boundary.text);
                        }
                    }
                }
            }
        } else {
            // Small sleep to avoid busy-waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void Pipeline::mt_thread_func() {
    std::string english_text;
    
    while (running_.load()) {
        if (asr_text_queue_.pop(english_text)) {
            PERF_START("mt_translate");
            
            // Translate to Hindi using LLM
            std::string hindi_text = mt_->translate(english_text);
            
            if (!hindi_text.empty()) {
                current_hindi_ = hindi_text;
                // Push to TTS queue
                mt_text_queue_.push(hindi_text);
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void Pipeline::tts_thread_func() {
    std::string hindi_text;
    size_t chunk_samples = (config_.sample_rate * 500) / 1000; // 500 ms chunks
    
    while (running_.load()) {
        if (mt_text_queue_.pop(hindi_text)) {
            PERF_START("tts_synthesize");
            
            // Synthesize audio in chunks
            std::vector<float> audio_chunk = tts_->synthesize_chunk(
                hindi_text, 
                chunk_samples);
            
            if (!audio_chunk.empty()) {
                // Push to playback queue
                tts_audio_queue_.push(audio_chunk);
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void Pipeline::flush_asr() {
    // Flush ASR to process remaining audio
    if (asr_) {
        std::string final_text = asr_->flush();
        if (!final_text.empty() && final_text != "[BLANK_AUDIO]") {
            current_english_ = final_text;
            // Push to MT queue so file mode gets a translation (phrase detector may not have emitted during streaming)
            asr_text_queue_.push(final_text);
        }
    }
}
