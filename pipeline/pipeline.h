/**
 * Main pipeline orchestrator
 * Manages parallel execution of ASR → MT → TTS → Playback
 */

#ifndef PIPELINE_H
#define PIPELINE_H

#include "asr_wrapper.h"
#include "mt_wrapper.h"
#include "tts_wrapper.h"
#include "phrase_detector.h"
#include "lockfree_queue.h"
#include "performance_tracker.h"

#include <thread>
#include <atomic>
#include <memory>
#include <vector>
#include <string>

struct PipelineConfig {
    std::string asr_model_path;
    std::string mt_model_path;
    std::string tts_model_path;
    
    int sample_rate = 16000;
    int chunk_size_ms = 80;
    
    // Threading
    bool pin_threads = true;
    int asr_thread_priority = 10;
    int mt_thread_priority = 5;
    int tts_thread_priority = 5;
};

class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    // Initialize pipeline with models
    bool init(const PipelineConfig& config);

    // Start pipeline (spawns threads)
    bool start();

    // Stop pipeline (joins threads)
    void stop();

    // Push audio samples (called from audio capture thread)
    void push_audio(const float* samples, size_t num_samples);

    // Get synthesized audio (called from playback thread)
    bool pop_audio(std::vector<float>& audio_samples);

    // Flush ASR to process remaining audio
    void flush_asr();

    // Get current status
    bool is_running() const { return running_; }
    std::string get_current_english() const { return current_english_; }
    std::string get_current_hindi() const { return current_hindi_; }
    
    // Performance tracking
    PerformanceTracker* get_performance_tracker() { return perf_tracker_.get(); }
    void enable_performance_tracking(bool enable);

private:
    // Thread functions
    void asr_thread_func();
    void mt_thread_func();
    void tts_thread_func();

    // Components
    std::unique_ptr<ASRWrapper> asr_;
    std::unique_ptr<MTWrapper> mt_;
    std::unique_ptr<TTSWrapper> tts_;
    std::unique_ptr<PhraseDetector> phrase_detector_;
    std::unique_ptr<PerformanceTracker> perf_tracker_;

    // Queues (SPSC - Single Producer Single Consumer)
    LockFreeQueue<std::vector<float>> audio_queue_;      // Audio → ASR
    LockFreeQueue<std::string> asr_text_queue_;          // ASR → MT
    LockFreeQueue<std::string> mt_text_queue_;           // MT → TTS
    LockFreeQueue<std::vector<float>> tts_audio_queue_;  // TTS → Playback

    // Threads
    std::thread asr_thread_;
    std::thread mt_thread_;
    std::thread tts_thread_;

    // State
    std::atomic<bool> running_;
    std::string current_english_;
    std::string current_hindi_;
    
    PipelineConfig config_;
};

#endif // PIPELINE_H
