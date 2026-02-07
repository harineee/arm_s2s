package com.armm.translation;

import android.util.Log;

/**
 * Java wrapper for native pipeline.
 * Provides type-safe interface to JNI functions.
 *
 * Loads shared libraries in dependency order:
 *   1. libonnxruntime.so (ONNX Runtime for MT + TTS inference)
 *   2. libnative-lib.so  (JNI bridge to C++ pipeline)
 */
public class NativePipeline {
    private static final String TAG = "NativePipeline";
    private static boolean librariesLoaded = false;

    static {
        try {
            // Load ONNX Runtime first (dependency of native-lib)
            System.loadLibrary("onnxruntime");
            Log.i(TAG, "Loaded libonnxruntime.so");
        } catch (UnsatisfiedLinkError e) {
            Log.w(TAG, "libonnxruntime.so not found — MT/TTS will use fallback: " + e.getMessage());
        }

        try {
            System.loadLibrary("native-lib");
            librariesLoaded = true;
            Log.i(TAG, "Loaded libnative-lib.so");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "FATAL: Failed to load libnative-lib.so: " + e.getMessage());
            librariesLoaded = false;
        }
    }

    /** Check if native libraries loaded successfully */
    public static boolean isLoaded() {
        return librariesLoaded;
    }

    /**
     * Initialize pipeline with model paths.
     * @param asrModelPath Path to ASR model file (ggml-tiny.en.bin)
     * @param mtModelPath  Path to MT model file (encoder_model.onnx)
     * @param ttsModelPath Path to TTS model file (hi_IN-rohan-medium.onnx)
     * @return true if initialization successful
     */
    public native boolean nativeInit(String asrModelPath,
                                    String mtModelPath,
                                    String ttsModelPath);

    /** Start pipeline (spawns processing threads) */
    public native boolean nativeStart();

    /** Stop pipeline (joins threads, cleans up) */
    public native void nativeStop();

    /** Push audio samples to pipeline (16 kHz mono float32) */
    public native void nativePushAudio(float[] audioSamples);

    /** Pop synthesized audio from pipeline (16 kHz mono float32) */
    public native float[] nativePopAudio();

    /** Get current English text from ASR */
    public native String nativeGetCurrentEnglish();

    /** Get current Hindi text from MT */
    public native String nativeGetCurrentHindi();
}
