package com.armm.translation;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.IOException;

/**
 * Copies model files from APK assets to app-internal storage
 * so that native C++ code can open them by file path.
 *
 * Models are stored as INT8 ONNX under their original names
 * (the prepare_models_android.sh script renames INT8→original).
 */
public class ModelManager {
    private static final String TAG = "ModelManager";

    /** All model files that must be extracted from assets. */
    private static final String[] MODEL_FILES = {
        // ASR (whisper.cpp ggml)
        "models/ggml-tiny.en.bin",

        // MT encoder-decoder (ONNX, INT8 quantized)
        "models/mt/onnx/encoder_model.onnx",
        "models/mt/onnx/decoder_model.onnx",

        // MT tokenizer
        "models/mt/onnx/source.spm",
        "models/mt/onnx/target.spm",
        "models/mt/onnx/vocab.json",
        "models/mt/onnx/config.json",
        "models/mt/onnx/generation_config.json",
        "models/mt/onnx/tokenizer_config.json",
        "models/mt/onnx/special_tokens_map.json",

        // TTS (Piper VITS ONNX, INT8 quantized)
        "models/tts/hi_IN-rohan-medium.onnx",
        "models/tts/hi_IN-rohan-medium.onnx.json",
    };

    /**
     * Extract all models from APK assets to internal storage.
     * Skips files that already exist (idempotent).
     * @param context Application context
     * @return true if all required models are present after extraction
     */
    public static boolean copyModelsFromAssets(Context context) {
        File base = context.getFilesDir();
        Log.d(TAG, "Extracting models to: " + base.getAbsolutePath());

        AssetManager assets = context.getAssets();
        int copied = 0, skipped = 0, failed = 0;

        for (String assetPath : MODEL_FILES) {
            File dest = new File(base, assetPath);
            if (dest.exists() && dest.length() > 0) {
                skipped++;
                continue;
            }
            dest.getParentFile().mkdirs();
            if (copyAsset(assets, assetPath, dest)) {
                long sizeMB = dest.length() / (1024 * 1024);
                Log.d(TAG, "  copied: " + assetPath + " (" + sizeMB + " MB)");
                copied++;
            } else {
                failed++;
            }
        }

        Log.d(TAG, "Model extraction: " + copied + " copied, " +
              skipped + " skipped, " + failed + " failed");
        return failed == 0;
    }

    private static boolean copyAsset(AssetManager assets, String assetPath, File dest) {
        try (InputStream is = assets.open(assetPath);
             FileOutputStream fos = new FileOutputStream(dest)) {
            byte[] buf = new byte[65536]; // 64 KB buffer for large models
            int n;
            while ((n = is.read(buf)) > 0) fos.write(buf, 0, n);
            return true;
        } catch (IOException e) {
            Log.w(TAG, "  skip (optional): " + assetPath + " — " + e.getMessage());
            return false;
        }
    }

    /** Returns absolute path to a model inside internal storage. */
    public static String getModelPath(Context context, String relativePath) {
        return new File(context.getFilesDir(), "models/" + relativePath).getAbsolutePath();
    }

    /** Check if all critical models exist after extraction. */
    public static boolean verifyModels(Context context) {
        String[] critical = {
            "ggml-tiny.en.bin",
            "mt/onnx/encoder_model.onnx",
            "mt/onnx/source.spm",
            "mt/onnx/target.spm",
            "tts/hi_IN-rohan-medium.onnx",
        };
        for (String rel : critical) {
            File f = new File(context.getFilesDir(), "models/" + rel);
            if (!f.exists() || f.length() == 0) {
                Log.e(TAG, "Missing critical model: " + rel);
                return false;
            }
        }
        return true;
    }
}
