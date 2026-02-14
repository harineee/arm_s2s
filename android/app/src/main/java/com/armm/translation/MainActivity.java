package com.armm.translation;

import android.Manifest;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.File;

/**
 * Main activity for real-time English → Hindi speech-to-speech translation.
 * Pipeline: Microphone → ASR → MT → TTS → Speaker
 */
public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private static final int SAMPLE_RATE = 16000;
    private static final int AUDIO_BUFFER_SIZE = SAMPLE_RATE / 10; // 100 ms chunks
    private static final int PERMISSION_REQUEST_CODE = 1;

    private NativePipeline pipeline;
    private AudioRecord audioRecord;
    private AudioTrack audioTrack;

    private Thread captureThread;
    private Thread playbackThread;
    private volatile boolean isRunning = false;
    private boolean modelsReady = false;

    private Button startButton;
    private Button stopButton;
    private TextView englishText;
    private TextView hindiText;
    private TextView statusText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        startButton = findViewById(R.id.startButton);
        stopButton = findViewById(R.id.stopButton);
        englishText = findViewById(R.id.englishText);
        hindiText = findViewById(R.id.hindiText);
        statusText = findViewById(R.id.statusText);

        startButton.setOnClickListener(v -> startTranslation());
        stopButton.setOnClickListener(v -> stopTranslation());

        startButton.setEnabled(false);
        stopButton.setEnabled(false);

        // Check native libraries
        if (!NativePipeline.isLoaded()) {
            setStatus("ERROR: Native library failed to load");
            return;
        }

        pipeline = new NativePipeline();
        setStatus("Extracting models...");

        // Extract models in background
        new Thread(() -> {
            ModelManager.copyModelsFromAssets(this);
            boolean verified = ModelManager.verifyModels(this);
            modelsReady = verified;

            runOnUiThread(() -> {
                if (modelsReady) {
                    setStatus("Ready. Tap Start to begin translation.");
                    startButton.setEnabled(true);
                } else {
                    setStatus("ERROR: Models missing or corrupt. Reinstall app.");
                }
            });
        }).start();

        // Request microphone permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.RECORD_AUDIO},
                    PERMISSION_REQUEST_CODE);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Microphone permission granted", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "Microphone permission required for translation",
                              Toast.LENGTH_LONG).show();
            }
        }
    }

    private void startTranslation() {
        if (isRunning || !modelsReady) return;

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.RECORD_AUDIO},
                    PERMISSION_REQUEST_CODE);
            return;
        }

        setStatus("Initializing pipeline...");

        // Model paths from internal storage
        String asrModel = ModelManager.getModelPath(this, "ggml-tiny.en.bin");
        String mtModel = ModelManager.getModelPath(this, "mt/onnx/encoder_model.onnx");
        String ttsModel = ModelManager.getModelPath(this, "tts/hi_IN-rohan-medium.onnx");

        Log.i(TAG, "ASR: " + asrModel + " exists=" + new File(asrModel).exists());
        Log.i(TAG, "MT:  " + mtModel + " exists=" + new File(mtModel).exists());
        Log.i(TAG, "TTS: " + ttsModel + " exists=" + new File(ttsModel).exists());

        if (!pipeline.nativeInit(asrModel, mtModel, ttsModel)) {
            setStatus("Pipeline init failed. Check logcat.");
            Log.e(TAG, "Pipeline init failed");
            return;
        }

        if (!pipeline.nativeStart()) {
            setStatus("Pipeline start failed. Check logcat.");
            pipeline.nativeStop();
            return;
        }

        // Setup audio capture
        int bufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT);
        if (bufferSize <= 0) {
            setStatus("Audio capture not supported");
            pipeline.nativeStop();
            return;
        }
        bufferSize = Math.max(bufferSize, AUDIO_BUFFER_SIZE * 4);

        audioRecord = new AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_FLOAT, bufferSize);

        if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
            setStatus("Microphone init failed");
            pipeline.nativeStop();
            return;
        }

        // Setup audio playback
        int trackBuf = AudioTrack.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT);
        if (trackBuf <= 0) trackBuf = AUDIO_BUFFER_SIZE * 4;

        audioTrack = new AudioTrack.Builder()
            .setAudioAttributes(new android.media.AudioAttributes.Builder()
                .setUsage(android.media.AudioAttributes.USAGE_MEDIA)
                .setContentType(android.media.AudioAttributes.CONTENT_TYPE_SPEECH)
                .build())
            .setAudioFormat(new AudioFormat.Builder()
                .setSampleRate(SAMPLE_RATE)
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .build())
            .setBufferSizeInBytes(trackBuf * 2)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build();

        if (audioTrack.getState() != AudioTrack.STATE_INITIALIZED) {
            setStatus("Speaker init failed");
            audioRecord.release();
            pipeline.nativeStop();
            return;
        }

        audioRecord.startRecording();
        audioTrack.play();
        isRunning = true;

        startButton.setEnabled(false);
        stopButton.setEnabled(true);
        setStatus("Listening... Speak English into microphone.");

        captureThread = new Thread(this::captureLoop, "AudioCapture");
        captureThread.start();

        playbackThread = new Thread(this::playbackLoop, "AudioPlayback");
        playbackThread.start();

        new Thread(this::updateUI, "UIUpdate").start();
    }

    private void stopTranslation() {
        if (!isRunning) return;
        Log.d(TAG, "Stopping translation...");
        isRunning = false;

        if (audioRecord != null) {
            try {
                if (audioRecord.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING)
                    audioRecord.stop();
                audioRecord.release();
            } catch (Exception e) { Log.e(TAG, "AudioRecord stop error", e); }
            audioRecord = null;
        }

        if (audioTrack != null) {
            try {
                if (audioTrack.getPlayState() == AudioTrack.PLAYSTATE_PLAYING)
                    audioTrack.stop();
                audioTrack.release();
            } catch (Exception e) { Log.e(TAG, "AudioTrack stop error", e); }
            audioTrack = null;
        }

        pipeline.nativeStop();

        try {
            if (captureThread != null) captureThread.join(1000);
            if (playbackThread != null) playbackThread.join(1000);
        } catch (InterruptedException e) { /* ignore */ }

        runOnUiThread(() -> {
            startButton.setEnabled(true);
            stopButton.setEnabled(false);
            setStatus("Stopped. Tap Start to resume.");
        });
    }

    private void captureLoop() {
        float[] buffer = new float[AUDIO_BUFFER_SIZE];
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_URGENT_AUDIO);

        while (isRunning && audioRecord != null) {
            int read = audioRecord.read(buffer, 0, buffer.length, AudioRecord.READ_BLOCKING);
            if (read > 0) {
                pipeline.nativePushAudio(buffer);
            } else if (read < 0) {
                Log.e(TAG, "AudioRecord read error: " + read);
                break;
            }
        }
    }

    private void playbackLoop() {
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_URGENT_AUDIO);

        while (isRunning && audioTrack != null) {
            float[] audio = pipeline.nativePopAudio();
            if (audio != null && audio.length > 0) {
                audioTrack.write(audio, 0, audio.length, AudioTrack.WRITE_BLOCKING);
            } else {
                try { Thread.sleep(10); } catch (InterruptedException e) { break; }
            }
        }
    }

    private void updateUI() {
        while (isRunning) {
            runOnUiThread(() -> {
                String english = pipeline.nativeGetCurrentEnglish();
                String hindi = pipeline.nativeGetCurrentHindi();
                if (english != null && !english.isEmpty() && !english.equals("[BLANK_AUDIO]"))
                    englishText.setText(english);
                if (hindi != null && !hindi.isEmpty())
                    hindiText.setText(hindi);
            });
            try { Thread.sleep(200); } catch (InterruptedException e) { break; }
        }
    }

    private void setStatus(String msg) {
        if (statusText != null) statusText.setText(msg);
        Log.i(TAG, "Status: " + msg);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopTranslation();
    }
}
