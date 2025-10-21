package com.example.parkinson

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.lifecycle.lifecycleScope
import com.example.parkinson.database.AppDatabase
import com.example.parkinson.database.ResultEntity
import com.example.parkinson.databinding.ActivityVoiceBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class VoiceActivity : AppCompatActivity() {
    private lateinit var binding: ActivityVoiceBinding
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private val SAMPLE_RATE = 16000
    private var recordingThread: Thread? = null
    private val audioData = mutableListOf<Float>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityVoiceBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnRecordVoice.setOnClickListener {
            if (isRecording) {
                stopRecording()
            } else {
                startRecording()
            }
        }
    }

    private fun startRecording() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Audio permission not granted", Toast.LENGTH_SHORT).show()
            return
        }

        val bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        audioRecord = AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufferSize)
        
        audioData.clear()
        isRecording = true
        binding.btnRecordVoice.text = "Stop Recording"
        binding.tvVoiceResult.text = "Recording..."

        audioRecord?.startRecording()
        
        recordingThread = Thread {
            val buffer = ShortArray(bufferSize)
            while (isRecording) {
                val read = audioRecord?.read(buffer, 0, bufferSize) ?: 0
                if (read > 0) {
                    // Convert Short to Float (-1.0 to 1.0)
                    for (i in 0 until read) {
                        audioData.add(buffer[i] / 32768.0f)
                    }
                }
            }
        }
        recordingThread?.start()
    }

    private fun stopRecording() {
        isRecording = false
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        recordingThread?.join()
        
        binding.btnRecordVoice.text = "Start Recording"
        binding.tvVoiceResult.text = "Processing..."
        
        processAudio()
    }

    private fun processAudio() {
        if (audioData.size < SAMPLE_RATE) { // Less than 1 sec
            binding.tvVoiceResult.text = "Recording too short"
            return
        }
        
        // Run feature extraction and ML model in a background coroutine
        lifecycleScope.launch(Dispatchers.Default) {
            val features = MLUtils.computeVoiceFeatures(audioData.toFloatArray(), SAMPLE_RATE)
            
            // Build feature vector
            val featureVector = MLUtils.buildFeatureVectorFromLocal(
                0.0,0.0,0.0,0.0,0.0, // Tapping
                0.0,0.0,0.0,0.0,0.0, // Tremor
                features["mfcc_mean"] as DoubleArray,
                features["mfcc_std"] as DoubleArray,
                features["spec_centroid_mean"] as Double,
                features["hnr"] as Double
            )
            
            // Run model
            val score = MLUtils.runTFLiteModel(this@VoiceActivity, featureVector)
            
            // Save to DB
            val result = ResultEntity(
                timestamp = System.currentTimeMillis(),
                testType = "VOICE",
                pdScore = score,
                featuresJson = "hnr: ${features["hnr"]}"
            )
            AppDatabase.getDatabase(this@VoiceActivity).resultDao().insert(result)

            // Update UI on the main thread
            withContext(Dispatchers.Main) {
                binding.tvVoiceResult.text = String.format("PD Score: %.3f", score)
                Toast.makeText(this@VoiceActivity, "Result saved!", Toast.LENGTH_SHORT).show()
            }
        }
    }
}