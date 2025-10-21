package com.example.parkinson

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.CountDownTimer
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.parkinson.database.AppDatabase
import com.example.parkinson.database.ResultEntity
import com.example.parkinson.databinding.ActivityTremorBinding
import kotlinx.coroutines.launch
import kotlin.math.sqrt

class TremorActivity : AppCompatActivity(), SensorEventListener {
    private lateinit var binding: ActivityTremorBinding
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    
    private val accelData = mutableListOf<Double>()
    private var isRecording = false
    private val TEST_DURATION_MS: Long = 15000 // 15 seconds
    private val SENSOR_SAMPLING_RATE_HZ = 50.0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityTremorBinding.inflate(layoutInflater)
        setContentView(binding.root)

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

        if (accelerometer == null) {
            Toast.makeText(this, "Accelerometer not available", Toast.LENGTH_SHORT).show()
            binding.btnStartTremor.isEnabled = false
        }

        binding.btnStartTremor.setOnClickListener {
            startTremorTest()
        }
    }

    private fun startTremorTest() {
        accelData.clear()
        isRecording = true
        binding.btnStartTremor.text = "Recording..."
        binding.btnStartTremor.isEnabled = false
        
        // Register listener at 50Hz (20,000 us delay)
        sensorManager.registerListener(this, accelerometer, (1_000_000 / SENSOR_SAMPLING_RATE_HZ).toInt())

        object : CountDownTimer(TEST_DURATION_MS, 1000) {
            override fun onTick(millisUntilFinished: Long) {
                binding.tvTremorResult.text = "Recording... ${millisUntilFinished / 1000}s left"
            }
            override fun onFinish() {
                stopAndProcessTremor()
            }
        }.start()
    }

    private fun stopAndProcessTremor() {
        isRecording = false
        sensorManager.unregisterListener(this)
        
        binding.btnStartTremor.text = "Start Tremor Test"
        binding.btnStartTremor.isEnabled = true

        if (accelData.size < SENSOR_SAMPLING_RATE_HZ) { // Less than 1 second of data
            binding.tvTremorResult.text = "Test failed: Not enough data."
            return
        }
        
        // Compute features
        val features = MLUtils.computeTremorFeatures(accelData.toDoubleArray(), SENSOR_SAMPLING_RATE_HZ)
        
        // Build feature vector
        val featureVector = MLUtils.buildFeatureVectorFromTremor(features)
        
        // Run model
        val score = MLUtils.runTFLiteModel(this, featureVector)
        binding.tvTremorResult.text = String.format("PD Score: %.3f", score)

        // Save to DB
        lifecycleScope.launch {
            val result = ResultEntity(
                timestamp = System.currentTimeMillis(),
                testType = "TREMOR",
                pdScore = score,
                featuresJson = "peak_freq: ${features["tremor_peak_freq"]}"
            )
            AppDatabase.getDatabase(this@TremorActivity).resultDao().insert(result)
            Toast.makeText(this@TremorActivity, "Result saved!", Toast.LENGTH_SHORT).show()
        }
    }
    
    override fun onSensorChanged(event: SensorEvent?) {
        if (!isRecording || event?.sensor?.type != Sensor.TYPE_ACCELEROMETER) {
            return
        }
        
        val x = event.values[0].toDouble()
        val y = event.values[1].toDouble()
        val z = event.values[2].toDouble()
        
        // Calculate magnitude
        val magnitude = sqrt(x*x + y*y + z*z)
        accelData.add(magnitude)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Not needed
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this) // Stop sensor when app is paused
    }
}