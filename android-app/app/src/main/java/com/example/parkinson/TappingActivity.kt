package com.example.parkinson

import android.graphics.Color
import android.os.Bundle
import android.os.CountDownTimer
import android.os.SystemClock
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.parkinson.database.AppDatabase
import com.example.parkinson.database.ResultEntity
import com.example.parkinson.databinding.ActivityTappingBinding
import kotlinx.coroutines.launch
import kotlin.math.sqrt

class TappingActivity : AppCompatActivity() {
    private lateinit var binding: ActivityTappingBinding
    private val timestamps = ArrayList<Long>()
    private var timer: CountDownTimer? = null
    private val TEST_DURATION_MS: Long = 10000 // 10 seconds

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityTappingBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnTap.visibility = View.GONE
        binding.tvTimer.visibility = View.GONE
        binding.tvTaps.visibility = View.GONE

        binding.btnStart.setOnClickListener {
            startTest()
        }

        binding.btnTap.setOnClickListener {
            val now = SystemClock.elapsedRealtime()
            timestamps.add(now)
            binding.tvTaps.text = "Taps: ${timestamps.size}"
        }
    }
    
    private fun startTest() {
        timestamps.clear()
        binding.btnStart.visibility = View.GONE
        binding.tvInstructions.text = "Keep tapping!"
        binding.btnTap.visibility = View.VISIBLE
        binding.tvTimer.visibility = View.VISIBLE
        binding.tvTaps.visibility = View.VISIBLE
        binding.tvTaps.text = "Taps: 0"
        binding.tvResult.text = ""

        timer = object: CountDownTimer(TEST_DURATION_MS, 100) {
            override fun onTick(millisUntilFinished: Long) {
                binding.tvTimer.text = String.format("Time: %.1f", millisUntilFinished / 1000.0)
            }
            override fun onFinish() {
                binding.tvTimer.text = "Time: 0.0"
                finishTest()
            }
        }.start()
    }

    private fun finishTest() {
        binding.btnTap.visibility = View.GONE
        binding.tvInstructions.text = "Test complete! Tap 'Start Test' to try again."
        binding.btnStart.visibility = View.VISIBLE
        
        processTapping()
    }

    private fun processTapping() {
        if (timestamps.size < 3) { // Need at least 2 ITIs for slope
            binding.tvResult.text = "Not enough taps to calculate."
            return
        }
        
        // Calculate ITIs in seconds
        val itis_seconds = DoubleArray(timestamps.size - 1) { i ->
            (timestamps[i+1] - timestamps[i]) / 1000.0
        }

        // ITI Statistics
        val mean = itis_seconds.average()
        val std = sqrt(itis_seconds.map { (it - mean).let{a->a*a} }.average())
        val cv = if (mean>0) std/mean else 0.0
        val numTaps = timestamps.size.toDouble()
        
        // Calculate Slope using our MLUtils function
        val slope = MLUtils.calculateSlope(itis_seconds)

        // Build feature vector (only tapping features)
        val featureVector = MLUtils.buildFeatureVectorFromLocal(
            mean, std, cv, numTaps, slope, // Tapping features
            0.0,0.0,0.0,0.0,0.0, // Tremor features
            DoubleArray(13){0.0}, DoubleArray(13){0.0}, 0.0, 0.0 // Voice features
        )
        
        // Run the TFLite Model
        val score = MLUtils.runTFLiteModel(this, featureVector)

        // Display Results
        binding.tvResult.text = String.format("PD Score: %.3f", score)
        binding.tvResult.setTextColor(if (score > 0.5) Color.RED else Color.GREEN)

        // Save to Database
        val result = ResultEntity(
            timestamp = System.currentTimeMillis(),
            testType = "TAPPING",
            pdScore = score,
            featuresJson = "iti_mean: $mean, iti_std: $std, slope: $slope" // Example serialization
        )
        lifecycleScope.launch {
            AppDatabase.getDatabase(this@TappingActivity).resultDao().insert(result)
            Toast.makeText(this@TappingActivity, "Result saved!", Toast.LENGTH_SHORT).show()
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        timer?.cancel() // Avoid memory leaks
    }
}