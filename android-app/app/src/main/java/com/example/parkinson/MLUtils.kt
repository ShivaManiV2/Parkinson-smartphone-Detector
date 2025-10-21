package com.example.parkinson

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*
import be.tarsos.dsp.mfcc.MFCC
import be.tarsos.dsp.util.fft.FFT
import be.tarsos.dsp.filters.HighPass
import be.tarsos.dsp.filters.LowPassFS
import be.tarsos.dsp.pitch.Yin
import java.util.LinkedList

object MLUtils {
    private var tflite: Interpreter? = null
    
    // Feature constants
    private const val N_MFCC = 13
    private const val FRAME_DURATION = 0.025 // 25ms
    private const val HOP_DURATION = 0.01 // 10ms

    // Model constants
    private const val INPUT_SIZE = 38 // 5 tap + 5 tremor + 13 mean + 13 std + 2 voice (centroid, hnr)
    private const val OUTPUT_SIZE = 1 

    private fun loadModel(context: Context) {
        if (tflite==null) {
            // 1. Load the model file from assets
            val model = context.assets.open("pd_model.tflite").use { it.readBytes() }
            val bb = ByteBuffer.allocateDirect(model.size).order(ByteOrder.nativeOrder())
            bb.put(model)
            
            // 2. Initialize the interpreter
            val options = Interpreter.Options()
            tflite = Interpreter(bb, options)
            android.util.Log.i("MLUtils", "TFLite Interpreter loaded successfully.")
        }
    }

    /**
     * Public function to run the model.
     */
    fun runTFLiteModel(context: Context, featureVector: FloatArray): Float {
        // 1. Load the model (it checks if tflite is null internally)
        loadModel(context)

        if (tflite == null) {
            android.util.Log.e("MLUtils", "TFLite Interpreter is null, cannot run inference.")
            return 0.0f
        }

        // 2. Check feature vector size
        if (featureVector.size != INPUT_SIZE) {
            android.util.Log.e("MLUtils", "Incorrect feature vector size. Expected $INPUT_SIZE, got ${featureVector.size}")
            return 0.0f
        }

        // 3. Allocate buffers
        val inputBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * 4) // 1 sample * 38 features * 4 bytes/float
        inputBuffer.order(ByteOrder.nativeOrder())
        
        val outputBuffer = ByteBuffer.allocateDirect(1 * OUTPUT_SIZE * 4) // 1 sample * 1 output * 4 bytes/float
        outputBuffer.order(ByteOrder.nativeOrder())

        // 4. Load data into input buffer
        inputBuffer.asFloatBuffer().put(featureVector)

        // 5. Run inference
        try {
            tflite?.run(inputBuffer, outputBuffer)
        } catch (e: Exception) {
            android.util.Log.e("MLUtils", "Error running TFLite inference: ${e.message}")
            return 0.0f
        }

        // 6. Get output
        outputBuffer.rewind()
        val probability = outputBuffer.asFloatBuffer().get() // Get the single float output

        return probability
    }

    /**
     * Builds the full 38-feature vector from all sub-features.
     * Use 0.0 or empty arrays for tests that are not performed.
     */
    fun buildFeatureVectorFromLocal(
        // Tapping (5)
        itiMean: Double, itiStd: Double, itiCv: Double, numTaps: Double, itiSlope: Double,
        // Tremor (5)
        tremorPeakFreq: Double, tremorPeakPower: Double, power3_7: Double, power7_12: Double, tremorRms: Double,
        // Voice (26 + 2)
        mfccMean: DoubleArray, mfccStd: DoubleArray, specCentroid: Double, hnr: Double
    ): FloatArray {
        
        val vec = ArrayList<Float>()
        
        // Tapping features (5)
        vec.add(itiMean.toFloat())
        vec.add(itiStd.toFloat())
        vec.add(itiCv.toFloat())
        vec.add(numTaps.toFloat())
        vec.add(itiSlope.toFloat())
        
        // Tremor features (5)
        vec.add(tremorPeakFreq.toFloat())
        vec.add(tremorPeakPower.toFloat())
        vec.add(power3_7.toFloat())
        vec.add(power7_12.toFloat())
        vec.add(tremorRms.toFloat())
        
        // Voice features (13 mean + 13 std = 26)
        for (i in 0 until N_MFCC) vec.add(mfccMean.getOrNull(i)?.toFloat() ?: 0f)
        for (i in 0 until N_MFCC) vec.add(mfccStd.getOrNull(i)?.toFloat() ?: 0f)
        
        // Additional voice features (2)
        vec.add(specCentroid.toFloat())
        vec.add(hnr.toFloat())

        // Total should be 38
        android.util.Log.i("MLUtils", "Built feature vector of size: ${vec.size}")
        return vec.toFloatArray()
    }
    
    /**
     * Overloaded helper for just tremor data
     */
    fun buildFeatureVectorFromTremor(tremorMap: Map<String, Double>): FloatArray {
        return buildFeatureVectorFromLocal(
            0.0, 0.0, 0.0, 0.0, 0.0, // Tapping features
            tremorMap.getOrDefault("tremor_peak_freq", 0.0),
            tremorMap.getOrDefault("tremor_peak_power", 0.0),
            tremorMap.getOrDefault("power_3_7", 0.0),
            tremorMap.getOrDefault("power_7_12", 0.0),
            tremorMap.getOrDefault("tremor_rms", 0.0),
            DoubleArray(N_MFCC) { 0.0 }, DoubleArray(N_MFCC) { 0.0 }, 0.0, 0.0 // Voice features
        )
    }

    // --- TAPPING FEATURES ---
    /**
     * Calculates the slope of a simple linear regression (like np.polyfit(t, itis, 1)[0]).
     */
    fun calculateSlope(y: DoubleArray): Double {
        val n = y.size
        if (n < 2) return 0.0
        
        val sumX = (n * (n - 1)) / 2.0
        val sumY = y.sum()
        val sumXY = y.indices.sumOf { it * y[it] }
        val sumX2 = y.indices.sumOf { (it * it).toDouble() }

        val numerator = (n * sumXY) - (sumX * sumY)
        val denominator = (n * sumX2) - (sumX * sumX)

        return if (abs(denominator) < 1e-9) 0.0 else numerator / denominator
    }
    
    // --- TREMOR FEATURES ---
    
    private fun computePowerSpectrum(signal: DoubleArray, fs: Double, nfft: Int): Pair<DoubleArray, DoubleArray> {
        val n = signal.size
        val floatSignal = FloatArray(n) { signal[it].toFloat() }
        val fftProcessor = FFT(nfft)
        val audioBuffer = FloatArray(nfft)
        System.arraycopy(floatSignal, 0, audioBuffer, 0, min(n, nfft))
        fftProcessor.forwardTransform(audioBuffer)
        
        val Pxx = DoubleArray(nfft / 2) { 0.0 }
        for (i in 0 until nfft / 2) {
            val real = audioBuffer[i]
            val imag = audioBuffer[i + nfft / 2] // Tarsos complex FFT packing
            Pxx[i] = (real * real + imag * imag).toDouble() / (fs * n)
        }
        val f = DoubleArray(nfft / 2) { it * fs / nfft }
        return Pair(f, Pxx)
    }

    private fun bandpassFilter(sig: DoubleArray, fs: Double, low: Double = 3.0, high: Double = 12.0): DoubleArray {
        val highPass = HighPass(low.toFloat(), fs.toFloat())
        val lowPass = LowPassFS(high.toFloat(), fs.toFloat())
        val floatInput = FloatArray(sig.size) { sig[it].toFloat() }
        val floatOutput = FloatArray(sig.size)
        
        // This is a simple single-pass filter, not zero-phase like filtfilt,
        // but it's the closest approximation with TarsosDSP's stream-based filters.
        for (i in floatInput.indices) {
            val h = highPass.process(floatArrayOf(floatInput[i]))[0]
            floatOutput[i] = lowPass.process(floatArrayOf(h))[0]
        }
        return floatOutput.map { it.toDouble() }.toDoubleArray()
    }
    
    private fun calculateBandPower(f: DoubleArray, Pxx: DoubleArray, low: Double, high: Double): Double {
        val indices = f.indices.filter { f[it] >= low && f[it] <= high }
        if (indices.size < 2) return 0.0

        val f_band = indices.map { f[it] }.toDoubleArray()
        val Pxx_band = indices.map { Pxx[it] }.toDoubleArray()

        var integral = 0.0
        for (i in 0 until f_band.size - 1) {
            val trapezoidArea = (Pxx_band[i] + Pxx_band[i + 1]) / 2.0 * (f_band[i + 1] - f_band[i])
            integral += trapezoidArea
        }
        return integral
    }

    fun computeTremorFeatures(accel: DoubleArray, fs: Double): Map<String, Double> {
        if (accel.size < fs) return mapOf()
        val n = accel.size
        val centered = DoubleArray(n) { accel[it] - accel.average() }
        val filt = bandpassFilter(centered, fs)
        val nfft = min(1024, filt.size)
        val (f, Pxx) = computePowerSpectrum(filt, fs, nfft)
        if (f.isEmpty()) return mapOf()

        val idx = Pxx.indices.maxByOrNull { Pxx[it] } ?: 0
        val peakFreq = f[idx]
        val peakPower = Pxx[idx]
        val power3_7 = calculateBandPower(f, Pxx, 3.0, 7.0)
        val power7_12 = calculateBandPower(f, Pxx, 7.0, 12.0)
        val tremorRms = sqrt(filt.map { it * it }.average())

        return mapOf(
            "tremor_peak_freq" to peakFreq,
            "tremor_peak_power" to peakPower,
            "power_3_7" to power3_7,
            "power_7_12" to power7_12,
            "tremor_rms" to tremorRms
        )
    }

    // --- VOICE FEATURES ---

    /**
     * Applies a pre-emphasis filter, just like librosa.
     */
    private fun preEmphasis(signal: FloatArray, coeff: Float = 0.97f): FloatArray {
        if (signal.isEmpty()) return FloatArray(0)
        val output = FloatArray(signal.size)
        output[0] = signal[0]
        for (i in 1 until signal.size) {
            output[i] = signal[i] - coeff * signal[i - 1]
        }
        return output
    }

    /**
     * Calculates the spectral centroid for a single audio frame.
     */
    private fun calculateSpectralCentroid(frame: FloatArray, sr: Float, fft: FFT): Double {
        val n = frame.size
        val audioBuffer = FloatArray(n)
        System.arraycopy(frame, 0, audioBuffer, 0, n)
        
        fft.forwardTransform(audioBuffer)
        
        val n_2 = n / 2
        val magnitudes = DoubleArray(n_2) {
            val real = audioBuffer[it]
            val imag = audioBuffer[it + n_2]
            sqrt(real * real + imag * imag).toDouble()
        }
        val freqs = DoubleArray(n_2) { it * sr.toDouble() / n }
        val weightedSum = magnitudes.indices.sumOf { magnitudes[it] * freqs[it] }
        val magnitudeSum = magnitudes.sum()

        return if (magnitudeSum < 1e-9) 0.0 else weightedSum / magnitudeSum
    }

    fun computeVoiceFeatures(floatData: FloatArray, sr: Int): Map<String, Any> {
        val emptyMap = mapOf(
            "mfcc_mean" to DoubleArray(N_MFCC) { 0.0 },
            "mfcc_std" to DoubleArray(N_MFCC) { 0.0 },
            "spec_centroid_mean" to 0.0,
            "hnr" to 0.0
        )
        if (floatData.size < (sr * FRAME_DURATION)) return emptyMap

        // 1. Apply Pre-emphasis
        val preEmphasizedData = preEmphasis(floatData)
        val frameSize = (sr * FRAME_DURATION).toInt()
        val hopSize = (sr * HOP_DURATION).toInt()
        
        // 2. Init Processors
        val mfccProcessor = MFCC(frameSize, sr.toFloat(), N_MFCC, 20, 150.0f, 6800.0f)
        val fftProcessor = FFT(frameSize)
        val yinProcessor = Yin(sr.toFloat(), frameSize)

        val mfccFrames = mutableListOf<FloatArray>()
        val centroidFrames = mutableListOf<Double>()
        val hnrFrames = mutableListOf<Double>()
        
        // 3. Process frames
        var currentStart = 0
        while (currentStart + frameSize <= preEmphasizedData.size) {
            val frame = FloatArray(frameSize)
            System.arraycopy(preEmphasizedData, currentStart, frame, 0, frameSize)
            
            mfccFrames.add(mfccProcessor.process(frame))
            centroidFrames.add(calculateSpectralCentroid(frame, sr.toFloat(), fftProcessor))
            hnrFrames.add(yinProcessor.getPitch(frame).hnr.toDouble())

            currentStart += hopSize
        }
        
        if (mfccFrames.isEmpty()) return emptyMap

        // 4. Summary Statistics (Mean & Std)
        val mfccMatrix = mfccFrames.toTypedArray()
        val totalFrames = mfccMatrix.size
        val mfccMean = DoubleArray(N_MFCC)
        val mfccStd = DoubleArray(N_MFCC)

        for (i in 0 until N_MFCC) {
            val column = mfccMatrix.map { it[i].toDouble() }
            val mean = column.average()
            val std = sqrt(column.sumOf { (it - mean) * (it - mean) } / totalFrames)
            mfccMean[i] = mean
            mfccStd[i] = std
        }

        // 5. Calculate mean for centroid and HNR
        val centroidMean = centroidFrames.average()
        val hnrMean = hnrFrames.average()

        return mapOf(
            "mfcc_mean" to mfccMean,
            "mfcc_std" to mfccStd,
            "spec_centroid_mean" to centroidMean,
            "hnr" to hnrMean
        )
    }
}