package com.example.parkinson.database

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "results_table")
data class ResultEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Int = 0,
    val timestamp: Long,
    val testType: String, // "TAPPING", "TREMOR", "VOICE"
    val pdScore: Float,
    val featuresJson: String // A simple way to store the feature vector
)