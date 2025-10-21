package com.example.parkinson.database

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface ResultDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(result: ResultEntity)

    @Query("SELECT * FROM results_table ORDER BY timestamp DESC")
    fun getAllResults(): Flow<List<ResultEntity>>
}