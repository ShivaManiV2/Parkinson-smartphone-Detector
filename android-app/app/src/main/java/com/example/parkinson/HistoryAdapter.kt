package com.example.parkinson

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.example.parkinson.database.ResultEntity
import java.text.SimpleDateFormat
import java.util.*

class HistoryAdapter : ListAdapter<ResultEntity, HistoryAdapter.ResultViewHolder>(ResultDiffCallback()) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ResultViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_history_result, parent, false)
        return ResultViewHolder(view)
    }

    override fun onBindViewHolder(holder: ResultViewHolder, position: Int) {
        val result = getItem(position)
        holder.bind(result)
    }

    class ResultViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val dateTextView: TextView = itemView.findViewById(R.id.tv_item_date)
        private val typeTextView: TextView = itemView.findViewById(R.id.tv_item_type)
        private val scoreTextView: TextView = itemView.findViewById(R.id.tv_item_score)
        
        private val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())

        fun bind(result: ResultEntity) {
            dateTextView.text = "Date: ${dateFormat.format(Date(result.timestamp))}"
            typeTextView.text = "Test Type: ${result.testType}"
            scoreTextView.text = String.format("PD Score: %.3f", result.pdScore)
            scoreTextView.setTextColor(if (result.pdScore > 0.5) Color.RED else Color.GREEN)
        }
    }

    class ResultDiffCallback : DiffUtil.ItemCallback<ResultEntity>() {
        override fun areItemsTheSame(oldItem: ResultEntity, newItem: ResultEntity): Boolean {
            return oldItem.id == newItem.id
        }
        override fun areContentsTheSame(oldItem: ResultEntity, newItem: ResultEntity): Boolean {
            return oldItem == newItem
        }
    }
}