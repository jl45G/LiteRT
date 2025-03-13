package com.google.ai.edge.litert.benchmark

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.MaterialTheme
import com.google.ai.edge.litert.CompiledModel

class BenchmarkActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContent { MaterialTheme {} }
    runBenchmark()
  }

  fun runBenchmark() {
    Log.i(TAG, "start")
    val compiledModel = createCompiledModel()
    val inputBuffers = compiledModel.createInputBuffers()
    val outputBuffers = compiledModel.createOutputBuffers()

    for (i in 0..getIntIntentExtra("num_iterations")) {
      Log.i(TAG, "iteration ${i}")
      compiledModel.run(inputBuffers, outputBuffers)
    }

    Log.i(TAG, "end")

    for (tensorBuffer in inputBuffers) {
      tensorBuffer.destroy()
    }
    for (tensorBuffer in outputBuffers) {
      tensorBuffer.destroy()
    }
    compiledModel.destroy()
    finish()
  }

  fun createCompiledModel(): CompiledModel {
    val model_path = intent.getStringExtra("model_path")!!
    return CompiledModel.create(model_path)
  }

  fun getIntIntentExtra(name: String): Int {
    return intent.getIntExtra(name, 0)
  }

  companion object {
    const val TAG = "BenchmarkActivity"
  }
}
