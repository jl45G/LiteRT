package org.tensorflow.lite.examples.classification.test

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.ai.edge.litert.*
import org.tensorflow.lite.examples.classification.R
import java.nio.ByteBuffer
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Test activity to verify LiteRT GPU functionality on physical devices
 */
class LiteRtTestActivity : AppCompatActivity() {
    private val TAG = "LiteRTTestApp"
    private lateinit var resultsTextView: TextView
    private lateinit var runTestsButton: Button
    private lateinit var logBuilder: StringBuilder

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_litert_test)

        resultsTextView = findViewById(R.id.results_text)
        runTestsButton = findViewById(R.id.run_tests_button)
        logBuilder = StringBuilder()

        runTestsButton.setOnClickListener {
            runTestsButton.isEnabled = false
            logBuilder.clear()
            resultsTextView.text = "Running tests..."

            // Run tests in background thread
            Thread {
                runAllTests()
                runOnUiThread {
                    resultsTextView.text = logBuilder.toString()
                    runTestsButton.isEnabled = true
                }
            }.start()
        }
    }

    private fun log(message: String) {
        Log.d(TAG, message)
        logBuilder.append(message).append("\n")
    }

    private fun runAllTests() {
        try {
            val env = Environment.create()
            log("Created environment")

            val availableAccelerators = env.getAvailableAccelerators()
            log("Available accelerators: $availableAccelerators")

            if (Accelerator.GPU in availableAccelerators) {
                try {
                    runBasicGpuTest(env)
                    log("✅ Basic GPU test passed")
                } catch (e: Exception) {
                    log("❌ Basic GPU test failed: ${e.message}")
                    e.printStackTrace()
                }

                try {
                    runAsyncGpuTest(env)
                    log("✅ Async GPU test passed")
                } catch (e: Exception) {
                    log("❌ Async GPU test failed: ${e.message}")
                    e.printStackTrace()
                }

                try {
                    runMultipleGpuExecutionsTest(env)
                    log("✅ Multiple GPU executions test passed")
                } catch (e: Exception) {
                    log("❌ Multiple GPU executions test failed: ${e.message}")
                    e.printStackTrace()
                }
            } else {
                log("⚠️ GPU not available on this device, skipping GPU tests")
            }

            // Run non-GPU tests that should pass regardless
            try {
                runCpuTest(env)
                log("✅ CPU test passed")
            } catch (e: Exception) {
                log("❌ CPU test failed: ${e.message}")
                e.printStackTrace()
            }

            try {
                runZeroCopyBufferTest()
                log("✅ Zero copy buffer test passed")
            } catch (e: Exception) {
                log("❌ Zero copy buffer test failed: ${e.message}")
                e.printStackTrace()
            }

        } catch (e: Exception) {
            log("❌ Test setup failed: ${e.message}")
            e.printStackTrace()
        }
    }

    private fun runBasicGpuTest(env: Environment) {
        // Create a CompiledModel with GPU acceleration
        val options = CompiledModel.Options(Accelerator.GPU)

        CompiledModel.create(assets, "simple_model.tflite", options, env).use { compiledModel ->
            // Create input and output buffers
            val inputBuffers = compiledModel.createInputBuffers()
            if (inputBuffers.size != 2) {
                throw RuntimeException("Expected 2 input buffers, got ${inputBuffers.size}")
            }

            // Fill input buffers with test data
            inputBuffers[0].writeFloat(testInputTensors[0])
            inputBuffers[1].writeFloat(testInputTensors[1])

            // Create output buffers
            val outputBuffers = compiledModel.createOutputBuffers()
            if (outputBuffers.size != 1) {
                throw RuntimeException("Expected 1 output buffer, got ${outputBuffers.size}")
            }

            // Run inference
            compiledModel.run(inputBuffers, outputBuffers)

            // Verify output
            val output = outputBuffers[0].readFloat()
            if (output.size != testOutputTensor.size) {
                throw RuntimeException("Output size mismatch: ${output.size} vs ${testOutputTensor.size}")
            }

            for (i in output.indices) {
                val diff = Math.abs(output[i] - testOutputTensor[i])
                if (diff > 1e-5f) {
                    throw RuntimeException("Output value mismatch at index $i: ${output[i]} vs ${testOutputTensor[i]}")
                }
            }

            // Clean up
            inputBuffers.forEach { it.close() }
            outputBuffers.forEach { it.close() }
        }
    }

    private fun runAsyncGpuTest(env: Environment) {
        val options = CompiledModel.Options(Accelerator.GPU)

        CompiledModel.create(assets, "simple_model.tflite", options, env).use { compiledModel ->
            // Create input buffers
            val inputBuffers = compiledModel.createInputBuffers()

            // Create a managed event
            val event = Event.createManaged(Event.TYPE_MANAGED)

            // Fill input with test data
            inputBuffers[0].writeFloat(testInputTensors[0])
            inputBuffers[1].writeFloat(testInputTensors[1])

            // Set the event on the first input buffer
            inputBuffers[0].setEvent(event.handle)

            // Create output buffers
            val outputBuffers = compiledModel.createOutputBuffers()

            // Run the model asynchronously
            val success = compiledModel.runAsync(inputBuffers, outputBuffers)
            if (!success) {
                throw RuntimeException("runAsync failed")
            }

            // Signal the event to allow execution to proceed
            event.waitFence(0)

            // Wait for completion
            if (outputBuffers[0].hasEvent()) {
                outputBuffers[0].waitOnEvent(-1)
            }

            // Verify output
            val output = outputBuffers[0].readFloat()
            if (output.size != testOutputTensor.size) {
                throw RuntimeException("Output size mismatch: ${output.size} vs ${testOutputTensor.size}")
            }

            for (i in output.indices) {
                val diff = Math.abs(output[i] - testOutputTensor[i])
                if (diff > 1e-5f) {
                    throw RuntimeException("Output value mismatch at index $i: ${output[i]} vs ${testOutputTensor[i]}")
                }
            }

            // Clean up
            inputBuffers.forEach { it.close() }
            outputBuffers.forEach { it.close() }
            event.destroy()
        }
    }

    private fun runMultipleGpuExecutionsTest(env: Environment) {
        // Run the GPU test twice to verify environment sharing
        for (i in 0..1) {
            val options = CompiledModel.Options(Accelerator.GPU)

            CompiledModel.create(assets, "simple_model.tflite", options, env).use { compiledModel ->
                // Create and populate input buffers
                val inputBuffers = compiledModel.createInputBuffers()
                inputBuffers[0].writeFloat(testInputTensors[0])
                inputBuffers[1].writeFloat(testInputTensors[1])

                // Create output buffers and run inference
                val outputBuffers = compiledModel.createOutputBuffers()
                compiledModel.run(inputBuffers, outputBuffers)

                // Verify output
                val output = outputBuffers[0].readFloat()
                if (output.size != testOutputTensor.size) {
                    throw RuntimeException("Output size mismatch in iteration $i: ${output.size} vs ${testOutputTensor.size}")
                }

                for (j in output.indices) {
                    val diff = Math.abs(output[j] - testOutputTensor[j])
                    if (diff > 1e-5f) {
                        throw RuntimeException("Output value mismatch in iteration $i at index $j: ${output[j]} vs ${testOutputTensor[j]}")
                    }
                }

                // Clean up
                inputBuffers.forEach { it.close() }
                outputBuffers.forEach { it.close() }
            }
        }
    }

    private fun runCpuTest(env: Environment) {
        // Create a CompiledModel with CPU acceleration
        val options = CompiledModel.Options(Accelerator.CPU)

        CompiledModel.create(assets, "simple_model.tflite", options, env).use { compiledModel ->
            // Create input and output buffers
            val inputBuffers = compiledModel.createInputBuffers()

            // Fill input buffers with test data
            inputBuffers[0].writeFloat(testInputTensors[0])
            inputBuffers[1].writeFloat(testInputTensors[1])

            // Create output buffers
            val outputBuffers = compiledModel.createOutputBuffers()

            // Run inference
            compiledModel.run(inputBuffers, outputBuffers)

            // Verify output
            val output = outputBuffers[0].readFloat()

            for (i in output.indices) {
                val diff = Math.abs(output[i] - testOutputTensor[i])
                if (diff > 1e-5f) {
                    throw RuntimeException("Output value mismatch at index $i: ${output[i]} vs ${testOutputTensor[i]}")
                }
            }

            // Clean up
            inputBuffers.forEach { it.close() }
            outputBuffers.forEach { it.close() }
        }
    }

    private fun runZeroCopyBufferTest() {
        // Create a buffer for test input values (8 bytes for 2 floats)
        val capacityBytes = testInputTensors[0].size * 4
        val directBuf = AlignedBufferUtils.create64ByteAlignedByteBuffer(capacityBytes)

        // Fill the buffer with test values
        testInputTensors[0].forEach { directBuf.putFloat(it) }
        directBuf.rewind()

        // Create a tensor buffer using the direct buffer
        val tensorBuffer = TensorBuffer.createFromDirectBuffer(
            elementTypeCode = 0, // float32
            shape = intArrayOf(1, testInputTensors[0].size),
            directBuffer = directBuf
        )

        // Verify the data was correctly written
        val readData = tensorBuffer.readFloat()
        if (readData.size != testInputTensors[0].size) {
            throw RuntimeException("Read data size mismatch: ${readData.size} vs ${testInputTensors[0].size}")
        }

        for (i in readData.indices) {
            if (readData[i] != testInputTensors[0][i]) {
                throw RuntimeException("Value mismatch at index $i: ${readData[i]} vs ${testInputTensors[0][i]}")
            }
        }

        // Clean up
        tensorBuffer.close()
    }

    companion object {
        private val testInputTensors = listOf(floatArrayOf(1.0f, 2.0f), floatArrayOf(10.0f, 20.0f))
        private val testOutputTensor = floatArrayOf(11.0f, 22.0f)
    }
}
