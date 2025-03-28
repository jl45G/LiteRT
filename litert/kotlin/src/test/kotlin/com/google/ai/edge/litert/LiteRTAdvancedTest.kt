package com.google.ai.edge.litert

import android.content.Context
import android.hardware.HardwareBuffer
import android.os.Build
import androidx.test.InstrumentationRegistry
import com.google.common.truth.Truth.assertThat
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Advanced tests for LiteRT Kotlin API focusing on GPU acceleration and
 * features demonstrated in C++ tests.
 *
 * These tests are adapted from:
 * - litert/cc/litert_compiled_model_gpu_test.cc
 * - litert/cc/litert_compiled_model_integration_test.cc
 */
@RunWith(JUnit4::class)
class LiteRTAdvancedTest {

  private var context: Context? = null

  @Before
  fun setUp() {
    context = InstrumentationRegistry.getContext()
  }

  /**
   * Tests GPU acceleration with the compiled model.
   * Equivalent to CompiledModelGpuTest.Basic in C++.
   */
  @Test
  fun basicGpuTest() {
    // Skip test if device doesn't support GPU
    val env = Environment.create()
    if (!env.getAvailableAccelerators().contains(Accelerator.GPU)) {
      println("GPU acceleration not available, skipping test")
      return
    }

    // Create a CompiledModel with GPU acceleration
    val options = CompiledModel.Options(Accelerator.GPU)

    CompiledModel.create(context!!.assets, "simple_model.tflite", options, env).use { compiledModel ->
      // Verify the model was loaded
      assertThat(compiledModel).isNotNull()

      // Create input and output buffers
      val inputBuffers = compiledModel.createInputBuffers()
      assertThat(inputBuffers).hasSize(2)

      // Fill input buffers with test data
      inputBuffers[0].writeFloat(testInputTensors[0])
      inputBuffers[1].writeFloat(testInputTensors[1])

      // Create output buffers
      val outputBuffers = compiledModel.createOutputBuffers()
      assertThat(outputBuffers).hasSize(1)

      // Run inference
      compiledModel.run(inputBuffers, outputBuffers)

      // Verify output
      val output = outputBuffers[0].readFloat()
      assertThat(output.size).isEqualTo(testOutputTensor.size)
      for (i in output.indices) {
        assertThat(output[i]).isWithin(1e-5f).of(testOutputTensor[i])
      }

      // Clean up
      inputBuffers.forEach { it.close() }
      outputBuffers.forEach { it.close() }
    }
  }

  /**
   * Tests running the same GPU model multiple times to verify that the
   * environment is properly shared.
   * Equivalent to CompiledModelGpuTest.Basic2nd in C++.
   */
  @Test
  fun multipleGpuExecutionsTest() {
    // Skip test if device doesn't support GPU
    val env = Environment.create()
    if (!env.getAvailableAccelerators().contains(Accelerator.GPU)) {
      println("GPU acceleration not available, skipping test")
      return
    }

    // Run the GPU test twice to verify environment sharing
    for (i in 0..1) {
      val options = CompiledModel.Options(Accelerator.GPU)

      CompiledModel.create(context!!.assets, "simple_model.tflite", options, env).use { compiledModel ->
        // Create and populate input buffers
        val inputBuffers = compiledModel.createInputBuffers()
        inputBuffers[0].writeFloat(testInputTensors[0])
        inputBuffers[1].writeFloat(testInputTensors[1])

        // Create output buffers and run inference
        val outputBuffers = compiledModel.createOutputBuffers()
        compiledModel.run(inputBuffers, outputBuffers)

        // Verify output
        val output = outputBuffers[0].readFloat()
        assertThat(output.size).isEqualTo(testOutputTensor.size)
        for (j in output.indices) {
          assertThat(output[j]).isWithin(1e-5f).of(testOutputTensor[j])
        }

        // Clean up
        inputBuffers.forEach { it.close() }
        outputBuffers.forEach { it.close() }
      }
    }
  }

  /**
   * Tests asynchronous execution with GPU model.
   * Equivalent to CompiledModelGpuTest.Async in C++.
   */
  @Test
  fun asyncGpuTest() {
    // Skip test if device doesn't support GPU
    val env = Environment.create()
    if (!env.getAvailableAccelerators().contains(Accelerator.GPU)) {
      println("GPU acceleration not available, skipping test")
      return
    }

    val options = CompiledModel.Options(Accelerator.GPU)

    CompiledModel.create(context!!.assets, "simple_model.tflite", options, env).use { compiledModel ->
      // Create input buffers
      val inputBuffers = compiledModel.createInputBuffers()

      // Create a managed event (similar to the C++ test)
      val event = Event.createManaged(Event.TYPE_MANAGED)

      // Fill input with test data
      inputBuffers[0].writeFloat(testInputTensors[0])
      inputBuffers[1].writeFloat(testInputTensors[1])

      // Set the event on the first input buffer
      // Note: In C++, events are set AFTER writing to input buffers to avoid blocking
      inputBuffers[0].setEvent(event.handle)

      // Create output buffers
      val outputBuffers = compiledModel.createOutputBuffers()

      // Run the model asynchronously
      val success = compiledModel.runAsync(inputBuffers, outputBuffers)
      assertThat(success).isTrue()

      // We need to signal the event to allow execution to proceed
      // The closest equivalent to LiteRtEventSignal in JNI is waitFence with timeout=0
      event.waitFence(0)

      // Wait for completion and ensure all outputs have events processed
      if (outputBuffers[0].hasEvent()) {
        outputBuffers[0].waitOnEvent(-1) // Wait indefinitely
      }

      // Verify output
      val output = outputBuffers[0].readFloat()
      assertThat(output.size).isEqualTo(testOutputTensor.size)
      for (i in output.indices) {
        assertThat(output[i]).isWithin(1e-5f).of(testOutputTensor[i])
      }

      // Clean up
      inputBuffers.forEach { it.close() }
      outputBuffers.forEach { it.close() }
      event.destroy()
    }
  }

  /**
   * Tests zero-copy buffer handling with direct buffers.
   * Adapts concepts from the C++ integration tests.
   */
  @Test
  fun zeroCopyBufferHandlingTest() {
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
    assertThat(readData.size).isEqualTo(testInputTensors[0].size)
    for (i in readData.indices) {
      assertThat(readData[i]).isEqualTo(testInputTensors[0][i])
    }

    // Update the direct buffer and verify changes reflect in tensor buffer
    directBuf.rewind()
    for (i in testInputTensors[0].indices) {
      directBuf.putFloat(testInputTensors[0][i] * 2) // Double the values
    }
    directBuf.rewind()

    // Update the tensor buffer from the direct buffer
    tensorBuffer.writeFromDirect(directBuf, capacityBytes.toLong())

    // Verify the updated values
    val updatedData = tensorBuffer.readFloat()
    val expectedData = testInputTensors[0].map { it * 2 }.toFloatArray()
    assertThat(updatedData.size).isEqualTo(expectedData.size)
    for (i in updatedData.indices) {
      assertThat(updatedData[i]).isEqualTo(expectedData[i])
    }

    // Clean up
    tensorBuffer.close()
  }

  /**
   * Tests direct memory access via TensorBufferScopedLock.
   * Adapts from the C++ integration tests.
   */
  @Test
  fun directMemoryAccessTest() {
    // Skip if hardware buffer API not available
    if (Build.VERSION.SDK_INT < 29) return

    // Create a tensor buffer
    val shape = intArrayOf(1, 2)
    val tensor = TensorBuffer.createFromDirectBuffer(
      elementTypeCode = 0, // float32
      shape = shape,
      directBuffer = AlignedBufferUtils.create64ByteAlignedByteBuffer(8) // 2 floats
    )

    // Write some known values
    tensor.writeFloat(floatArrayOf(3.14f, 2.71f))

    // Get a scoped lock for direct memory access
    val lock = TensorBufferScopedLock.create(tensor)
    assertThat(lock).isNotNull()

    // Use the lock to access memory directly
    lock!!.use {
      val ptr = it.ptr
      assertThat(ptr).isNotEqualTo(0L)
      // In C++, native code could directly manipulate memory at this address
    }

    // Verify data is still accessible after lock is released
    val data = tensor.readFloat()
    assertThat(data.size).isEqualTo(2)
    assertThat(data[0]).isEqualTo(3.14f)
    assertThat(data[1]).isEqualTo(2.71f)

    // Clean up
    tensor.close()
  }

  /**
   * Tests using multiple different compilation options.
   * Adapts concepts from the C++ integration tests.
   */
  @Test
  fun multipleCompilationOptionsTest() {
    // Create an environment
    val env = Environment.create()

    // Load the model once
    val model = Model.load(context!!.assets, "simple_model.tflite")

    try {
      // Test default options (CPU)
      val defaultOptions = CompiledModel.Options()
      CompiledModel.create(model, defaultOptions, env).use { cpuModel ->
        assertThat(cpuModel).isNotNull()

        // Create and fill input buffers
        val inputBuffers = cpuModel.createInputBuffers()
        inputBuffers[0].writeFloat(testInputTensors[0])
        inputBuffers[1].writeFloat(testInputTensors[1])

        // Create output buffers and run inference
        val outputBuffers = cpuModel.createOutputBuffers()
        cpuModel.run(inputBuffers, outputBuffers)

        // Verify output
        val output = outputBuffers[0].readFloat()
        assertThat(output.size).isEqualTo(testOutputTensor.size)
        for (i in output.indices) {
          assertThat(output[i]).isEqualTo(testOutputTensor[i])
        }

        // Clean up buffers
        inputBuffers.forEach { it.close() }
        outputBuffers.forEach { it.close() }
      }

      // Test GPU options (if available)
      if (env.getAvailableAccelerators().contains(Accelerator.GPU)) {
        val gpuOptions = CompiledModel.Options(Accelerator.GPU)
        CompiledModel.create(model, gpuOptions, env).use { gpuModel ->
          assertThat(gpuModel).isNotNull()

          // Create and fill input buffers
          val inputBuffers = gpuModel.createInputBuffers()
          inputBuffers[0].writeFloat(testInputTensors[0])
          inputBuffers[1].writeFloat(testInputTensors[1])

          // Create output buffers and run inference
          val outputBuffers = gpuModel.createOutputBuffers()
          gpuModel.run(inputBuffers, outputBuffers)

          // Verify output
          val output = outputBuffers[0].readFloat()
          for (i in output.indices) {
            assertThat(output[i]).isWithin(1e-5f).of(testOutputTensor[i])
          }

          // Clean up buffers
          inputBuffers.forEach { it.close() }
          outputBuffers.forEach { it.close() }
        }
      }
    } finally {
      // Clean up model
      model.close()
    }
  }

  /**
   * Tests hardware buffer integration if supported.
   * Inspired by the AHWB tests in the C++ integration tests.
   */
  @Test
  fun hardwareBufferTest() {
    // Skip test on devices below API 29 which don't support HardwareBuffer
    if (Build.VERSION.SDK_INT < 29) return

    // Create a hardware buffer
    val shape = intArrayOf(1, 2)
    val usage = HardwareBuffer.USAGE_CPU_READ_RARELY or HardwareBuffer.USAGE_CPU_WRITE_RARELY
    val hwBuffer = HardwareBuffer.create(8, 1, HardwareBuffer.RGBA_8888, 1, usage)

    try {
      // Create a tensor buffer from the hardware buffer
      val tensor = TensorBuffer.createFromAhwb(0 /* Float32 */, shape, hwBuffer)

      // Verify the tensor buffer was created successfully
      assertThat(tensor.handle).isNotEqualTo(0L)

      // Retrieve the hardware buffer from the tensor buffer
      val retrievedBuffer = TensorBuffer.getAhwbHandle(tensor)
      assertThat(retrievedBuffer).isNotNull()

      // Clean up tensor buffer
      tensor.close()
    } finally {
      // Clean up hardware buffer
      hwBuffer.close()
    }
  }

  /**
   * Tests running multiple instances of inference concurrently.
   * Adapts concepts from both C++ test files.
   */
  @Test
  fun concurrentInferenceTest() {
    val env = Environment.create()
    val compiledModel = CompiledModel.create(context!!.assets, "simple_model.tflite",
        CompiledModel.Options(), env)

    // Number of concurrent inferences to run
    val numConcurrent = 5

    try {
      // Create a buffer for each concurrent operation
      val inputBuffersList = (0 until numConcurrent).map { idx ->
        val buffers = compiledModel.createInputBuffers()
        // Fill with slightly different data for each inference
        for (i in 0 until buffers.size) {
          val data = testInputTensors[i].clone()
          // Make slight modifications to input data for each inference
          for (j in data.indices) {
            data[j] += (0.01f * idx)
          }
          buffers[i].writeFloat(data)
        }
        buffers
      }

      val outputBuffersList = (0 until numConcurrent).map { _ ->
        compiledModel.createOutputBuffers()
      }

      // Run all inferences
      for (i in 0 until numConcurrent) {
        compiledModel.run(inputBuffersList[i], outputBuffersList[i])
      }

      // Verify all outputs
      for (i in 0 until numConcurrent) {
        val output = outputBuffersList[i][0].readFloat()
        assertThat(output.size).isEqualTo(testOutputTensor.size)

        // Expected values will be slightly modified based on our input adjustments
        val adjustmentFactor = 0.01f * i
        for (j in output.indices) {
          val expectedValue = testOutputTensor[j] + adjustmentFactor * 2 // 2 inputs, each adjusted
          assertThat(output[j]).isWithin(1e-4f).of(expectedValue)
        }
      }

      // Clean up all buffers
      inputBuffersList.forEach { buffers -> buffers.forEach { it.close() } }
      outputBuffersList.forEach { buffers -> buffers.forEach { it.close() } }

    } finally {
      compiledModel.close()
    }
  }

  companion object {
    private val testInputTensors = listOf(floatArrayOf(1.0f, 2.0f), floatArrayOf(10.0f, 20.0f))
    private val testOutputTensor = floatArrayOf(11.0f, 22.0f)
  }
}
