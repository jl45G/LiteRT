package com.google.ai.edge.litert

import android.content.Context
import android.hardware.HardwareBuffer
import android.opengl.GLES31
import androidx.test.InstrumentationRegistry
import com.google.common.truth.Truth.assertThat
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

/**
 * Advanced tests for LiteRT focusing on buffer interop and sync fence based
 * asynchronous execution.
 *
 * These tests are adapted from:
 * - litert/cc/litert_compiled_model_gpu_test.cc
 * - litert/cc/litert_compiled_model_integration_test.cc
 * 
 * Note: These tests require hardware capabilities that may not be available
 * in emulators. To run on a physical device, see LiteRtInteropActivity.
 */
@RunWith(JUnit4::class)
class LiteRtAdvancedInteropTest {

  private var context: Context? = null

  @Before
  fun setUp() {
    context = InstrumentationRegistry.getContext()
  }

  /**
   * Tests the basic asynchronous execution capability using events.
   */
  @Test
  fun asyncEventBasedExecutionTest() {
    // Skip test if device doesn't support GPU
    val env = Environment.create()
    if (!env.getAvailableAccelerators().contains(Accelerator.GPU)) {
      println("GPU acceleration not available, skipping test")
      return
    }

    // Create a CompiledModel with GPU acceleration
    val options = CompiledModel.Options(Accelerator.GPU)
    CompiledModel.create(context!!.assets, "simple_model.tflite", options, env).use { compiledModel ->
      // Create input and output buffers
      val inputBuffers = compiledModel.createInputBuffers()
      val outputBuffers = compiledModel.createOutputBuffers()

      // Create managed event for signaling execution start
      val event = Event.createManaged(Event.TYPE_MANAGED)

      // Fill input buffers with test data
      inputBuffers[0].writeFloat(testInputTensors[0])
      inputBuffers[1].writeFloat(testInputTensors[1])

      // Set the event on the first input buffer
      // This causes the execution to wait for this event to be signaled
      inputBuffers[0].setEvent(event.handle)

      // Run the model asynchronously
      val success = compiledModel.runAsync(inputBuffers, outputBuffers)
      assertThat(success).isTrue()

      // Signal the event to allow execution to proceed
      event.waitFence(0)

      // Wait for completion if output buffer has events
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
   * Tests OpenGL buffer to TensorBuffer interoperation.
   * This showcases writing to a tensor via OpenGL and then using
   * it for inference.
   */
  @Test
  fun glBufferInteropTest() {
    // Skip test if OpenGL is not supported
    if (!LiteRtAdvancedFeatures.hasOpenGLSupport()) {
      println("OpenGL ES 3.1 not available, skipping test")
      return
    }

    // Skip test if GPU acceleration is not available
    val env = Environment.create()
    if (!env.getAvailableAccelerators().contains(Accelerator.GPU)) {
      println("GPU acceleration not available, skipping test")
      return
    }

    // Create two OpenGL buffers for the input tensors
    val bufferSize1 = testInputTensors[0].size
    val bufferSize2 = testInputTensors[1].size
    val glBufferId1 = LiteRtAdvancedFeatures.createGlBuffer(bufferSize1)
    val glBufferId2 = LiteRtAdvancedFeatures.createGlBuffer(bufferSize2)

    assertThat(glBufferId1).isNotEqualTo(0)
    assertThat(glBufferId2).isNotEqualTo(0)

    // Fill the GL buffers with test data using compute shaders
    LiteRtAdvancedFeatures.fillGlBuffer(glBufferId1, bufferSize1, 1.0f)
    LiteRtAdvancedFeatures.fillGlBuffer(glBufferId2, bufferSize2, 0.1f)

    // Create a CompiledModel with GPU acceleration
    val options = CompiledModel.Options(Accelerator.GPU)
    CompiledModel.create(context!!.assets, "simple_model.tflite", options, env).use { compiledModel ->
      // Create input buffers from the GL buffers
      val inputBuffer1 = TensorBuffer.createFromGlBuffer(
        0, // float32
        intArrayOf(1, bufferSize1),
        GLES31.GL_SHADER_STORAGE_BUFFER,
        glBufferId1,
        (bufferSize1 * 4).toLong(), // 4 bytes per float
        0L // offset
      )

      val inputBuffer2 = TensorBuffer.createFromGlBuffer(
        0, // float32
        intArrayOf(1, bufferSize2),
        GLES31.GL_SHADER_STORAGE_BUFFER,
        glBufferId2,
        (bufferSize2 * 4).toLong(), // 4 bytes per float
        0L // offset
      )

      // Create output buffers
      val outputBuffers = compiledModel.createOutputBuffers()

      // Run the model
      compiledModel.run(listOf(inputBuffer1, inputBuffer2), outputBuffers)

      // Verify output - note values differ from standard test data
      // since we're using the GL shader to generate input values
      val output = outputBuffers[0].readFloat()
      assertThat(output.size).isEqualTo(testOutputTensor.size)

      // GL buffer 1 should contain [1.0, 2.0] (value/1.0)
      // GL buffer 2 should contain [10.0, 20.0] (value/0.1)
      // So output should be [11.0, 22.0]
      val expectedOutput = floatArrayOf(11.0f, 22.0f)
      for (i in output.indices) {
        assertThat(output[i]).isWithin(1e-4f).of(expectedOutput[i])
      }

      // Clean up
      inputBuffer1.close()
      inputBuffer2.close()
      outputBuffers.forEach { it.close() }
    }

    // Clean up OpenGL resources
    GLES31.glDeleteBuffers(1, intArrayOf(glBufferId1), 0)
    GLES31.glDeleteBuffers(1, intArrayOf(glBufferId2), 0)
  }

  /**
   * Tests OpenGL-AHWB interop using sync fences.
   * This shows how to use GL to write to a buffer, then use a sync fence
   * to safely access it via AHWB for LiteRT inference.
   */
  @Test
  fun glAhwbInteropTest() {
    // Skip test if AHWB is not supported
    if (!LiteRtAdvancedFeatures.hasAhwbSupport()) {
      println("AHWB not available, skipping test")
      return
    }

    // Skip test if OpenGL is not supported
    if (!LiteRtAdvancedFeatures.hasOpenGLSupport()) {
      println("OpenGL ES 3.1 not available, skipping test")
      return
    }

    // Skip test if GPU acceleration is not available 
    val env = Environment.create()
    if (!env.getAvailableAccelerators().contains(Accelerator.GPU)) {
      println("GPU acceleration not available, skipping test")
      return
    }

    // Create a CompiledModel with GPU acceleration (this may create AHWB buffers)
    val options = CompiledModel.Options(Accelerator.GPU)
    CompiledModel.create(context!!.assets, "simple_model.tflite", options, env).use { compiledModel ->
      // Create input buffers and check if they're AHWB type
      val inputBuffers = compiledModel.createInputBuffers()
      val outputBuffers = compiledModel.createOutputBuffers()

      // Get GL buffer handles from the tensor buffers
      val glBuffer1 = inputBuffers[0].getGlBufferInfo()
      val glBuffer2 = inputBuffers[1].getGlBufferInfo()

      // Skip test if buffer conversion to GL isn't supported
      if (glBuffer1 == null || glBuffer2 == null) {
        println("GL buffer interop not supported, skipping test")
        return
      }

      // Get the GL buffer IDs
      val glId1 = glBuffer1[1].toInt()
      val glId2 = glBuffer2[1].toInt()

      // Fill the GL buffers with test data using compute shaders
      LiteRtAdvancedFeatures.fillGlBuffer(glId1, 2, 1.0f)
      LiteRtAdvancedFeatures.fillGlBuffer(glId2, 2, 0.1f)

      // Create EGL sync and fence before AHWB read
      val nativeFence = LiteRtAdvancedFeatures.createEglSyncAndFenceFd()
      assertThat(nativeFence).isNotEqualTo(-1)

      // Create events from the sync fence
      val event1 = Event.createFromSyncFenceFd(nativeFence, false)
      val event2 = Event.createFromSyncFenceFd(nativeFence, false)

      // Set events on input buffers to block AHWB read until GPU write completes
      inputBuffers[0].setEvent(event1.handle)
      inputBuffers[1].setEvent(event2.handle)

      // Run model asynchronously
      val success = compiledModel.runAsync(inputBuffers, outputBuffers)
      assertThat(success).isTrue()

      // Wait for output to be ready
      if (outputBuffers[0].hasEvent()) {
        outputBuffers[0].waitOnEvent(-1)
      }

      // Verify output - note values differ from standard test data
      // since we're using the GL shader to generate input values
      val output = outputBuffers[0].readFloat()
      assertThat(output.size).isEqualTo(testOutputTensor.size)

      // GL buffer 1 should contain [1.0, 2.0] (value/1.0)
      // GL buffer 2 should contain [10.0, 20.0] (value/0.1)
      // So output should be [11.0, 22.0]
      val expectedOutput = floatArrayOf(11.0f, 22.0f)
      for (i in output.indices) {
        assertThat(output[i]).isWithin(1e-4f).of(expectedOutput[i])
      }

      // Clean up
      inputBuffers.forEach { it.close() }
      outputBuffers.forEach { it.close() }
    }
  }

  companion object {
    private val testInputTensors = listOf(floatArrayOf(1.0f, 2.0f), floatArrayOf(10.0f, 20.0f))
    private val testOutputTensor = floatArrayOf(11.0f, 22.0f)
  }
}