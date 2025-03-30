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

@RunWith(JUnit4::class)
class LiteRtTest {

  private var context: Context? = null

  @Before
  fun setUp() {
    context = InstrumentationRegistry.getContext()
  }

  @Test
  fun e2eFlow_acceleratorNone() {
    CompiledModel.create(context!!.assets, "simple_model.tflite").use { compiledModel ->
      val inputBuffers = compiledModel.createInputBuffers()
      assertThat(inputBuffers).hasSize(2)
      for (i in 0 until inputBuffers.size) {
        inputBuffers[i].writeFloat(testInputTensors[i])
      }
      val outputBuffers = compiledModel.createOutputBuffers()
      assertThat(outputBuffers).hasSize(1)
      compiledModel.run(inputBuffers, outputBuffers)
      val output = outputBuffers[0].readFloat()
      assertThat(output.size).isEqualTo(testOutputTensor.size)
      for (i in 0 until output.size) {
        assertThat(output[i]).isWithin(1e-5f).of(testOutputTensor[i])
      }

      inputBuffers.forEach { it.close() }
      outputBuffers.forEach { it.close() }
    }
  }

  @Test
  fun e2eFlow_withAlternativeMethods() {
    CompiledModel.create(context!!.assets, "simple_model.tflite").use { compiledModel ->
      val inputBuffer0 = compiledModel.createInputBuffer("arg0")
      inputBuffer0.writeFloat(testInputTensors[0])
      val inputBuffer1 = compiledModel.createInputBuffer("arg1")
      inputBuffer1.writeFloat(testInputTensors[1])
      val inputBufferMap = mapOf("arg0" to inputBuffer0, "arg1" to inputBuffer1)

      val outputBuffer = compiledModel.createOutputBuffer("tfl.add")
      val outputBufferMap = mapOf("tfl.add" to outputBuffer)

      compiledModel.run(inputBufferMap, outputBufferMap)

      val output = outputBuffer.readFloat()
      assertThat(output.size).isEqualTo(testOutputTensor.size)
      for (i in 0 until output.size) {
        assertThat(output[i]).isWithin(1e-5f).of(testOutputTensor[i])
      }

      inputBuffer0.close()
      inputBuffer1.close()
      outputBuffer.close()
    }
  }

  @Test
  fun e2eFlow_modelSelector_cpu() {
    val cpuModelProvider =
      ModelProvider.staticModel(ModelProvider.Type.ASSET, "simple_model.tflite", Accelerator.CPU)
    val modelSelector = ModelSelector(cpuModelProvider)

    CompiledModel.create(modelSelector, assetManager = context!!.assets).use { compiledModel ->
      val inputBuffers = compiledModel.createInputBuffers()
      assertThat(inputBuffers).hasSize(2)
      for (i in 0 until inputBuffers.size) {
        inputBuffers[i].writeFloat(testInputTensors[i])
      }
      val outputBuffers = compiledModel.createOutputBuffers()
      assertThat(outputBuffers).hasSize(1)
      compiledModel.run(inputBuffers, outputBuffers)
      val output = outputBuffers[0].readFloat()
      assertThat(output.size).isEqualTo(testOutputTensor.size)
      for (i in 0 until output.size) {
        assertThat(output[i]).isWithin(1e-5f).of(testOutputTensor[i])
      }

      inputBuffers.forEach { it.close() }
      outputBuffers.forEach { it.close() }
    }
  }

  @Test
  fun modelSelector_gpu() {
    val cpuModelProvider =
      ModelProvider.staticModel(ModelProvider.Type.ASSET, "not_exist.tflite", Accelerator.CPU)
    val gpuModelProvider =
      ModelProvider.staticModel(ModelProvider.Type.ASSET, "simple_model.tflite", Accelerator.GPU)
    val modelSelector = ModelSelector(cpuModelProvider, gpuModelProvider)
    val modelProvider = modelSelector.selectModel(Environment.create())
    assertThat(modelProvider.getCompatibleAccelerators()).containsExactly(Accelerator.GPU)
    assertThat(modelProvider.getPath()).isEqualTo("simple_model.tflite")
  }

  @Test
  fun environment_lifecycle() {
    val env = Environment.create()
    env.close()
    try {
      env.getAvailableAccelerators()
      fail("Environment should be destroyed")
    } catch (e: IllegalStateException) {
      // Expected
    }
    // Mutiple calls to destroy() should be no-op.
    env.close()
  }

  @Test
  fun compiledModel_lifecycle() {
    val compiledModel = CompiledModel.create(context!!.assets, "simple_model.tflite")
    compiledModel.close()
    try {
      compiledModel.run(listOf(), listOf())
      fail("CompiledModel should be destroyed")
    } catch (e: IllegalStateException) {
      // Expected
    }
    // Multiple calls to destroy() should be no-op.
    compiledModel.close()
  }

  @Test
  fun tensorBuffer_lifecycle() {
    CompiledModel.create(context!!.assets, "simple_model.tflite").use {
      val inputBuffers = it.createInputBuffers()
      assertThat(inputBuffers).hasSize(2)
      for (i in 0 until inputBuffers.size) {
        inputBuffers[i].writeFloat(testInputTensors[i])
      }

      inputBuffers.forEach { it.close() }
      try {
        val unused = inputBuffers[0].readFloat()
        fail("TensorBuffer should be destroyed")
      } catch (e: IllegalStateException) {
        // Expected
      }
      // Multiple calls to destroy() should be no-op.
      inputBuffers.forEach { it.close() }
    }
  }

  @Test
  fun zeroCopyTensorBufferTest() {
    // Create a buffer for 2 float values (8 bytes total)
    val capacityBytes = 8

    val directBuf = AlignedBufferUtils.create64ByteAlignedByteBuffer(capacityBytes)
    directBuf.putFloat(123.0f)
    directBuf.putFloat(456.0f)
    directBuf.rewind()

    // Create a tensor buffer from the direct buffer with shape [1,2] and Float32 type (code 0)
    val tb = TensorBuffer.createFromDirectBuffer(0, intArrayOf(1, 2), directBuf)
    val readFloats = tb.readFloat()
    assertThat(readFloats).usingExactEquality().containsExactly(123.0f, 456.0f).inOrder()

    // Demonstrate partial buffer updates
    directBuf.rewind()
    directBuf.putFloat(123.0f)
    directBuf.putFloat(999.0f)
    directBuf.rewind()

    tb.writeFromDirect(directBuf, capacityBytes.toLong())

    val read2 = tb.readFloat()
    assertThat(read2).usingExactEquality().containsExactly(123.0f, 999.0f).inOrder()

    tb.close()
  }

  @Test
  fun eventUsageTest() {
    val compiledModel = CompiledModel.create(context!!.assets, "simple_model.tflite")
    val inputBuffers = compiledModel.createInputBuffers()
    val outputBuffers = compiledModel.createOutputBuffers()

    // Populate input buffers with test data
    for (i in inputBuffers.indices) {
      inputBuffers[i].writeFloat(testInputTensors[i])
    }

    // Execute the model
    compiledModel.run(inputBuffers, outputBuffers)

    // Check for and handle any events associated with the output buffer
    val out0 = outputBuffers[0]
    val hasEv = out0.hasEvent()
    // Events may not be present when using CPU acceleration
    if (hasEv) {
      val evHandle = out0.getEventHandle()
      // Wait indefinitely for the event to complete
      out0.waitOnEvent(-1L)
      // Release the event resources
      out0.clearEvent()
    }

    // Verify output data
    val outData = outputBuffers[0].readFloat()
    assertThat(outData.size).isEqualTo(testOutputTensor.size)

    // Clean up resources
    inputBuffers.forEach { it.close() }
    outputBuffers.forEach { it.close() }
    compiledModel.close()
  }

  @Test
  fun createAhwbBufferTest() {
    // Skip test on devices below API 29 which don't support HardwareBuffer
    if (Build.VERSION.SDK_INT < 29) return

    val shape = intArrayOf(1, 2)
    // Create a HardwareBuffer with appropriate usage flags
    val usage = HardwareBuffer.USAGE_CPU_READ_RARELY or HardwareBuffer.USAGE_CPU_WRITE_RARELY
    val hwBuf = HardwareBuffer.create(8, 1, HardwareBuffer.RGBA_8888, 1, usage)

    // Create a tensor buffer from the hardware buffer (Float32 type)
    val tb = TensorBuffer.createFromAhwb(0 /* Float32 */, shape, hwBuf, 0)
    assertThat(tb.handle).isNotEqualTo(0L)

    // Clean up resources
    tb.close()
  }

  @Test
  fun tensorScopedLockTest() {
    val capacityBytes = 8
    val directBuf = AlignedBufferUtils.create64ByteAlignedByteBuffer(capacityBytes)
    directBuf.putFloat(123.0f)
    directBuf.putFloat(456.0f)
    directBuf.rewind()

    // Create a tensor buffer with shape [1,2] and Float32 type
    val tb =
      TensorBuffer.createFromDirectBuffer(
        elementTypeCode = 0, // float32
        shape = intArrayOf(1, 2),
        directBuffer = directBuf,
      )
    // Initialize buffer with test values
    tb.writeFloat(floatArrayOf(100f, 200f))

    // Acquire a scoped lock for direct memory access
    val lock = TensorBufferScopedLock.create(tb)
    assertThat(lock).isNotNull()
    lock!!.use {
      // Get the native memory pointer for potential native operations
      val nativePtr = it.ptr
      assertThat(nativePtr).isNotEqualTo(0L)
      // Native operations could be performed here with the pointer
    } // Lock is automatically released when exiting the scope

    // Verify data integrity after lock release
    val readBack = tb.readFloat()
    assertThat(readBack).usingExactEquality().containsExactly(100f, 200f)
    tb.close()
  }

  @Test
  fun eventAsyncWaitTest() {
    // Create an event from a sync fence file descriptor
    // Using -1 as a dummy FD for testing purposes
    val dummyFd = -1
    val event = Event.createFromSyncFenceFd(dummyFd, ownsFd = false)
    // Verify event creation succeeded
    assertThat(event.handle).isNotEqualTo(0L)
    // Wait for the fence with a timeout of 500ms
    event.waitFence(500 /* ms */)
    // Clean up resources
    event.destroy()
  }

  companion object {
    private val testInputTensors = listOf(floatArrayOf(1.0f, 2.0f), floatArrayOf(10.0f, 20.0f))
    private val testOutputTensor = floatArrayOf(11.0f, 22.0f)
  }
}
