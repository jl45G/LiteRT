package com.google.ai.edge.litert

import android.content.Context
import androidx.test.InstrumentationRegistry
import com.google.ai.edge.litert.acceleration.ModelProvider
import com.google.ai.edge.litert.acceleration.ModelSelector
import com.google.common.truth.Truth.assertThat
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4
import java.nio.ByteBuffer
import java.nio.ByteOrder

@RunWith(JUnit4::class)
class LiteRtTest {

    private lateinit var context: Context

    @Before
    fun setUp() {
        context = InstrumentationRegistry.getContext()
    }

    @Test
    fun e2eFlow_acceleratorNone() {
        val compiledModel = CompiledModel.create(context.assets, "simple_model.tflite")

        val inputBuffers = compiledModel.createInputBuffers()
        assertThat(inputBuffers).hasSize(2)
        for (i in inputBuffers.indices) {
            inputBuffers[i].writeFloat(testInputTensors[i])
        }

        val outputBuffers = compiledModel.createOutputBuffers()
        assertThat(outputBuffers).hasSize(1)

        // Normal synchronous run
        compiledModel.run(inputBuffers, outputBuffers)
        val output = outputBuffers[0].readFloat()
        assertThat(output.size).isEqualTo(testOutputTensor.size)
        for (i in output.indices) {
            assertThat(output[i]).isWithin(1e-5f).of(testOutputTensor[i])
        }

        // NEW: Test async run
        val wasAsync = compiledModel.runAsync(inputBuffers, outputBuffers)
        // Depending on your hardware or driver, it might or might not be truly async.
        // For test, let's just confirm no crash:
        assertThat(wasAsync).isNotNull()

        // Clean up
        outputBuffers.forEach { it.destroy() }
        inputBuffers.forEach { it.destroy() }
        compiledModel.destroy()
    }

    // Example test for zero-copy TensorBuffer creation
    @Test
    fun zeroCopyTensorBufferTest() {
        // Suppose we want a 2-element float buffer: [123.0f, 456.0f]
        val capacityBytes = 8







        val directBuf = AlignedBufferUtils.create64ByteAlignedByteBuffer(capacityBytes)
        directBuf.putFloat(123.0f)
        directBuf.putFloat(456.0f)
        directBuf.rewind()

        // shape => [1,2], elementTypeCode => 0 => Float32
        val tb = TensorBuffer.createFromDirectBuffer(0, intArrayOf(1, 2), directBuf)
        val readFloats = tb.readFloat()
        // Should match [123f, 456f]
        assertThat(readFloats).usingExactEquality().containsExactly(123.0f, 456.0f).inOrder()

        // We can do partial writes, e.g. update the second float to 999.0f:
        directBuf.rewind()
        directBuf.putFloat(123.0f)
        directBuf.putFloat(999.0f)
        directBuf.rewind()

        tb.writeFromDirect(directBuf, capacityBytes.toLong())

        val read2 = tb.readFloat()
        assertThat(read2).usingExactEquality().containsExactly(123.0f, 999.0f).inOrder()

        tb.destroy()
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
  fun e2eFlow_modelSelector_gpu() {
    val cpuModelProvider =
      ModelProvider.staticModel(ModelProvider.Type.ASSET, "not_exist.tflite", Accelerator.CPU)
    val gpuModelProvider =
      ModelProvider.staticModel(ModelProvider.Type.ASSET, "simple_model.tflite", Accelerator.GPU)
    val modelSelector = ModelSelector(cpuModelProvider, gpuModelProvider)
    val compiledModel = CompiledModel.create(modelSelector, assetManager = context!!.assets)

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
  }

    // Example test for event usage
    @Test
    fun eventUsageTest() {
        val compiledModel = CompiledModel.create(context.assets, "simple_model.tflite")
        val inputBuffers = compiledModel.createInputBuffers()
        val outputBuffers = compiledModel.createOutputBuffers()

        // Write input
        for (i in inputBuffers.indices) {
            inputBuffers[i].writeFloat(testInputTensors[i])
        }

        // run
        compiledModel.run(inputBuffers, outputBuffers)

        // Typically event is set by the accelerator. Let's just see if it's there:
        val out0 = outputBuffers[0]
        val hasEv = out0.hasEvent()
        // Might be false in CPU scenario
        if (hasEv) {
            val evHandle = out0.getEventHandle()
            // wait on it
            out0.waitOnEvent(-1L)
            // clear
            out0.clearEvent()
        }

        // read output
        val outData = outputBuffers[0].readFloat()
        assertThat(outData.size).isEqualTo(testOutputTensor.size)

        // cleanup
        inputBuffers.forEach { it.destroy() }
        outputBuffers.forEach { it.destroy() }
        compiledModel.destroy()
    }

    @Test
    fun e2eFlow_modelSelector_cpu() {
        val cpuModelProvider = ModelProvider.staticModel(ModelProvider.Type.ASSET, "simple_model.tflite", Accelerator.CPU)
        val modelSelector = ModelSelector(cpuModelProvider)
        val compiledModel = CompiledModel.create(modelSelector, assetManager = context.assets)

        val inputBuffers = compiledModel.createInputBuffers()
        for (i in inputBuffers.indices) {
            inputBuffers[i].writeFloat(testInputTensors[i])
        }
        val outputBuffers = compiledModel.createOutputBuffers()
        compiledModel.run(inputBuffers, outputBuffers)
        val output = outputBuffers[0].readFloat()
        for (i in output.indices) {
            assertThat(output[i]).isWithin(1e-5f).of(testOutputTensor[i])
        }

        inputBuffers.forEach { it.destroy() }
        outputBuffers.forEach { it.destroy() }
        compiledModel.destroy()
    }

    companion object {
        private val testInputTensors = listOf(
            floatArrayOf(1.0f, 2.0f),
            floatArrayOf(10.0f, 20.0f)
        )
        private val testOutputTensor = floatArrayOf(11.0f, 22.0f)
    }
}
