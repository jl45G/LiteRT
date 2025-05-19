/*
 * Copyright 2025 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.litert

import android.content.Context
import androidx.test.InstrumentationRegistry
import com.google.common.truth.Truth.assertThat
import kotlinx.coroutines.test.runTest
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
    val options =
      CompiledModel.Options(Accelerator.CPU).apply {
        cpuOptions = CompiledModel.CpuOptions(numThreads = 4)
      }
    CompiledModel.create(context!!.assets, "simple_model.tflite", options).use { compiledModel ->
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
  fun e2eFlow_modelSelector_cpu() = runTest {
    val env = Environment.create()
    val cpuModelProvider =
      ModelProvider.staticModel(ModelProvider.Type.ASSET, "simple_model.tflite", Accelerator.CPU)
    val npuModelProvider =
      ModelProvider.staticModel(
        ModelProvider.Type.ASSET,
        "simple_model_npu.tflite",
        Accelerator.NPU,
      )
    val modelProvider = ModelSelector(cpuModelProvider, npuModelProvider).selectModel(env)

    assertThat(modelProvider.getCompatibleAccelerators()).containsExactly(Accelerator.CPU)
    assertThat(modelProvider.getPath()).isEqualTo("simple_model.tflite")

    val options = CompiledModel.Options(*modelProvider.getCompatibleAccelerators().toTypedArray())
    val compiledModel =
      if (modelProvider.getType() == ModelProvider.Type.FILE) {
        CompiledModel.create(modelProvider.getPath(), options, env)
      } else {
        CompiledModel.create(context!!.assets, modelProvider.getPath(), options, env)
      }

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
    compiledModel.close()
  }

  @Test
  fun modelSelector_gpu() = runTest {
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
  fun modelSelector_fallbackToCpu() = runTest {
    val cpuModelProvider =
      ModelProvider.staticModel(ModelProvider.Type.ASSET, "simple_model.tflite", Accelerator.CPU)
    val gpuModelProvider =
      object : ModelProvider {
        override fun getType() = ModelProvider.Type.ASSET

        override fun isReady() = false

        override fun getPath() = "simple_model_gpu.tflite"

        override fun getCompatibleAccelerators() = setOf(Accelerator.GPU)

        override suspend fun download() {
          // This will make the fallback happen.
          throw IllegalStateException("Failed to download")
        }
      }
    val modelSelector = ModelSelector(cpuModelProvider, gpuModelProvider)
    val modelProvider = modelSelector.selectModel(Environment.create())
    assertThat(modelProvider.getCompatibleAccelerators()).containsExactly(Accelerator.CPU)
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
  fun createModel_exception() {
    try {
      CompiledModel.create("not_exist.tflite")
      fail("Exception should be thrown")
    } catch (e: LiteRtException) {
      // Expected
    }
  }

  @Test
  fun tensorBufferRequirements() {
    val compiledModel = CompiledModel.create(context!!.assets, "simple_model.tflite")

    val input0Requirements = compiledModel.getInputBufferRequirements("arg0")
    assertThat(input0Requirements.supportedTypes()).containsExactly(TensorBufferType.HostMemory)
    // The size of a float number is 4 bytes.
    assertThat(input0Requirements.bufferSize()).isEqualTo(4 * 2)
    assertThat(input0Requirements.strides()).isEqualTo(intArrayOf(0))

    val input1Requirements = compiledModel.getInputBufferRequirements("arg1")
    assertThat(input1Requirements.supportedTypes()).containsExactly(TensorBufferType.HostMemory)
    // The size of a float number is 4 bytes.
    assertThat(input1Requirements.bufferSize()).isEqualTo(4 * 2)
    assertThat(input1Requirements.strides()).isEqualTo(intArrayOf(0))

    val outputRequirements = compiledModel.getOutputBufferRequirements("tfl.add")
    assertThat(outputRequirements.supportedTypes()).containsExactly(TensorBufferType.HostMemory)
    // The size of a float number is 4 bytes.
    assertThat(outputRequirements.bufferSize()).isEqualTo(4 * 2)
    assertThat(outputRequirements.strides()).isEqualTo(intArrayOf(0))

    compiledModel.close()
  }

  @Test
  fun tensorType_simpleModel() {
    val model = Model.load(context!!.assets, "simple_model.tflite")
    val input0TensorType = model.getInputTensorType("arg0")
    assertThat(input0TensorType.elementType).isEqualTo(TensorType.ElementType.FLOAT)
    assertThat(input0TensorType.layout).isNotNull()
    val input0Layout = input0TensorType.layout!!
    assertThat(input0Layout.rank).isEqualTo(1)
    assertThat(input0Layout.dimensions).isEqualTo(intArrayOf(2))
    assertThat(input0Layout.hasStrides).isFalse()

    val input1TensorType = model.getInputTensorType("arg1")
    assertThat(input1TensorType.elementType).isEqualTo(TensorType.ElementType.FLOAT)
    assertThat(input1TensorType.layout).isNotNull()
    val input1Layout = input1TensorType.layout!!
    assertThat(input1Layout.rank).isEqualTo(1)
    assertThat(input1Layout.dimensions).isEqualTo(intArrayOf(2))
    assertThat(input1Layout.hasStrides).isFalse()

    val outputTensorType = model.getOutputTensorType("tfl.add")
    assertThat(outputTensorType.elementType).isEqualTo(TensorType.ElementType.FLOAT)
    assertThat(outputTensorType.layout).isNotNull()
    val outputLayout = outputTensorType.layout!!
    assertThat(outputLayout.rank).isEqualTo(1)
    assertThat(outputLayout.dimensions).isEqualTo(intArrayOf(2))
    assertThat(outputLayout.hasStrides).isFalse()

    model.close()
  }

  @Test
  fun tensorType_selfieMulticlass() {
    val model = Model.load(context!!.assets, "selfie_multiclass.tflite")
    val input0TensorType = model.getInputTensorType("input_29")
    assertThat(input0TensorType.elementType).isEqualTo(TensorType.ElementType.FLOAT)
    assertThat(input0TensorType.layout).isNotNull()
    val input0Layout = input0TensorType.layout!!
    assertThat(input0Layout.rank).isEqualTo(4)
    assertThat(input0Layout.dimensions).isEqualTo(intArrayOf(1, 256, 256, 3))
    assertThat(input0Layout.hasStrides).isFalse()

    val outputTensorType = model.getOutputTensorType("Identity")
    assertThat(outputTensorType.elementType).isEqualTo(TensorType.ElementType.FLOAT)
    assertThat(outputTensorType.layout).isNotNull()
    val outputLayout = outputTensorType.layout!!
    assertThat(outputLayout.rank).isEqualTo(4)
    assertThat(outputLayout.dimensions).isEqualTo(intArrayOf(1, 256, 256, 6))
    assertThat(outputLayout.hasStrides).isFalse()

    model.close()
  }

  companion object {
    private val testInputTensors = listOf(floatArrayOf(1.0f, 2.0f), floatArrayOf(10.0f, 20.0f))
    private val testOutputTensor = floatArrayOf(11.0f, 22.0f)
  }
}
