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
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class LiteRtMhTest {

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
  fun e2eFlow_modelSelector_cpu() = runTest {
    val env = Environment.create()
    val cpuModelProvider =
      ModelProvider.staticModel(ModelProvider.Type.ASSET, "simple_model.tflite", Accelerator.CPU)
    val modelProvider = ModelSelector(cpuModelProvider).selectModel(env)
    val compiledModel = createCompiledModel(modelProvider, env)

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
  fun e2eFlow_modelSelector_gpu() = runTest {
    val env = Environment.create()
    val cpuModelProvider =
      ModelProvider.staticModel(ModelProvider.Type.ASSET, "not_exist.tflite", Accelerator.CPU)
    val gpuModelProvider =
      ModelProvider.staticModel(ModelProvider.Type.ASSET, "simple_model.tflite", Accelerator.GPU)
    val modelProvider = ModelSelector(cpuModelProvider, gpuModelProvider).selectModel(env)
    val gpuOptions =
      CompiledModel.GpuOptions(
        constantTensorSharing = true,
        infiniteFloatCapping = true,
        allowSrcQuantizedFcConvOps = false,
        precision = CompiledModel.GpuOptions.Precision.FP16,
        bufferStorageType = CompiledModel.GpuOptions.BufferStorageType.BUFFER,
      )
    val compiledModel = createCompiledModel(modelProvider, env, gpuOptions)

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

  private fun createCompiledModel(
    modelProvider: ModelProvider,
    env: Environment = Environment.create(),
    gpuOptions: CompiledModel.GpuOptions? = null,
  ): CompiledModel {
    val options = CompiledModel.Options(*modelProvider.getCompatibleAccelerators().toTypedArray())
    if (gpuOptions != null) {
      options.gpuOptions = gpuOptions
    }
    if (modelProvider.getType() == ModelProvider.Type.FILE) {
      return CompiledModel.create(modelProvider.getPath(), options, env)
    } else {
      return CompiledModel.create(context!!.assets, modelProvider.getPath(), options, env)
    }
  }

  companion object {
    private val testInputTensors = listOf(floatArrayOf(1.0f, 2.0f), floatArrayOf(10.0f, 20.0f))
    private val testOutputTensor = floatArrayOf(11.0f, 22.0f)
  }
}
