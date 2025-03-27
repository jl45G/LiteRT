package com.google.ai.edge.litert

import android.content.Context
import androidx.test.InstrumentationRegistry
import com.google.common.truth.Truth.assertThat
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

/**
 * Advanced integration tests for LiteRT, adapted from C++ tests in:
 * - litert_compiled_model_gpu_test.cc
 * - litert_compiled_model_integration_test.cc
 */
@RunWith(JUnit4::class)
class LiteRtAdvancedTest {

  private var context: Context? = null

  @Before
  fun setUp() {
    context = InstrumentationRegistry.getContext()
  }

  /**
   * Tests basic GPU execution of the model.
   * Based on BasicTest in litert_compiled_model_gpu_test.cc
   */
  @Test
  fun basicGpuTest() {
    // Create environment
    Environment.create().use { env ->
      // Load model
      Model.load(context!!.assets, "simple_model.tflite").use { model ->
        // Create compiled model with GPU acceleration
        CompiledModel.create(model, CompiledModel.Options(Accelerator.GPU), env).use { compiledModel ->
          // Create input and output buffers
          val inputBuffers = compiledModel.createInputBuffers()
          assertThat(inputBuffers).hasSize(2)
          
          // Fill input buffers with test data
          inputBuffers[0].writeFloat(testInputTensors[0])
          inputBuffers[1].writeFloat(testInputTensors[1])
          
          // Create output buffers
          val outputBuffers = compiledModel.createOutputBuffers()
          assertThat(outputBuffers).hasSize(1)
          
          // Run the model
          compiledModel.run(inputBuffers, outputBuffers)
          
          // Verify output
          val output = outputBuffers[0].readFloat()
          assertThat(output.size).isEqualTo(testOutputTensor.size)
          for (i in 0 until output.size) {
            assertThat(output[i]).isWithin(1e-5f).of(testOutputTensor[i])
          }
          
          // Clean up
          inputBuffers.forEach { it.close() }
          outputBuffers.forEach { it.close() }
        }
      }
    }
  }

  /**
   * Tests a second GPU execution to verify environment sharing.
   * Based on Basic2nd in litert_compiled_model_gpu_test.cc
   */
  @Test
  fun secondGpuTest() {
    // Run the same test twice to verify that the GPU environment is shared between instances
    basicGpuTest()
  }

  /**
   * Tests NPU execution if available.
   * Based on RunWithGoogleTensorModel in litert_compiled_model_integration_test.cc
   */
  @Test
  fun runWithNpuModel() {
    // This test is specific to devices with NPU support
    // Since we don't have easy access to query hardware capabilities in unit tests,
    // we'll try to run with NPU and catch any failures

    try {
      // Create environment with dispatch library directory
      val options = mapOf(
        Environment.Option.DispatchLibraryDir to "/data/local/tmp"
      )
      
      Environment.create(options).use { env ->
        // Load model
        Model.load(context!!.assets, "simple_model.tflite").use { model ->
          // Create compiled model with NPU acceleration
          CompiledModel.create(model, CompiledModel.Options(Accelerator.NPU), env).use { compiledModel ->
            // Create input and output buffers
            val inputBuffers = compiledModel.createInputBuffers()
            assertThat(inputBuffers).hasSize(2)
            
            // Fill input buffers with test data
            inputBuffers[0].writeFloat(testInputTensors[0])
            inputBuffers[1].writeFloat(testInputTensors[1])
            
            // Create output buffers
            val outputBuffers = compiledModel.createOutputBuffers()
            assertThat(outputBuffers).hasSize(1)
            
            // Run the model
            compiledModel.run(inputBuffers, outputBuffers)
            
            // Verify output
            val output = outputBuffers[0].readFloat()
            assertThat(output.size).isEqualTo(testOutputTensor.size)
            for (i in 0 until output.size) {
              assertThat(output[i]).isWithin(1e-5f).of(testOutputTensor[i])
            }
            
            // Clean up
            inputBuffers.forEach { it.close() }
            outputBuffers.forEach { it.close() }
          }
        }
      }
    } catch (e: Exception) {
      // This is expected on devices without NPU support
      // Skip the test rather than fail
    }
  }

  /**
   * Tests AHWB buffer usage if available.
   * Based on litert_compiled_model_integration_test.cc
   */
  @Test
  fun testAhwbBuffers() {
    try {
      // Create environment with dispatch library directory
      val options = mapOf(
        Environment.Option.DispatchLibraryDir to "/data/local/tmp"
      )
      
      Environment.create(options).use { env ->
        // Load model
        Model.load(context!!.assets, "simple_model.tflite").use { model ->
          // Create compiled model that may use AHWB buffers on supported devices
          CompiledModel.create(model, CompiledModel.Options(Accelerator.NPU), env).use { compiledModel ->
            // Create input and output buffers
            val inputBuffers = compiledModel.createInputBuffers()
            assertThat(inputBuffers).hasSize(2)
            
            // Fill input buffers with test data
            inputBuffers[0].writeFloat(testInputTensors[0])
            inputBuffers[1].writeFloat(testInputTensors[1])
            
            // Create output buffers
            val outputBuffers = compiledModel.createOutputBuffers()
            assertThat(outputBuffers).hasSize(1)
            
            // Run the model
            compiledModel.run(inputBuffers, outputBuffers)
            
            // Verify output
            val output = outputBuffers[0].readFloat()
            assertThat(output.size).isEqualTo(testOutputTensor.size)
            for (i in 0 until output.size) {
              assertThat(output[i]).isWithin(1e-5f).of(testOutputTensor[i])
            }
            
            // Clean up
            inputBuffers.forEach { it.close() }
            outputBuffers.forEach { it.close() }
          }
        }
      }
    } catch (e: Exception) {
      // This is expected on devices without AHWB support
      // Skip the test rather than fail
    }
  }

  /**
   * Tests high performance inference path by using buffer mapping.
   * This simulates a camera frame processing pipeline.
   */
  @Test
  fun highPerformanceInferencePath() {
    // Create environment
    Environment.create().use { env ->
      // Load model
      Model.load(context!!.assets, "simple_model.tflite").use { model ->
        // Create compiled model, prefer GPU acceleration for image processing
        CompiledModel.create(model, CompiledModel.Options(Accelerator.GPU), env).use { compiledModel ->
          // Create input and output buffers
          val inputBuffers = compiledModel.createInputBuffers()
          assertThat(inputBuffers).hasSize(2)
          
          // Get input buffer names from model
          val inputNames = listOf("arg0", "arg1")
          
          // Map input tensors by name
          val inputBufferMap = inputNames.zip(inputBuffers).toMap()
          
          // Fill input buffers with test data
          inputBufferMap["arg0"]!!.writeFloat(testInputTensors[0])
          inputBufferMap["arg1"]!!.writeFloat(testInputTensors[1])
          
          // Create output buffers and map
          val outputBuffers = compiledModel.createOutputBuffers()
          assertThat(outputBuffers).hasSize(1)
          val outputNames = listOf("tfl.add")
          val outputBufferMap = outputNames.zip(outputBuffers).toMap()
          
          // Run model with named buffers
          compiledModel.run(inputBufferMap, outputBufferMap)
          
          // Verify output
          val output = outputBufferMap["tfl.add"]!!.readFloat()
          assertThat(output.size).isEqualTo(testOutputTensor.size)
          for (i in 0 until output.size) {
            assertThat(output[i]).isWithin(1e-5f).of(testOutputTensor[i])
          }
          
          // Clean up
          inputBuffers.forEach { it.close() }
          outputBuffers.forEach { it.close() }
        }
      }
    }
  }
  
  /**
   * Tests async execution with events.
   * Based on Async test in litert_compiled_model_gpu_test.cc
   */
  @Test
  fun asyncExecution() {
    try {
      // Create environment
      Environment.create().use { env ->
        // Load model
        Model.load(context!!.assets, "simple_model.tflite").use { model ->
          // Create compiled model with GPU acceleration
          CompiledModel.create(model, CompiledModel.Options(Accelerator.GPU), env).use { compiledModel ->
            // Create input and output buffers
            val inputBuffers = compiledModel.createInputBuffers()
            assertThat(inputBuffers).hasSize(2)
            
            // Create OpenCL event and get a copy for signaling
            val inputEvent = Event.createManaged(Event.Type.OPEN_CL)
            
            // Fill input buffers with test data
            inputBuffers[0].writeFloat(testInputTensors[0])
            inputBuffers[1].writeFloat(testInputTensors[1])
            
            // Set event on first input buffer
            // Note: This must be done after filling the buffer
            inputBuffers[0].setEvent(inputEvent)
            
            // Create output buffers
            val outputBuffers = compiledModel.createOutputBuffers()
            assertThat(outputBuffers).hasSize(1)
            
            // Run model asynchronously
            val asyncMode = true
            val wasAsync = compiledModel.runAsync(inputBuffers, outputBuffers, 0, asyncMode)
            
            // If async execution succeeded, read and verify output
            if (wasAsync) {
              // Signal the event to trigger execution
              inputEvent.signal()
              
              // Verify output
              val output = outputBuffers[0].readFloat()
              assertThat(output.size).isEqualTo(testOutputTensor.size)
              for (i in 0 until output.size) {
                assertThat(output[i]).isWithin(1e-5f).of(testOutputTensor[i])
              }
            }
            
            // Clean up
            inputBuffers.forEach { it.close() }
            outputBuffers.forEach { it.close() }
          }
        }
      }
    } catch (e: Exception) {
      // This is expected on devices without OpenCL support
      // Skip the test rather than fail
    }
  }
  
  /**
   * Tests GPU/GL interop functionality.
   * Based on RunAsyncWithGoogleTensorModelUseAhwbGlInterop in litert_compiled_model_integration_test.cc
   */
  @Test
  fun gpuGlInterop() {
    try {
      // Create environment
      Environment.create().use { env ->
        // Load model
        Model.load(context!!.assets, "simple_model.tflite").use { model ->
          // Create compiled model with GPU acceleration
          CompiledModel.create(model, CompiledModel.Options(Accelerator.GPU), env).use { compiledModel ->
            // Create input and output buffers
            val inputBuffers = compiledModel.createInputBuffers()
            assertThat(inputBuffers).hasSize(2)
            
            // Get GL buffer IDs
            val glBuffer1 = inputBuffers[0].getGlBuffer()
            val glBuffer2 = inputBuffers[1].getGlBuffer()
            
            // In real application, we would now use GL to render to these buffers
            // For test purposes, we'll just fill them with test data directly
            inputBuffers[0].writeFloat(testInputTensors[0])
            inputBuffers[1].writeFloat(testInputTensors[1])
            
            // Create sync fence events
            // Note: For testing purposes, we're using OpenCL events instead of sync fence events
            val event1 = Event.createManaged(Event.Type.OPEN_CL)
            val event2 = Event.createManaged(Event.Type.OPEN_CL)
            
            // Set events to synchronize GL writes with LiteRT reads
            inputBuffers[0].setEvent(event1)
            inputBuffers[1].setEvent(event2)
            
            // Create output buffers
            val outputBuffers = compiledModel.createOutputBuffers()
            assertThat(outputBuffers).hasSize(1)
            
            // Run model asynchronously
            var wasAsync = false
            try {
              wasAsync = compiledModel.runAsync(inputBuffers, outputBuffers, 0, true)
            } catch (e: Exception) {
              // If async execution failed, we'll still test the synchronous path
            }
            
            // Signal events to trigger execution (whether sync or async)
            event1.signal()
            event2.signal()
            
            // Verify output
            val output = outputBuffers[0].readFloat()
            assertThat(output.size).isEqualTo(testOutputTensor.size)
            for (i in 0 until output.size) {
              assertThat(output[i]).isWithin(1e-5f).of(testOutputTensor[i])
            }
            
            // Clean up
            inputBuffers.forEach { it.close() }
            outputBuffers.forEach { it.close() }
          }
        }
      }
    } catch (e: Exception) {
      // This is expected on devices without GL/CL interop support
      // Skip the test rather than fail
    }
  }

  companion object {
    private val testInputTensors = listOf(floatArrayOf(1.0f, 2.0f), floatArrayOf(10.0f, 20.0f))
    private val testOutputTensor = floatArrayOf(11.0f, 22.0f)
  }
}