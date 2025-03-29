package org.tensorflow.lite.examples.classification.test

import android.opengl.GLES31
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.ai.edge.litert.*
import org.tensorflow.lite.examples.classification.R
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * Activity for testing advanced LiteRT features that may not work in emulators.
 * 
 * This activity provides a UI to run tests that demonstrate:
 * 1. Sync fence based async execution
 * 2. Zero copy buffer interop (GL to AHWB)
 * 
 * To launch this activity from adb:
 * adb shell am start -n org.tensorflow.lite.examples.classification/.test.LiteRtInteropActivity
 */
class LiteRtInteropActivity : AppCompatActivity(), GLSurfaceView.Renderer {
    private val TAG = "LiteRTInteropActivity"
    
    private lateinit var glSurfaceView: GLSurfaceView
    private lateinit var resultsTextView: TextView
    private lateinit var runTestsButton: Button
    private lateinit var logBuilder: StringBuilder
    
    // OpenGL resources
    private var glBufferId1 = 0
    private var glBufferId2 = 0
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_litert_interop)
        
        // Initialize views
        glSurfaceView = findViewById(R.id.gl_surface_view)
        resultsTextView = findViewById(R.id.results_text)
        runTestsButton = findViewById(R.id.run_tests_button)
        logBuilder = StringBuilder()
        
        // Configure GLSurfaceView
        glSurfaceView.setEGLContextClientVersion(3)
        glSurfaceView.setRenderer(this)
        glSurfaceView.renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY
        
        // Set up button click listener
        runTestsButton.setOnClickListener {
            runTestsButton.isEnabled = false
            logBuilder.clear()
            resultsTextView.text = "Running tests..."
            
            // Request a render to ensure GL context is current
            glSurfaceView.queueEvent {
                runAllTests()
                
                runOnUiThread {
                    resultsTextView.text = logBuilder.toString()
                    runTestsButton.isEnabled = true
                }
            }
        }
    }
    
    private fun log(message: String) {
        Log.d(TAG, message)
        logBuilder.append(message).append("\n")
    }
    
    private fun runAllTests() {
        try {
            // Check for required capabilities
            log("Checking device capabilities...")
            
            val hasGpu = Environment.create().use { env ->
                val accelerators = env.getAvailableAccelerators()
                log("Available accelerators: $accelerators")
                accelerators.contains(Accelerator.GPU)
            }
            
            val hasOpenGL = LiteRtAdvancedFeatures.hasOpenGLSupport()
            log("OpenGL ES 3.1 support: $hasOpenGL")
            
            val hasAhwb = LiteRtAdvancedFeatures.hasAhwbSupport()
            log("AHWB support: $hasAhwb")
            
            if (hasGpu) {
                try {
                    runAsyncExecutionTest()
                    log("✅ Async execution test passed")
                } catch (e: Exception) {
                    log("❌ Async execution test failed: ${e.message}")
                    e.printStackTrace()
                }
                
                if (hasOpenGL) {
                    try {
                        runGlBufferInteropTest()
                        log("✅ GL buffer interop test passed")
                    } catch (e: Exception) {
                        log("❌ GL buffer interop test failed: ${e.message}")
                        e.printStackTrace()
                    }
                    
                    if (hasAhwb) {
                        try {
                            runGlAhwbInteropTest()
                            log("✅ GL-AHWB interop test passed")
                        } catch (e: Exception) {
                            log("❌ GL-AHWB interop test failed: ${e.message}")
                            e.printStackTrace()
                        }
                    } else {
                        log("⚠️ AHWB not available, skipping GL-AHWB test")
                    }
                } else {
                    log("⚠️ OpenGL not available, skipping GL tests")
                }
            } else {
                log("⚠️ GPU not available, skipping all tests")
            }
        } catch (e: Exception) {
            log("❌ Test setup failed: ${e.message}")
            e.printStackTrace()
        }
    }
    
    private fun runAsyncExecutionTest() {
        val env = Environment.create()
        val options = CompiledModel.Options(Accelerator.GPU)
        
        CompiledModel.create(assets, "simple_model.tflite", options, env).use { compiledModel ->
            // Create input/output buffers
            val inputBuffers = compiledModel.createInputBuffers()
            val outputBuffers = compiledModel.createOutputBuffers()
            
            // Create managed event
            val event = Event.createManaged(Event.TYPE_MANAGED)
            
            // Fill input buffers with test data
            inputBuffers[0].writeFloat(testInputTensors[0])
            inputBuffers[1].writeFloat(testInputTensors[1])
            
            // Set the event on the first input buffer
            inputBuffers[0].setEvent(event.handle)
            
            // Run asynchronously
            val success = compiledModel.runAsync(inputBuffers, outputBuffers)
            if (!success) {
                throw RuntimeException("Failed to start async execution")
            }
            
            // Signal the event to proceed
            event.waitFence(0)
            
            // Wait for output if needed
            if (outputBuffers[0].hasEvent()) {
                outputBuffers[0].waitOnEvent(-1)
            }
            
            // Verify output
            val output = outputBuffers[0].readFloat()
            verifyOutput(output, testOutputTensor)
            
            // Clean up
            inputBuffers.forEach { it.close() }
            outputBuffers.forEach { it.close() }
            event.destroy()
            env.close()
        }
    }
    
    private fun runGlBufferInteropTest() {
        val env = Environment.create()
        val options = CompiledModel.Options(Accelerator.GPU)
        
        // Create OpenGL buffers for input tensors
        val bufferSize1 = testInputTensors[0].size
        val bufferSize2 = testInputTensors[1].size
        
        if (glBufferId1 == 0 || glBufferId2 == 0) {
            throw RuntimeException("GL buffers not created. Make sure onSurfaceCreated was called.")
        }
        
        // Fill GL buffers with test data
        LiteRtAdvancedFeatures.fillGlBuffer(glBufferId1, bufferSize1, 1.0f)
        LiteRtAdvancedFeatures.fillGlBuffer(glBufferId2, bufferSize2, 0.1f)
        
        CompiledModel.create(assets, "simple_model.tflite", options, env).use { compiledModel ->
            // Create input buffers from GL buffers
            val inputBuffer1 = TensorBuffer.createFromGlBuffer(
                0, // float32
                intArrayOf(1, bufferSize1),
                GLES31.GL_SHADER_STORAGE_BUFFER,
                glBufferId1,
                (bufferSize1 * 4).toLong(),
                0L
            )
            
            val inputBuffer2 = TensorBuffer.createFromGlBuffer(
                0, // float32
                intArrayOf(1, bufferSize2),
                GLES31.GL_SHADER_STORAGE_BUFFER,
                glBufferId2,
                (bufferSize2 * 4).toLong(),
                0L
            )
            
            // Create output buffers
            val outputBuffers = compiledModel.createOutputBuffers()
            
            // Run inference
            compiledModel.run(listOf(inputBuffer1, inputBuffer2), outputBuffers)
            
            // Verify output
            val output = outputBuffers[0].readFloat()
            
            // Expected output based on shader calculations
            val expectedOutput = floatArrayOf(11.0f, 22.0f)
            verifyOutput(output, expectedOutput)
            
            // Clean up
            inputBuffer1.close()
            inputBuffer2.close()
            outputBuffers.forEach { it.close() }
            env.close()
        }
    }
    
    private fun runGlAhwbInteropTest() {
        val env = Environment.create()
        val options = CompiledModel.Options(Accelerator.GPU)
        
        CompiledModel.create(assets, "simple_model.tflite", options, env).use { compiledModel ->
            // Create input/output buffers
            val inputBuffers = compiledModel.createInputBuffers()
            val outputBuffers = compiledModel.createOutputBuffers()
            
            // Get GL buffer handles
            val glBuffer1 = inputBuffers[0].getGlBufferInfo()
            val glBuffer2 = inputBuffers[1].getGlBufferInfo()
            
            if (glBuffer1 == null || glBuffer2 == null) {
                throw RuntimeException("GL buffer handles not available")
            }
            
            val glId1 = glBuffer1[1].toInt()
            val glId2 = glBuffer2[1].toInt()
            
            // Fill GL buffers with test data
            LiteRtAdvancedFeatures.fillGlBuffer(glId1, 2, 1.0f)
            LiteRtAdvancedFeatures.fillGlBuffer(glId2, 2, 0.1f)
            
            // Create sync fence
            val nativeFence = LiteRtAdvancedFeatures.createEglSyncAndFenceFd()
            if (nativeFence == -1) {
                throw RuntimeException("Failed to create EGL sync fence")
            }
            
            // Create events from fence
            val event1 = Event.createFromSyncFenceFd(nativeFence, false)
            val event2 = Event.createFromSyncFenceFd(nativeFence, false)
            
            // Set events on buffers
            inputBuffers[0].setEvent(event1.handle)
            inputBuffers[1].setEvent(event2.handle)
            
            // Run async
            val success = compiledModel.runAsync(inputBuffers, outputBuffers)
            if (!success) {
                throw RuntimeException("Failed to start async execution")
            }
            
            // Wait for output
            if (outputBuffers[0].hasEvent()) {
                outputBuffers[0].waitOnEvent(-1)
            }
            
            // Verify output
            val output = outputBuffers[0].readFloat()
            
            // Expected output based on shader calculations
            val expectedOutput = floatArrayOf(11.0f, 22.0f)
            verifyOutput(output, expectedOutput)
            
            // Clean up
            inputBuffers.forEach { it.close() }
            outputBuffers.forEach { it.close() }
            env.close()
        }
    }
    
    private fun verifyOutput(output: FloatArray, expected: FloatArray) {
        if (output.size != expected.size) {
            throw RuntimeException("Output size mismatch: ${output.size} vs ${expected.size}")
        }
        
        for (i in output.indices) {
            val diff = Math.abs(output[i] - expected[i])
            if (diff > 1e-4f) {
                throw RuntimeException("Output value mismatch at index $i: ${output[i]} vs ${expected[i]}")
            }
        }
    }
    
    // GLSurfaceView.Renderer implementation
    override fun onSurfaceCreated(gl: GL10, config: EGLConfig) {
        // Create OpenGL buffers for tests
        if (glBufferId1 == 0) {
            glBufferId1 = LiteRtAdvancedFeatures.createGlBuffer(testInputTensors[0].size)
        }
        
        if (glBufferId2 == 0) {
            glBufferId2 = LiteRtAdvancedFeatures.createGlBuffer(testInputTensors[1].size)
        }
        
        Log.d(TAG, "GL surface created, buffers: $glBufferId1, $glBufferId2")
    }
    
    override fun onSurfaceChanged(gl: GL10, width: Int, height: Int) {
        // Not used
    }
    
    override fun onDrawFrame(gl: GL10) {
        // Not used - we don't need continuous rendering
    }
    
    override fun onDestroy() {
        super.onDestroy()
        
        // Clean up GL resources
        if (glBufferId1 != 0 || glBufferId2 != 0) {
            glSurfaceView.queueEvent {
                if (glBufferId1 != 0) {
                    GLES31.glDeleteBuffers(1, intArrayOf(glBufferId1), 0)
                    glBufferId1 = 0
                }
                
                if (glBufferId2 != 0) {
                    GLES31.glDeleteBuffers(1, intArrayOf(glBufferId2), 0)
                    glBufferId2 = 0
                }
            }
        }
    }
    
    companion object {
        private val testInputTensors = listOf(floatArrayOf(1.0f, 2.0f), floatArrayOf(10.0f, 20.0f))
        private val testOutputTensor = floatArrayOf(11.0f, 22.0f)
    }
}