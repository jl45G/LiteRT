package com.google.ai.edge.litert

import android.hardware.HardwareBuffer
import android.opengl.EGL14
import android.opengl.GLES31
import android.os.Build
import androidx.opengl.EGLExt
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Advanced features for LiteRT to support sync-fence based async execution and zero-copy buffer
 * operations.
 *
 * This class provides Kotlin implementations of features demonstrated in:
 * - litert/cc/litert_compiled_model_gpu_test.cc
 * - litert/cc/litert_compiled_model_integration_test.cc
 */
class LiteRtAdvancedFeatures {

  companion object {
    init {
      System.loadLibrary("litert_jni")
    }

    /**
     * Creates an EGL sync object and returns the associated native fence file descriptor.
     *
     * This is used for GPU-CPU synchronization when using GL buffers.
     *
     * @return The native fence file descriptor, or -1 if failed
     */
    @JvmStatic
    fun createEglSyncAndFenceFd(): Int {
      if (!hasOpenGLSupport()) {
        return -1
      }

      // Create EGL sync object
      val eglSync =
        EGLExt.eglCreateSyncKHR(EGL14.eglGetCurrentDisplay(), EGLExt.EGL_SYNC_FENCE_KHR, null)

      if (eglSync == EGL14.EGL_NO_SYNC_KHR) {
        return -1
      }

      try {
        // Ensure all GL commands are issued before creating sync fence
        GLES31.glFlush()

        // Create a native fence FD from the EGL sync object
        val fdBuffer = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder()).asIntBuffer()
        fdBuffer.position(0)

        val result = EGLExt.eglDupNativeFenceFDANDROID(EGL14.eglGetCurrentDisplay(), eglSync)

        // Destroy the EGL sync object
        EGLExt.eglDestroySyncKHR(EGL14.eglGetCurrentDisplay(), eglSync)

        return result
      } catch (e: Exception) {
        // Clean up on exception
        EGLExt.eglDestroySyncKHR(EGL14.eglGetCurrentDisplay(), eglSync)
        return -1
      }
    }

    /**
     * Checks if the device supports OpenGL ES 3.1.
     *
     * @return True if OpenGL ES 3.1 is supported, false otherwise.
     */
    @JvmStatic
    fun hasOpenGLSupport(): Boolean {
      return try {
        val version = GLES31.glGetString(GLES31.GL_VERSION)
        true
      } catch (e: Exception) {
        false
      }
    }

    /**
     * Checks if AHWB (Android Hardware Buffer) is available on this device.
     *
     * @return True if AHWB is supported, false otherwise.
     */
    @JvmStatic
    fun hasAhwbSupport(): Boolean {
      return Build.VERSION.SDK_INT >= 29
    }

    /**
     * Fills an OpenGL buffer with test data using compute shaders. This is equivalent to
     * FillGlBuffer1/2 in the C++ tests.
     *
     * @param glId The OpenGL buffer ID to fill.
     * @param size The number of float elements in the buffer.
     * @param divFactor The factor to divide values by (1.0 or 0.1 like in C++ tests).
     */
    @JvmStatic
    fun fillGlBuffer(glId: Int, size: Int, divFactor: Float = 1.0f) {
      if (!hasOpenGLSupport()) {
        return
      }

      // Create and compile shader
      val shaderSource =
        """#version 310 es
        precision highp float;
        layout(local_size_x = 1, local_size_y = 1) in;
        layout(std430, binding = 0) buffer Output {float elements[];} output_data;
        void main() {
          uint v = gl_GlobalInvocationID.x * 2u;
          output_data.elements[v] = float(v + 1u) / $divFactor;
          output_data.elements[v + 1u] = float(v + 2u) / $divFactor;
        }
      """
          .trimIndent()

      val shader = GLES31.glCreateShader(GLES31.GL_COMPUTE_SHADER)
      GLES31.glShaderSource(shader, shaderSource)
      GLES31.glCompileShader(shader)

      // Create and link program
      val program = GLES31.glCreateProgram()
      GLES31.glAttachShader(program, shader)
      GLES31.glDeleteShader(shader)
      GLES31.glLinkProgram(program)

      // Bind buffer and dispatch compute shader
      GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 0, glId)
      GLES31.glUseProgram(program)
      GLES31.glDispatchCompute(size / 2, 1, 1)

      // Clean up
      GLES31.glMemoryBarrier(GLES31.GL_BUFFER_UPDATE_BARRIER_BIT)
      GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)
      GLES31.glDeleteProgram(program)
    }

    /**
     * Creates an OpenGL buffer that can be used with LiteRT.
     *
     * @param sizeFloats The number of float elements to allocate.
     * @return The OpenGL buffer ID, or 0 if creation failed.
     */
    @JvmStatic
    fun createGlBuffer(sizeFloats: Int): Int {
      if (!hasOpenGLSupport()) {
        return 0
      }

      val sizeBytes = sizeFloats * 4 // 4 bytes per float

      // Generate buffer
      val bufferIds = IntArray(1)
      GLES31.glGenBuffers(1, bufferIds, 0)
      val bufferId = bufferIds[0]

      // Allocate buffer storage
      GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, bufferId)
      GLES31.glBufferData(GLES31.GL_SHADER_STORAGE_BUFFER, sizeBytes, null, GLES31.GL_DYNAMIC_DRAW)
      GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0)

      return bufferId
    }

    /**
     * Creates an OpenGL texture that can be used with LiteRT.
     *
     * @param width Width of the texture.
     * @param height Height of the texture.
     * @return The OpenGL texture ID, or 0 if creation failed.
     */
    @JvmStatic
    fun createGlTexture(width: Int, height: Int): Int {
      if (!hasOpenGLSupport()) {
        return 0
      }

      // Generate texture
      val textureIds = IntArray(1)
      GLES31.glGenTextures(1, textureIds, 0)
      val textureId = textureIds[0]

      // Initialize texture
      GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, textureId)
      GLES31.glTexStorage2D(GLES31.GL_TEXTURE_2D, 1, GLES31.GL_RGBA8, width, height)

      // Set texture parameters
      GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MIN_FILTER, GLES31.GL_LINEAR)
      GLES31.glTexParameteri(GLES31.GL_TEXTURE_2D, GLES31.GL_TEXTURE_MAG_FILTER, GLES31.GL_LINEAR)
      GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, 0)

      return textureId
    }

    /**
     * Convenience method to create a HardwareBuffer suitable for LiteRT tensor operations.
     *
     * @param width Width in pixels.
     * @param height Height in pixels.
     * @param format The HardwareBuffer format to use.
     * @return A new HardwareBuffer instance, or null if creation failed.
     */
    @JvmStatic
    fun createAhwb(
      width: Int,
      height: Int,
      format: Int = HardwareBuffer.RGBA_8888,
    ): HardwareBuffer? {
      if (!hasAhwbSupport()) {
        return null
      }

      val usage =
        HardwareBuffer.USAGE_CPU_READ_RARELY or
          HardwareBuffer.USAGE_CPU_WRITE_RARELY or
          HardwareBuffer.USAGE_GPU_SAMPLED_IMAGE

      return try {
        HardwareBuffer.create(width, height, format, 1, usage)
      } catch (e: Exception) {
        null
      }
    }
  }
}
