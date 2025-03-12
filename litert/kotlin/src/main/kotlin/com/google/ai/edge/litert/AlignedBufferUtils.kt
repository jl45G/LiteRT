package com.google.ai.edge.litert

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Helper methods for creating ByteBuffers aligned to a 64-byte boundary
 * without using reflection. Instead, we call a JNI helper to retrieve
 * the actual pointer.
 */
object AlignedBufferUtils {
  private const val LITERT_ALIGNMENT = 64 // Must match your LiteRT alignment.

  init {
    // Make sure to load the native library containing nativeGetDirectBufferAddress()
    System.loadLibrary("litert_jni")
  }

  /**
   * Creates a 64-byte aligned ByteBuffer with capacity `capacityBytes`.
   * If alignment fails or reflection is blocked, you can handle fallback
   * in your native code, or here.
   */
  @JvmStatic
  fun create64ByteAlignedByteBuffer(capacityBytes: Int): ByteBuffer {
    // Over-allocate by up to 63 bytes
    val big = ByteBuffer.allocateDirect(capacityBytes + LITERT_ALIGNMENT - 1)
      .order(ByteOrder.nativeOrder())

    // Use our native helper to get the pointer:
    val baseAddress = nativeGetDirectBufferAddress(big)
    if (baseAddress == 0L) {
      // Means GetDirectBufferAddress failed or not truly direct
      throw RuntimeException("GetDirectBufferAddress returned 0 -- cannot do alignment!")
    }

    // Round up to next multiple-of-64
    val alignedAddress = (baseAddress + (LITERT_ALIGNMENT - 1)) and ((LITERT_ALIGNMENT - 1).inv().toLong())
    val shift = (alignedAddress - baseAddress).toInt()

    big.position(shift)
    val aligned = big.slice().order(ByteOrder.nativeOrder())
    aligned.limit(capacityBytes)

    // Double-check alignment
    val checkAddr = nativeGetDirectBufferAddress(aligned)
    if (checkAddr == 0L || (checkAddr % LITERT_ALIGNMENT) != 0L) {
      throw RuntimeException("Failed to achieve 64-byte alignment (addr=$checkAddr)")
    }
    return aligned
  }

  /**
   * JNI method that calls env->GetDirectBufferAddress and returns it as jlong.
   */
  @JvmStatic private external fun nativeGetDirectBufferAddress(buffer: ByteBuffer): Long
}
