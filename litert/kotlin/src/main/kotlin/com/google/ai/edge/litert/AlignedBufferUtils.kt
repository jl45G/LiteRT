package com.google.ai.edge.litert

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Utility for creating ByteBuffers aligned to a 64-byte boundary.
 *
 * This class uses JNI to determine the actual memory address of direct ByteBuffers
 * and creates properly aligned buffers for optimal performance with LiteRT operations.
 */
object AlignedBufferUtils {
  /** Required memory alignment for LiteRT operations in bytes. */
  private const val LITERT_ALIGNMENT = 64

  init {
    // Load the native library containing the JNI implementation
    System.loadLibrary("litert_jni")
  }

  /**
   * Creates a ByteBuffer aligned to a 64-byte boundary.
   *
   * @param capacityBytes The desired capacity of the buffer in bytes.
   * @return A direct ByteBuffer with the requested capacity, aligned to a 64-byte boundary.
   * @throws RuntimeException If alignment cannot be achieved.
   */
  @JvmStatic
  fun create64ByteAlignedByteBuffer(capacityBytes: Int): ByteBuffer {
    // Allocate extra space to ensure we can achieve alignment
    val buffer = ByteBuffer.allocateDirect(capacityBytes + LITERT_ALIGNMENT - 1)
      .order(ByteOrder.nativeOrder())

    // Get the actual memory address of the buffer
    val baseAddress = nativeGetDirectBufferAddress(buffer)
    if (baseAddress == 0L) {
      throw RuntimeException("Failed to get direct buffer address for alignment")
    }

    // Calculate the aligned address by rounding up to the next multiple of LITERT_ALIGNMENT
    val alignedAddress = (baseAddress + (LITERT_ALIGNMENT - 1)) and 
      ((LITERT_ALIGNMENT - 1).inv().toLong())
    val shift = (alignedAddress - baseAddress).toInt()

    // Create a slice of the original buffer at the aligned position
    buffer.position(shift)
    val alignedBuffer = buffer.slice().order(ByteOrder.nativeOrder())
    alignedBuffer.limit(capacityBytes)

    // Verify the alignment was successful
    val checkAddr = nativeGetDirectBufferAddress(alignedBuffer)
    if (checkAddr == 0L || (checkAddr % LITERT_ALIGNMENT) != 0L) {
      throw RuntimeException("Failed to achieve 64-byte alignment (address=$checkAddr)")
    }
    
    return alignedBuffer
  }

  /**
   * Returns the memory address of a direct ByteBuffer.
   *
   * @param buffer The direct ByteBuffer to get the address for.
   * @return The memory address as a long, or 0 if the operation failed.
   */
  @JvmStatic private external fun nativeGetDirectBufferAddress(buffer: ByteBuffer): Long
}
