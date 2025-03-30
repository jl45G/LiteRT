package com.google.ai.edge.litert

import android.content.res.AssetManager
import android.hardware.HardwareBuffer
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean

/** Kotlin-level wrapper for a LiteRtEvent. */
class Event internal constructor(internal val handle: Long) {

  companion object {
    init {
      System.loadLibrary("litert_jni")
    }

    /** Event type constants corresponding to LiteRtEventType */
    const val TYPE_UNKNOWN = 0
    const val TYPE_SYNC_FENCE = 1
    const val TYPE_OPENCL = 2
    const val TYPE_MANAGED = 3

    /**
     * Creates an Event from a sync fence file descriptor.
     *
     * @param syncFenceFd The sync fence file descriptor to wrap.
     * @param ownsFd Whether this Event should take ownership of the file descriptor.
     * @return A new Event instance wrapping the sync fence.
     * @throws IllegalArgumentException if event creation fails.
     */
    @JvmStatic
    fun createFromSyncFenceFd(syncFenceFd: Int, ownsFd: Boolean = false): Event {
      val h = nativeCreateFromSyncFenceFd(syncFenceFd, ownsFd)
      require(h != 0L) { "Failed to create event from fd=$syncFenceFd" }
      return Event(h)
    }

    /**
     * Creates an Event from an OpenCL event handle.
     *
     * @param clEventHandle The OpenCL event handle to wrap.
     * @return A new Event instance wrapping the OpenCL event.
     * @throws IllegalArgumentException if event creation fails.
     */
    @JvmStatic
    fun createFromOpenClEvent(clEventHandle: Long): Event {
      val h = nativeCreateFromOpenClEvent(clEventHandle)
      require(h != 0L) { "Failed to create event from CL handle=$clEventHandle" }
      return Event(h)
    }

    /**
     * Creates a managed Event of the specified type.
     *
     * @param eventType The type of event to create, defaults to TYPE_UNKNOWN.
     * @return A new managed Event instance.
     * @throws IllegalArgumentException if event creation fails.
     */
    @JvmStatic
    fun createManaged(eventType: Int = TYPE_UNKNOWN): Event {
      val h = nativeCreateManaged(eventType)
      require(h != 0L) { "Failed to create managed event type=$eventType" }
      return Event(h)
    }

    @JvmStatic private external fun nativeCreateFromSyncFenceFd(fd: Int, ownsFd: Boolean): Long

    @JvmStatic private external fun nativeCreateFromOpenClEvent(clEventHandle: Long): Long

    @JvmStatic private external fun nativeCreateManaged(eventType: Int): Long
  }

  /**
   * Retrieves the sync fence file descriptor associated with this event.
   *
   * @return The sync fence file descriptor.
   */
  fun getSyncFenceFd(): Int {
    return nativeGetSyncFenceFd(handle)
  }

  /**
   * Retrieves the OpenCL event handle associated with this event.
   *
   * @return The OpenCL event handle.
   */
  fun getOpenClEvent(): Long {
    return nativeGetOpenClEvent(handle)
  }

  /**
   * Gets the type of this event.
   *
   * @return The event type (one of the TYPE_* constants).
   */
  fun getType(): Int {
    return nativeGetType(handle)
  }

  /**
   * Waits for this event to complete.
   *
   * @param timeoutMs Timeout in milliseconds. Use -1 for indefinite wait.
   */
  fun waitFence(timeoutMs: Long = -1) {
    nativeWait(handle, timeoutMs)
  }

  /** Destroys this event and releases associated resources. */
  fun destroy() {
    nativeDestroy(handle)
  }

  private external fun nativeGetSyncFenceFd(eventHandle: Long): Int

  private external fun nativeGetOpenClEvent(eventHandle: Long): Long

  private external fun nativeGetType(eventHandle: Long): Int

  private external fun nativeWait(eventHandle: Long, timeoutMs: Long)

  private external fun nativeDestroy(eventHandle: Long)
}

/**
 * A scoped lock for a TensorBuffer that automatically unlocks when closed.
 *
 * This class provides RAII-style access to locked TensorBuffer memory. It calls
 * LiteRtLockTensorBuffer on creation and LiteRtUnlockTensorBuffer when closed.
 *
 * Example usage:
 * ```
 * val tensorBuffer: TensorBuffer = ...
 * TensorBufferScopedLock.create(tensorBuffer)?.use { lock ->
 *   // Access the native memory via lock.ptr
 *   // Memory is automatically unlocked when exiting this block
 * }
 * ```
 */
class TensorBufferScopedLock private constructor(private val lockHandle: Long) : AutoCloseable {

  /**
   * The direct pointer to the locked memory region.
   *
   * This pointer can be passed to JNI functions for direct memory access.
   */
  val ptr: Long
    get() = nativeGetLockedPointer(lockHandle)

  /**
   * Unlocks the buffer and releases the lock.
   *
   * This is automatically called when used with use {} or try-with-resources.
   */
  override fun close() {
    if (lockHandle != 0L) {
      nativeDestroyScopedLock(lockHandle)
    }
  }

  companion object {
    init {
      System.loadLibrary("litert_jni")
    }

    /**
     * Creates and locks a new TensorBufferScopedLock for the given TensorBuffer.
     *
     * @param buffer The TensorBuffer to lock.
     * @return A new TensorBufferScopedLock instance, or null if locking fails.
     */
    @JvmStatic
    fun create(buffer: TensorBuffer): TensorBufferScopedLock? {
      val lockH = nativeCreateScopedLock(buffer.handle)
      if (lockH == 0L) return null
      return TensorBufferScopedLock(lockH)
    }

    @JvmStatic private external fun nativeCreateScopedLock(tensorBufferHandle: Long): Long

    @JvmStatic private external fun nativeGetLockedPointer(scopedLockHandle: Long): Long

    @JvmStatic private external fun nativeDestroyScopedLock(scopedLockHandle: Long)
  }
}

/** TensorBuffer represents the raw memory where tensor data is stored. */
class TensorBuffer internal constructor(handle: Long) : JniHandle(handle) {
  // TODO(niuchl): Add support for different types of tensor buffers.
  // TODO(niuchl): Add tests for different element types.

  /**
   * Writes integer data to the tensor buffer.
   *
   * @param data The integer array to write.
   */
  fun writeInt(data: IntArray) {
    assertNotDestroyed()

    nativeWriteInt(handle, data)
  }

  /**
   * Writes float data to the tensor buffer.
   *
   * @param data The float array to write.
   */
  fun writeFloat(data: FloatArray) {
    assertNotDestroyed()

    nativeWriteFloat(handle, data)
  }

  fun writeInt8(data: ByteArray) {
    assertNotDestroyed()

    nativeWriteInt8(handle, data)
  }

  fun writeBoolean(data: BooleanArray) {
    assertNotDestroyed()

    nativeWriteBoolean(handle, data)
  }

  /**
   * Reads integer data from the tensor buffer.
   *
   * @return An integer array containing the tensor data.
   */
  fun readInt(): IntArray {
    assertNotDestroyed()

    return nativeReadInt(handle)
  }

  /**
   * Reads float data from the tensor buffer.
   *
   * @return A float array containing the tensor data.
   */
  fun readFloat(): FloatArray {
    assertNotDestroyed()

    return nativeReadFloat(handle)
  }

  fun readInt8(): ByteArray {
    assertNotDestroyed()

    return nativeReadInt8(handle)
  }

  fun readBoolean(): BooleanArray {
    assertNotDestroyed()

    return nativeReadBoolean(handle)
  }

  protected override fun destroy() {
    nativeDestroy(handle)
  }

  companion object {
    init {
      System.loadLibrary("litert_jni")
    }

    /**
     * Creates a TensorBuffer from a direct ByteBuffer.
     *
     * @param elementTypeCode The type code for tensor elements.
     * @param shape The shape of the tensor.
     * @param directBuffer A direct ByteBuffer containing the tensor data.
     * @return A new TensorBuffer instance.
     */
    @JvmStatic
    fun createFromDirectBuffer(
      elementTypeCode: Int,
      shape: IntArray,
      directBuffer: java.nio.ByteBuffer,
    ): TensorBuffer {
      val sizeBytes = directBuffer.capacity().toLong()
      val handle = nativeCreateFromDirectBuffer(elementTypeCode, shape, directBuffer, sizeBytes)
      return TensorBuffer(handle)
    }

    @JvmStatic
    private external fun nativeCreateFromDirectBuffer(
      elementTypeCode: Int,
      shape: IntArray,
      directBuffer: java.nio.ByteBuffer,
      sizeInBytes: Long,
    ): Long

    /**
     * Creates a TensorBuffer from an Android HardwareBuffer.
     *
     * @param elementTypeCode The type code for tensor elements.
     * @param shape The shape of the tensor.
     * @param hardwareBuffer The Android HardwareBuffer to use.
     * @param offset Optional byte offset into the hardware buffer.
     * @return A new TensorBuffer instance.
     * @throws IllegalArgumentException If creation fails.
     */
    @JvmStatic
    fun createFromAhwb(
      elementTypeCode: Int,
      shape: IntArray,
      hardwareBuffer: HardwareBuffer,
      offset: Long = 0,
    ): TensorBuffer {
      // Note: Requires Android API level 29 or higher
      val h = nativeCreateFromAhwb(elementTypeCode, shape, hardwareBuffer, offset)
      require(h != 0L) { "CreateFromAhwb failed" }
      return TensorBuffer(h)
    }

    /**
     * Retrieves the underlying Android HardwareBuffer from a TensorBuffer.
     *
     * @param tensorBuffer The TensorBuffer to query.
     * @return The HardwareBuffer if the TensorBuffer was created from one, null otherwise.
     */
    @JvmStatic
    fun getAhwbHandle(tensorBuffer: TensorBuffer): HardwareBuffer? {
      return nativeGetAhwb(tensorBuffer.handle) as HardwareBuffer?
    }

    @JvmStatic private external fun nativeWriteInt(handle: Long, data: IntArray)

    @JvmStatic private external fun nativeWriteFloat(handle: Long, data: FloatArray)

    @JvmStatic private external fun nativeWriteInt8(handle: Long, data: ByteArray)

    @JvmStatic private external fun nativeWriteBoolean(handle: Long, data: BooleanArray)

    @JvmStatic private external fun nativeReadInt(handle: Long): IntArray

    @JvmStatic private external fun nativeReadFloat(handle: Long): FloatArray

    @JvmStatic private external fun nativeReadInt8(handle: Long): ByteArray

    @JvmStatic private external fun nativeReadBoolean(handle: Long): BooleanArray

    @JvmStatic private external fun nativeDestroy(handle: Long)

    /**
     * Creates a TensorBuffer from an OpenGL texture.
     *
     * @param elementTypeCode The type code for tensor elements.
     * @param shape The shape of the tensor.
     * @param glTarget The OpenGL texture target (e.g., GL_TEXTURE_2D).
     * @param glId The OpenGL texture ID.
     * @param glFormat The OpenGL texture format.
     * @param sizeBytes The size of the texture data in bytes.
     * @param layer The texture layer for array textures (default 0).
     * @return A new TensorBuffer instance.
     * @throws IllegalArgumentException If creation fails.
     */
    @JvmStatic
    fun createFromGlTexture(
      elementTypeCode: Int,
      shape: IntArray,
      glTarget: Int,
      glId: Int,
      glFormat: Int,
      sizeBytes: Long,
      layer: Int = 0,
    ): TensorBuffer {
      val h =
        nativeCreateFromGlTexture(
          elementTypeCode,
          shape,
          glTarget,
          glId,
          glFormat,
          sizeBytes,
          layer,
        )
      require(h != 0L) { "CreateFromGlTexture failed" }
      return TensorBuffer(h)
    }

    /**
     * Retrieves information about the OpenGL texture associated with a TensorBuffer.
     *
     * @param tensorBuffer The TensorBuffer to query.
     * @return An array of 6 integers containing
     *   [target, id, format, sizeBytesLow, sizeBytesHigh, layer], or null if the TensorBuffer is
     *   not associated with a GL texture.
     */
    @JvmStatic
    fun getGlTextureInfo(tensorBuffer: TensorBuffer): IntArray? {
      return nativeGetGlTexture(tensorBuffer.handle)
    }

    /**
     * Creates a TensorBuffer from an OpenGL buffer.
     *
     * @param elementTypeCode The type code for tensor elements.
     * @param shape The shape of the tensor.
     * @param glTarget The OpenGL buffer target (e.g., GL_ARRAY_BUFFER).
     * @param glId The OpenGL buffer ID.
     * @param sizeBytes The size of the buffer data in bytes.
     * @param offset The byte offset into the buffer.
     * @return A new TensorBuffer instance.
     * @throws IllegalArgumentException If creation fails.
     */
    @JvmStatic
    fun createFromGlBuffer(
      elementTypeCode: Int,
      shape: IntArray,
      glTarget: Int,
      glId: Int,
      sizeBytes: Long,
      offset: Long,
    ): TensorBuffer {
      val h = nativeCreateFromGlBuffer(elementTypeCode, shape, glTarget, glId, sizeBytes, offset)
      require(h != 0L) { "CreateFromGlBuffer failed" }
      return TensorBuffer(h)
    }

    /**
     * Retrieves information about the OpenGL buffer associated with a TensorBuffer.
     *
     * @param tensorBuffer The TensorBuffer to query.
     * @return An array of 6 longs containing [target, id, sizeBytes, offset, 0, 0], or null if the
     *   TensorBuffer is not associated with a GL buffer.
     */
    @JvmStatic
    fun getGlBufferInfo(tensorBuffer: TensorBuffer): LongArray? {
      return nativeGetGlBuffer(tensorBuffer.handle)
    }

    @JvmStatic
    private external fun nativeCreateFromAhwb(
      elementTypeCode: Int,
      shape: IntArray,
      hardwareBuffer: Any, // HardwareBuffer
      offset: Long,
    ): Long

    @JvmStatic private external fun nativeGetAhwb(tensorBufferHandle: Long): Any?

    @JvmStatic
    private external fun nativeCreateFromGlTexture(
      elementTypeCode: Int,
      shape: IntArray,
      glTarget: Int,
      glId: Int,
      glFormat: Int,
      sizeBytes: Long,
      layer: Int,
    ): Long

    @JvmStatic private external fun nativeGetGlTexture(tensorBufferHandle: Long): IntArray?

    @JvmStatic
    private external fun nativeCreateFromGlBuffer(
      elementTypeCode: Int,
      shape: IntArray,
      glTarget: Int,
      glId: Int,
      sizeBytes: Long,
      offset: Long,
    ): Long

    @JvmStatic private external fun nativeGetGlBuffer(tensorBufferHandle: Long): LongArray?
  }

  /**
   * Writes data from a direct ByteBuffer to this tensor buffer.
   *
   * @param srcBuffer The source direct ByteBuffer.
   * @param copyBytes The number of bytes to copy.
   */
  fun writeFromDirect(srcBuffer: java.nio.ByteBuffer, copyBytes: Long) {
    nativeWriteFromDirect(handle, srcBuffer, copyBytes)
  }

  /**
   * Reads data from this tensor buffer into a direct ByteBuffer.
   *
   * @param dstBuffer The destination direct ByteBuffer.
   * @param copyBytes The number of bytes to copy.
   */
  fun readToDirect(dstBuffer: java.nio.ByteBuffer, copyBytes: Long) {
    nativeReadToDirect(handle, dstBuffer, copyBytes)
  }

  private external fun nativeWriteFromDirect(
    tbHandle: Long,
    srcBuffer: java.nio.ByteBuffer,
    sizeInBytes: Long,
  )

  private external fun nativeReadToDirect(
    tbHandle: Long,
    dstBuffer: java.nio.ByteBuffer,
    sizeInBytes: Long,
  )

  /**
   * Checks if this tensor buffer has an associated event.
   *
   * @return True if an event is associated with this buffer, false otherwise.
   */
  fun hasEvent(): Boolean = nativeHasEvent(handle)

  /**
   * Gets the handle of the event associated with this tensor buffer.
   *
   * @return The event handle.
   */
  fun getEventHandle(): Long = nativeGetEvent(handle)

  /**
   * Associates an event with this tensor buffer.
   *
   * @param eventHandle The event handle to associate.
   */
  fun setEvent(eventHandle: Long) = nativeSetEvent(handle, eventHandle)

  /** Clears any event associated with this tensor buffer. */
  fun clearEvent() = nativeClearEvent(handle)

  /**
   * Waits for the event associated with this tensor buffer to complete.
   *
   * @param timeoutMs The timeout in milliseconds.
   */
  fun waitOnEvent(timeoutMs: Long) = nativeWaitOnEvent(handle, timeoutMs)

  private external fun nativeHasEvent(handle: Long): Boolean

  private external fun nativeGetEvent(handle: Long): Long

  private external fun nativeSetEvent(handle: Long, eventHandle: Long)

  private external fun nativeClearEvent(handle: Long)

  private external fun nativeWaitOnEvent(handle: Long, timeoutMs: Long)
}
