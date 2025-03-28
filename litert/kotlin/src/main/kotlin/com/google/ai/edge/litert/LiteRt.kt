package com.google.ai.edge.litert

import android.content.res.AssetManager
import android.hardware.HardwareBuffer
import com.google.ai.edge.litert.acceleration.ModelProvider
import com.google.ai.edge.litert.acceleration.ModelSelector
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean

// TODO(niuchl): propagate errors from native code to Kotlin.

/** Model represents a LiteRT model file. */
class Model private constructor(handle: Long) : JniHandle(handle) {

  protected override fun destroy() {
    nativeDestroy(handle)
  }

  companion object {
    init {
      System.loadLibrary("litert_jni")
    }

    @JvmStatic
    fun load(assetManager: AssetManager, assetName: String): Model {
      return Model(nativeLoadAsset(assetManager, assetName))
    }

    @JvmStatic
    fun load(filePath: String): Model {
      return Model(nativeLoadFile(filePath))
    }

    @JvmStatic
    private external fun nativeLoadAsset(assetManager: AssetManager, assetName: String): Long

    @JvmStatic private external fun nativeLoadFile(filePath: String): Long

    @JvmStatic private external fun nativeDestroy(handle: Long)
  }
}

/** Class that represents a compiled LiteRT model. */
class CompiledModel
private constructor(
  handle: Long,
  private val model: Model,
  private val env: Environment,
  private val modelManaged: Boolean = false,
  private val envManaged: Boolean = false,
) : JniHandle(handle) {

  /** Options to specify hardware acceleration for compiling a model. */
  class Options constructor(internal vararg val accelerators: Accelerator) {

    companion object {
      @JvmStatic val NONE = Options()
    }
  }

  fun createInputBuffer(inputName: String, signature: String? = null): TensorBuffer {
    assertNotDestroyed()

    val handle = nativeCreateInputBuffer(handle, model.handle, signature, inputName)
    return TensorBuffer(handle)
  }

  fun createOutputBuffer(outputName: String, signature: String? = null): TensorBuffer {
    assertNotDestroyed()

    val handle = nativeCreateOutputBuffer(handle, model.handle, signature, outputName)
    return TensorBuffer(handle)
  }

  @JvmOverloads
  fun createInputBuffers(signatureIndex: Int = 0): List<TensorBuffer> {
    assertNotDestroyed()

    val handles = nativeCreateInputBuffers(handle, model.handle, signatureIndex)
    return handles.map { TensorBuffer(it) }
  }

  fun createInputBuffers(signature: String): List<TensorBuffer> {
    assertNotDestroyed()

    val handles = nativeCreateInputBuffersBySignature(handle, model.handle, signature)
    return handles.map { TensorBuffer(it) }
  }

  @JvmOverloads
  fun createOutputBuffers(signatureIndex: Int = 0): List<TensorBuffer> {
    assertNotDestroyed()

    val handles = nativeCreateOutputBuffers(handle, model.handle, signatureIndex)
    return handles.map { TensorBuffer(it) }
  }

  fun createOutputBuffers(signature: String): List<TensorBuffer> {
    assertNotDestroyed()

    val handles = nativeCreateOutputBuffersBySignature(handle, model.handle, signature)
    return handles.map { TensorBuffer(it) }
  }

  @JvmOverloads
  fun run(inputs: List<TensorBuffer>, signatureIndex: Int = 0): List<TensorBuffer> {
    assertNotDestroyed()

    val outputs = createOutputBuffers(signatureIndex)
    run(inputs, outputs, signatureIndex)
    return outputs
  }

  fun run(inputs: List<TensorBuffer>, signature: String): List<TensorBuffer> {
    assertNotDestroyed()

    val outputs = createOutputBuffers(signature)
    run(inputs, outputs, signature)
    return outputs
  }

  @JvmOverloads
  fun run(inputs: List<TensorBuffer>, outputs: List<TensorBuffer>, signatureIndex: Int = 0) {
    assertNotDestroyed()

    nativeRun(
      handle,
      model.handle,
      signatureIndex,
      inputs.map { it.handle }.toLongArray(),
      outputs.map { it.handle }.toLongArray(),
    )
  }

  fun run(inputs: List<TensorBuffer>, outputs: List<TensorBuffer>, signature: String) {
    assertNotDestroyed()

    nativeRunBySignature(
      handle,
      model.handle,
      signature,
      inputs.map { it.handle }.toLongArray(),
      outputs.map { it.handle }.toLongArray(),
    )
  }

  fun run(
    inputs: Map<String, TensorBuffer>,
    outputs: Map<String, TensorBuffer>,
    signature: String? = null,
  ) {
    assertNotDestroyed()

    nativeRunBySignatureWithMap(
      handle,
      model.handle,
      signature,
      inputs.keys.toTypedArray(),
      inputs.values.map { it.handle }.toLongArray(),
      outputs.keys.toTypedArray(),
      outputs.values.map { it.handle }.toLongArray(),
    )
  }

  protected override fun destroy() {
    nativeDestroy(handle)
    if (modelManaged) {
      model.close()
    }
    if (envManaged) {
      env.close()
    }
  }

  /**
   * Executes the model asynchronously with the provided input and output buffers.
   *
   * @param inputBuffers The list of input tensor buffers.
   * @param outputBuffers The list of output tensor buffers where results will be stored.
   * @param signatureIndex The index of the signature to use, defaults to 0.
   * @return True if the asynchronous execution was successfully queued, false otherwise.
   */
  @JvmOverloads
  fun runAsync(
    inputBuffers: List<TensorBuffer>,
    outputBuffers: List<TensorBuffer>,
    signatureIndex: Int = 0,
  ): Boolean {
    return nativeRunAsync(
      handle,
      model.handle,
      signatureIndex,
      inputBuffers.map { it.handle }.toLongArray(),
      outputBuffers.map { it.handle }.toLongArray(),
    )
  }

  companion object {
    init {
      System.loadLibrary("litert_jni")
    }

    private fun create(
      model: Model,
      options: Options = Options.NONE,
      optionalEnv: Environment? = null,
      modelManaged: Boolean,
      envManaged: Boolean = optionalEnv == null,
    ): CompiledModel {
      val env = optionalEnv ?: Environment.create()
      return CompiledModel(
        nativeCreate(env.handle, model.handle, options.accelerators.map { it.value }.toIntArray()),
        model,
        env,
        modelManaged,
        envManaged,
      )
    }

    @JvmOverloads
    @JvmStatic
    fun create(
      model: Model,
      options: Options = Options.NONE,
      optionalEnv: Environment? = null,
    ): CompiledModel {
      return create(model, options, optionalEnv, modelManaged = false)
    }

    @JvmOverloads
    @JvmStatic
    fun create(
      modelSelector: ModelSelector,
      optionalEnv: Environment? = null,
      assetManager: AssetManager? = null,
    ): CompiledModel {
      val env = optionalEnv ?: Environment.create()
      val modelProvider = modelSelector.selectModel(env)
      val model =
        when (modelProvider.getType()) {
          ModelProvider.Type.ASSET -> Model.load(assetManager!!, modelProvider.getPath())
          ModelProvider.Type.FILE -> Model.load(modelProvider.getPath())
        }
      return create(
        model,
        Options(*modelProvider.getCompatibleAccelerators().toTypedArray()),
        env,
        modelManaged = true,
        envManaged = optionalEnv == null,
      )
    }

    @JvmOverloads
    @JvmStatic
    fun create(
      assetManager: AssetManager,
      assetName: String,
      options: Options = Options.NONE,
      optionalEnv: Environment? = null,
    ): CompiledModel {
      return create(Model.load(assetManager, assetName), options, optionalEnv, modelManaged = true)
    }

    @JvmOverloads
    @JvmStatic
    fun create(
      filePath: String,
      options: Options = Options.NONE,
      optionalEnv: Environment? = null,
    ): CompiledModel {
      return create(Model.load(filePath), options, optionalEnv, modelManaged = true)
    }

    @JvmStatic
    private external fun nativeCreate(envHandle: Long, modelHandle: Long, options: IntArray): Long

    @JvmStatic
    private external fun nativeCreateInputBuffer(
      compiledModelHandle: Long,
      modelHandle: Long,
      signature: String?,
      inputName: String,
    ): Long

    @JvmStatic
    private external fun nativeCreateOutputBuffer(
      compiledModelHandle: Long,
      modelHandle: Long,
      signature: String?,
      outputName: String,
    ): Long

    @JvmStatic
    private external fun nativeCreateInputBuffers(
      compiledModelHandle: Long,
      modelHandle: Long,
      signatureIndex: Int,
    ): LongArray

    @JvmStatic
    private external fun nativeCreateInputBuffersBySignature(
      compiledModelHandle: Long,
      modelHandle: Long,
      signature: String,
    ): LongArray

    @JvmStatic
    private external fun nativeCreateOutputBuffers(
      compiledModelHandle: Long,
      modelHandle: Long,
      signatureIndex: Int,
    ): LongArray

    @JvmStatic
    private external fun nativeCreateOutputBuffersBySignature(
      compiledModelHandle: Long,
      modelHandle: Long,
      signature: String,
    ): LongArray

    @JvmStatic
    private external fun nativeRun(
      compiledModelHandle: Long,
      modelHandle: Long,
      signatureIndex: Int,
      inputBuffers: LongArray,
      outputBuffers: LongArray,
    )

    @JvmStatic
    private external fun nativeRunBySignature(
      compiledModelHandle: Long,
      modelHandle: Long,
      signature: String,
      inputBuffers: LongArray,
      outputBuffers: LongArray,
    )

    @JvmStatic
    private external fun nativeRunBySignatureWithMap(
      compiledModelHandle: Long,
      modelHandle: Long,
      signature: String?,
      inputKeys: Array<String>,
      inputBuffers: LongArray,
      outputKeys: Array<String>,
      outputBuffers: LongArray,
    )

    /**
     * Asynchronously executes the model with the given inputs and outputs.
     *
     * @param compiledModelHandle Handle to the compiled model.
     * @param modelHandle Handle to the model.
     * @param sigIndex Index of the signature to run.
     * @param inputBufs Array of input buffer handles.
     * @param outputBufs Array of output buffer handles.
     * @return True if the asynchronous execution was successfully queued, false otherwise.
     */
    @JvmStatic
    private external fun nativeRunAsync(
      compiledModelHandle: Long,
      modelHandle: Long,
      sigIndex: Int,
      inputBufs: LongArray,
      outputBufs: LongArray,
    ): Boolean

    @JvmStatic private external fun nativeDestroy(handle: Long)
  }
}

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

  @JvmStatic private external fun nativeReadInt8(handle: Long): ByteArray

  @JvmStatic private external fun nativeReadBoolean(handle: Long): BooleanArray

  @JvmStatic private external fun nativeDestroy(handle: Long)

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

  private external fun nativeWriteInt(handle: Long, data: IntArray)

  private external fun nativeWriteFloat(handle: Long, data: FloatArray)

  private external fun nativeReadInt(handle: Long): IntArray

  private external fun nativeReadFloat(handle: Long): FloatArray

  private external fun nativeDestroy(handle: Long)
}
