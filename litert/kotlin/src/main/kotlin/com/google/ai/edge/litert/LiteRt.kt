package com.google.ai.edge.litert

import android.content.res.AssetManager
import android.hardware.HardwareBuffer
import com.google.ai.edge.litert.acceleration.ModelProvider
import com.google.ai.edge.litert.acceleration.ModelSelector
import java.nio.ByteBuffer

/** Represents a .tflite model. */
class Model private constructor(internal val handle: Long) {

  fun destroy() {
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

/**
 * A Kotlin “scoped lock” that references a locked [TensorBuffer] region. Under the hood, this calls
 * [LiteRtLockTensorBuffer] once, and calls [LiteRtUnlockTensorBuffer] when [close] is called.
 *
 * Usage: val tb: TensorBuffer = ... TensorBufferScopedLock(tb).use { lock -> // lock.ptr is the
 * native address // do read/writes, etc. } // automatically unlock on exit
 */
class TensorBufferScopedLock private constructor(private val lockHandle: Long) : AutoCloseable {

  /**
   * The direct pointer to the locked memory. You can pass it to other JNI calls if you do custom
   * read/writes.
   */
  val ptr: Long
    get() = nativeGetLockedPointer(lockHandle)

  /** Unlocks the buffer (destroying the lock) if not already done. */
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
     * Creates (and locks) a new [TensorBufferScopedLock] for the given [TensorBuffer]. Returns null
     * if locking fails.
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

/** Represents a compiled model for inference. */
class CompiledModel
private constructor(
  private val handle: Long,
  private val model: Model,
  private val env: Environment,
) {

  /** Options for specifying accelerators. */
  class Options constructor(vararg val accelerators: Accelerator) {
    companion object {
      @JvmField val NONE = Options()
    }
  }

  // Creates single input buffer
  @JvmOverloads
  fun createInputBuffer(signature: String? = null, inputName: String? = null): TensorBuffer {
    val h = nativeCreateInputBuffer(handle, model.handle, signature, inputName)
    return TensorBuffer(h)
  }

  // Creates single output buffer
  @JvmOverloads
  fun createOutputBuffer(signature: String? = null, outputName: String? = null): TensorBuffer {
    val h = nativeCreateOutputBuffer(handle, model.handle, signature, outputName)
    return TensorBuffer(h)
  }

  // Creates all input buffers for a given subgraph (by index)
  @JvmOverloads
  fun createInputBuffers(signatureIndex: Int = 0): List<TensorBuffer> {
    val arr = nativeCreateInputBuffers(handle, model.handle, signatureIndex)
    return arr.map { TensorBuffer(it) }
  }

  // Creates all output buffers for a given subgraph (by index)
  @JvmOverloads
  fun createOutputBuffers(signatureIndex: Int = 0): List<TensorBuffer> {
    val arr = nativeCreateOutputBuffers(handle, model.handle, signatureIndex)
    return arr.map { TensorBuffer(it) }
  }

  // Simple run that auto-creates output buffers
  @JvmOverloads
  fun run(inputBuffers: List<TensorBuffer>, signatureIndex: Int = 0): List<TensorBuffer> {
    val outputBuffers = createOutputBuffers(signatureIndex)
    run(inputBuffers, outputBuffers, signatureIndex)
    return outputBuffers
  }

  // Overload that uses user-provided output buffers
  @JvmOverloads
  fun run(
    inputBuffers: List<TensorBuffer>,
    outputBuffers: List<TensorBuffer>,
    signatureIndex: Int = 0,
  ) {
    nativeRun(
      handle,
      model.handle,
      signatureIndex,
      inputBuffers.map { it.handle }.toLongArray(),
      outputBuffers.map { it.handle }.toLongArray(),
    )
  }

  // NEW: Asynchronous run method
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

  fun destroy() {
    nativeDestroy(handle)
  }

  companion object {
    init {
      System.loadLibrary("litert_jni")
    }

    // Factory method taking a Model, optional environment
    @JvmOverloads
    @JvmStatic
    fun create(
      model: Model,
      options: Options = Options.NONE,
      env: Environment = Environment.create(),
    ): CompiledModel {
      val handle =
        nativeCreate(env.handle, model.handle, options.accelerators.map { it.value }.toIntArray())
      return CompiledModel(handle, model, env)
    }

    // Example using a ModelSelector
    @JvmOverloads
    @JvmStatic
    fun create(
      modelSelector: ModelSelector,
      env: Environment = Environment.create(),
      assetManager: AssetManager? = null,
    ): CompiledModel {
      val modelProvider = modelSelector.selectModel(env)
      val mod =
        when (modelProvider.getType()) {
          ModelProvider.Type.ASSET -> Model.load(assetManager!!, modelProvider.getPath())
          ModelProvider.Type.FILE -> Model.load(modelProvider.getPath())
        }
      // return create(model, Options(*modelProvider.getCompatibleAccelerators().toTypedArray()),
      // env)
      return create(mod, Options(Accelerator.NONE), env)
    }

    @JvmOverloads
    @JvmStatic
    fun create(
      assetManager: AssetManager,
      assetName: String,
      options: Options = Options.NONE,
      env: Environment = Environment.create(),
    ): CompiledModel {
      return create(Model.load(assetManager, assetName), options, env)
    }

    @JvmOverloads
    @JvmStatic
    fun create(
      filePath: String,
      options: Options = Options.NONE,
      env: Environment = Environment.create(),
    ): CompiledModel {
      return create(Model.load(filePath), options, env)
    }

    // Native calls
    @JvmStatic
    private external fun nativeCreate(
      envHandle: Long,
      modelHandle: Long,
      acceleratorCodes: IntArray,
    ): Long

    @JvmStatic
    private external fun nativeCreateInputBuffer(
      compiledModelHandle: Long,
      modelHandle: Long,
      s: String?,
      inName: String?,
    ): Long

    @JvmStatic
    private external fun nativeCreateOutputBuffer(
      compiledModelHandle: Long,
      modelHandle: Long,
      s: String?,
      outName: String?,
    ): Long

    @JvmStatic
    private external fun nativeCreateInputBuffers(
      compiledModelHandle: Long,
      modelHandle: Long,
      sigIndex: Int,
    ): LongArray

    @JvmStatic
    private external fun nativeCreateOutputBuffers(
      compiledModelHandle: Long,
      modelHandle: Long,
      sigIndex: Int,
    ): LongArray

    @JvmStatic
    private external fun nativeRun(
      compiledModelHandle: Long,
      modelHandle: Long,
      sigIndex: Int,
      inputBufs: LongArray,
      outputBufs: LongArray,
    )

    // NEW: async run
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

    // Example event type constants (LiteRtEventType)
    const val TYPE_UNKNOWN = 0
    const val TYPE_SYNC_FENCE = 1
    const val TYPE_OPENCL = 2
    const val TYPE_MANAGED = 3

    // etc.

    @JvmStatic
    fun createFromSyncFenceFd(syncFenceFd: Int, ownsFd: Boolean = false): Event {
      val h = nativeCreateFromSyncFenceFd(syncFenceFd, ownsFd)
      require(h != 0L) { "Failed to create event from fd=$syncFenceFd" }
      return Event(h)
    }

    @JvmStatic
    fun createFromOpenClEvent(clEventHandle: Long): Event {
      val h = nativeCreateFromOpenClEvent(clEventHandle)
      require(h != 0L) { "Failed to create event from CL handle=$clEventHandle" }
      return Event(h)
    }

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

  fun getSyncFenceFd(): Int {
    return nativeGetSyncFenceFd(handle)
  }

  fun getOpenClEvent(): Long {
    return nativeGetOpenClEvent(handle)
  }

  fun getType(): Int {
    return nativeGetType(handle)
  }

  /** Wait on the event. If timeoutMs = -1 => indefinite. */
  fun waitFence(timeoutMs: Long = -1) {
    nativeWait(handle, timeoutMs)
  }

  fun destroy() {
    nativeDestroy(handle)
  }

  private external fun nativeGetSyncFenceFd(eventHandle: Long): Int

  private external fun nativeGetOpenClEvent(eventHandle: Long): Long

  private external fun nativeGetType(eventHandle: Long): Int

  private external fun nativeWait(eventHandle: Long, timeoutMs: Long)

  private external fun nativeDestroy(eventHandle: Long)
}

/** Represents the memory block for a tensor, including zero-copy & events. */
class TensorBuffer internal constructor(internal val handle: Long) {

  // Array-based I/O
  fun writeInt(data: IntArray) {
    nativeWriteInt(handle, data)
  }

  fun writeFloat(data: FloatArray) {
    nativeWriteFloat(handle, data)
  }

  fun readInt(): IntArray {
    return nativeReadInt(handle)
  }

  fun readFloat(): FloatArray {
    return nativeReadFloat(handle)
  }

  // ... existing array-based read/write and direct read/write code ...
  // ... existing event methods (hasEvent, getEventHandle, etc.) ...

  companion object {
    init {
      System.loadLibrary("litert_jni")
    }

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

    // Native zero-copy bridging
    @JvmStatic
    private external fun nativeCreateFromDirectBuffer(
      elementTypeCode: Int,
      shape: IntArray,
      directBuffer: java.nio.ByteBuffer,
      sizeInBytes: Long,
    ): Long

    // ---------- NEW: createFromAhwb ----------
    /**
     * Create from Android HardwareBuffer (API 29+).
     *
     * @param elementTypeCode e.g. 0 => Float32
     * @param shape the shape of the tensor
     * @param hardwareBuffer the Android HardwareBuffer
     * @param offset optional offset in bytes
     */
    @JvmStatic
    fun createFromAhwb(
      elementTypeCode: Int,
      shape: IntArray,
      hardwareBuffer: HardwareBuffer,
      offset: Long = 0,
    ): TensorBuffer {
      // requires android:minSdkVersion >= 29 if you really use it
      val h = nativeCreateFromAhwb(elementTypeCode, shape, hardwareBuffer, offset)
      require(h != 0L) { "CreateFromAhwb failed" }
      return TensorBuffer(h)
    }

    // Retrieve the underlying AHardwareBuffer. May be null if not from AHWB.
    @JvmStatic
    fun getAhwbHandle(tensorBuffer: TensorBuffer): HardwareBuffer? {
      // If < 29, returns null or throws
      return nativeGetAhwb(tensorBuffer.handle) as HardwareBuffer?
    }

    // ---------- NEW: createFromGlTexture ----------
    /** Create from a GL texture (2D, 3D, or array). Must have GL support built in. */
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

    // Retrieve info about the GL texture
    // The array has 6 ints: [target, id, format, sizeBytesLow, sizeBytesHigh, layer]
    @JvmStatic
    fun getGlTextureInfo(tensorBuffer: TensorBuffer): IntArray? {
      return nativeGetGlTexture(tensorBuffer.handle)
    }

    // ---------- NEW: createFromGlBuffer ----------
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

    // Retrieve info about the GL buffer
    // We return a long[] with 6 elements: [target, id, sizeBytes, offset, 0,0]
    @JvmStatic
    fun getGlBufferInfo(tensorBuffer: TensorBuffer): LongArray? {
      return nativeGetGlBuffer(tensorBuffer.handle)
    }

    // -------- Native bridging for the new methods --------
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

  // Expose partial read/write from direct buffers:
  fun writeFromDirect(srcBuffer: java.nio.ByteBuffer, copyBytes: Long) {
    nativeWriteFromDirect(handle, srcBuffer, copyBytes)
  }

  fun readToDirect(dstBuffer: java.nio.ByteBuffer, copyBytes: Long) {
    nativeReadToDirect(handle, dstBuffer, copyBytes)
  }

  // Native partial read/write
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

  // EVENT METHODS
  fun hasEvent(): Boolean = nativeHasEvent(handle)

  fun getEventHandle(): Long = nativeGetEvent(handle)

  fun setEvent(eventHandle: Long) = nativeSetEvent(handle, eventHandle)

  fun clearEvent() = nativeClearEvent(handle)

  fun waitOnEvent(timeoutMs: Long) = nativeWaitOnEvent(handle, timeoutMs)

  private external fun nativeHasEvent(handle: Long): Boolean

  private external fun nativeGetEvent(handle: Long): Long

  private external fun nativeSetEvent(handle: Long, eventHandle: Long)

  private external fun nativeClearEvent(handle: Long)

  private external fun nativeWaitOnEvent(handle: Long, timeoutMs: Long)

  fun destroy() {
    nativeDestroy(handle)
  }

  // Array-based JNI calls
  private external fun nativeWriteInt(handle: Long, data: IntArray)

  private external fun nativeWriteFloat(handle: Long, data: FloatArray)

  private external fun nativeReadInt(handle: Long): IntArray

  private external fun nativeReadFloat(handle: Long): FloatArray

  private external fun nativeDestroy(handle: Long)
}
