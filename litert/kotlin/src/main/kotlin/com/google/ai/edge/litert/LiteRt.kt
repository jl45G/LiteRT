package com.google.ai.edge.litert

import android.content.res.AssetManager
import com.google.ai.edge.litert.acceleration.ModelProvider
import com.google.ai.edge.litert.acceleration.ModelSelector


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

        @JvmStatic private external fun nativeLoadAsset(assetManager: AssetManager, assetName: String): Long
        @JvmStatic private external fun nativeLoadFile(filePath: String): Long
        @JvmStatic private external fun nativeDestroy(handle: Long)
    }
}

/** Represents a compiled model for inference. */
class CompiledModel private constructor(
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
    fun run(inputBuffers: List<TensorBuffer>, outputBuffers: List<TensorBuffer>, signatureIndex: Int = 0) {
        nativeRun(
            handle,
            model.handle,
            signatureIndex,
            inputBuffers.map { it.handle }.toLongArray(),
            outputBuffers.map { it.handle }.toLongArray()
        )
    }

    // NEW: Asynchronous run method
    @JvmOverloads
    fun runAsync(inputBuffers: List<TensorBuffer>, outputBuffers: List<TensorBuffer>, signatureIndex: Int = 0): Boolean {
        return nativeRunAsync(
            handle,
            model.handle,
            signatureIndex,
            inputBuffers.map { it.handle }.toLongArray(),
            outputBuffers.map { it.handle }.toLongArray()
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
            val handle = nativeCreate(env.handle, model.handle, options.accelerators.map { it.value }.toIntArray())
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
            val mod = when (modelProvider.getType()) {
                ModelProvider.Type.ASSET -> Model.load(assetManager!!, modelProvider.getPath())
                ModelProvider.Type.FILE -> Model.load(modelProvider.getPath())
            }
      // return create(model, Options(*modelProvider.getCompatibleAccelerators().toTypedArray()), env)
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
        @JvmStatic private external fun nativeCreate(envHandle: Long, modelHandle: Long, acceleratorCodes: IntArray): Long
        @JvmStatic private external fun nativeCreateInputBuffer(compiledModelHandle: Long, modelHandle: Long, s: String?, inName: String?): Long
        @JvmStatic private external fun nativeCreateOutputBuffer(compiledModelHandle: Long, modelHandle: Long, s: String?, outName: String?): Long
        @JvmStatic private external fun nativeCreateInputBuffers(compiledModelHandle: Long, modelHandle: Long, sigIndex: Int): LongArray
        @JvmStatic private external fun nativeCreateOutputBuffers(compiledModelHandle: Long, modelHandle: Long, sigIndex: Int): LongArray
        @JvmStatic private external fun nativeRun(compiledModelHandle: Long, modelHandle: Long, sigIndex: Int, inputBufs: LongArray, outputBufs: LongArray)
        // NEW: async run
        @JvmStatic private external fun nativeRunAsync(compiledModelHandle: Long, modelHandle: Long, sigIndex: Int, inputBufs: LongArray, outputBufs: LongArray): Boolean
        @JvmStatic private external fun nativeDestroy(handle: Long)
    }
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

    // Zero-copy static constructor
    companion object {
        init {
            System.loadLibrary("litert_jni")
        }

        @JvmStatic
        fun createFromDirectBuffer(elementTypeCode: Int, shape: IntArray, directBuffer: java.nio.ByteBuffer): TensorBuffer {
            val sizeBytes = directBuffer.capacity().toLong()
            val handle = nativeCreateFromDirectBuffer(elementTypeCode, shape, directBuffer, sizeBytes)
            return TensorBuffer(handle)
        }

        // Native zero-copy bridging
        @JvmStatic private external fun nativeCreateFromDirectBuffer(
            elementTypeCode: Int,
            shape: IntArray,
            directBuffer: java.nio.ByteBuffer,
            sizeInBytes: Long
        ): Long
    }

    // Expose partial read/write from direct buffers:
    fun writeFromDirect(srcBuffer: java.nio.ByteBuffer, copyBytes: Long) {
        nativeWriteFromDirect(handle, srcBuffer, copyBytes)
    }

    fun readToDirect(dstBuffer: java.nio.ByteBuffer, copyBytes: Long) {
        nativeReadToDirect(handle, dstBuffer, copyBytes)
    }

    // Native partial read/write
    private external fun nativeWriteFromDirect(tbHandle: Long, srcBuffer: java.nio.ByteBuffer, sizeInBytes: Long)
    private external fun nativeReadToDirect(tbHandle: Long, dstBuffer: java.nio.ByteBuffer, sizeInBytes: Long)

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
