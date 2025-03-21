package com.google.ai.edge.litert

import com.google.ai.edge.litert.acceleration.NpuAcceleratorProvider
import java.util.concurrent.atomic.AtomicBoolean

/** Hardware accelerators supported by LiteRT. */
enum class Accelerator private constructor(val value: Int) {
  NONE(0),
  CPU(1),
  GPU(2),
  NPU(3);

  companion object {
    /** Converts an integer value to an [Accelerator]. */
    internal fun of(value: Int): Accelerator {
      return when (value) {
        NONE.value -> NONE
        CPU.value -> CPU
        GPU.value -> GPU
        NPU.value -> NPU
        else -> throw IllegalArgumentException("Invalid accelerator value: $value")
      }
    }
  }
}

/** A base class for all Kotlin types that wrap a JNI handle. */
abstract class JniHandle internal constructor(internal val handle: Long) : AutoCloseable {
  /** Whether the handle has been destroyed. */
  private val destroyed = AtomicBoolean(false)

  /** Asserts that the handle is not destroyed, otherwise throws an [IllegalStateException]. */
  protected fun assertNotDestroyed() {
    if (destroyed.get()) {
      throw IllegalStateException("The handle has been destroyed.")
    }
  }

  /** Clean up resources associated with the handle. */
  protected abstract fun destroy()

  /** Clean up the handle safely to avoid releasing the same JNI handle multiple times. */
  override final fun close() {
    if (destroyed.compareAndSet(false, true)) {
      destroy()
    }
  }
}

/** Environment to hold configuration options for LiteRT runtime. */
class Environment private constructor(handle: Long) : JniHandle(handle) {

  /** Options configurable in LiteRT environment. */
  enum class Option private constructor(val value: Int) {
    CompilerPluginLibraryDir(0),
    DispatchLibraryDir(1),
  }

  override protected fun destroy() {
    nativeDestroy(handle)
  }

  /** Returns the set of accelerators available in the environment. */
  fun getAvailableAccelerators(): Set<Accelerator> {
    assertNotDestroyed()

    val accelerators = nativeGetAvailableAccelerators(handle)
    return accelerators
      .map { Accelerator.of(it) }
      .toMutableSet()
      .apply { add(Accelerator.CPU) } // CPU is always available.
      .toSet()
  }

  companion object {
    init {
      System.loadLibrary("litert_jni")
    }

    @JvmOverloads
    @JvmStatic
    fun create(options: Map<Option, String> = mapOf()): Environment {
      return Environment(
        nativeCreate(options.keys.map { it.value }.toIntArray(), options.values.toTypedArray())
      )
    }

    // TODO(niuchl): Add support for downloading NPU library.
    /**
     * Creates an environment with a [NpuAcceleratorProvider], which provides the NPU libraries.
     *
     * @param npuAcceleratorProvider The NPU accelerator provider.
     * @param options The options to configure the environment.
     */
    @JvmOverloads
    @JvmStatic
    fun create(
      npuAcceleratorProvider: NpuAcceleratorProvider,
      options: Map<Option, String> = mapOf(),
    ): Environment {
      val mutableOptions = options.toMutableMap()
      if (npuAcceleratorProvider.isDeviceSupported() && npuAcceleratorProvider.isLibraryReady()) {
        mutableOptions[Option.DispatchLibraryDir] = npuAcceleratorProvider.getLibraryDir()
        mutableOptions[Option.CompilerPluginLibraryDir] = npuAcceleratorProvider.getLibraryDir()
      }

      return Environment(
        nativeCreate(
          mutableOptions.keys.map { it.value }.toIntArray(),
          mutableOptions.values.toTypedArray(),
        )
      )
    }

    @JvmStatic private external fun nativeCreate(keys: IntArray, values: Array<String>): Long

    @JvmStatic private external fun nativeGetAvailableAccelerators(handle: Long): IntArray

    @JvmStatic private external fun nativeDestroy(handle: Long)
  }
}
