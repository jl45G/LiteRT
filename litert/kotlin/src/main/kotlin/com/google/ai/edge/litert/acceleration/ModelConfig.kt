package com.google.ai.edge.litert.acceleration

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.Environment

/** A model provider that provides a model file and relevant information. */
interface ModelProvider {

  /** Model files could be either an asset or a file. */
  enum class Type {
    ASSET,
    FILE,
  }

  fun getType(): Type

  /** Returns true if the model is ready to be used. */
  fun isReady(): Boolean

  /** Returns the path to the model asset or file. */
  fun getPath(): String

  /** Returns the set of accelerators that the model is compatible with. */
  fun getCompatibleAccelerators(): Set<Accelerator>

  /** Downloads the model if it is not available on the device. */
  suspend fun download()

  companion object {
    /** Creates a model provider that represents a model available on the device. */
    @JvmStatic
    fun staticModel(type: Type, path: String, vararg accelerators: Accelerator): ModelProvider {
      return object : ModelProvider {
        override fun getType() = type

        override fun isReady() = true

        override fun getPath() = path

        // TODO(niuchl): TBD for the default accelerator(?).
        override fun getCompatibleAccelerators() =
          if (accelerators.isEmpty()) setOf(Accelerator.CPU) else setOf(*accelerators)

        override suspend fun download() {}
      }
    }
  }
}

/**
 * ModelSelector allows to dynamically select a ModelProvider from a given set based on the
 * available accelerators, in the order of NPU, GPU, CPU.
 */
class ModelSelector constructor(private vararg val modelProviders: ModelProvider) {

  @Throws(IllegalStateException::class)
  fun selectModel(env: Environment): ModelProvider {
    // TODO(niuchl): Implement dynamic model download.

    val accelerators = env.getAvailableAccelerators()

    val npuModelProvider =
      modelProviders.firstOrNull { it.getCompatibleAccelerators().contains(Accelerator.NPU) }
    if (npuModelProvider?.isReady() == true && accelerators.contains(Accelerator.NPU)) {
      return ensureModelFileAvailable(npuModelProvider)
    }

    val gpuModelProvider =
      modelProviders.firstOrNull { it.getCompatibleAccelerators().contains(Accelerator.GPU) }
    if (gpuModelProvider?.isReady() == true && accelerators.contains(Accelerator.GPU)) {
      return ensureModelFileAvailable(gpuModelProvider)
    }

    val cpuModelProvider =
      modelProviders.firstOrNull { it.getCompatibleAccelerators().contains(Accelerator.CPU) }
    if (cpuModelProvider?.isReady() == true && accelerators.contains(Accelerator.CPU)) {
      return ensureModelFileAvailable(cpuModelProvider)
    }

    throw IllegalStateException("No model is available.")
  }

  @Throws(IllegalStateException::class)
  private fun ensureModelFileAvailable(modelProvider: ModelProvider): ModelProvider {
    if (!modelProvider.isReady()) {
      throw IllegalStateException("Model is not ready to be used yet.")
    }
    return modelProvider
  }
}
