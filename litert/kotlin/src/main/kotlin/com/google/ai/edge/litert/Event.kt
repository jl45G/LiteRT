package com.google.ai.edge.litert

/**
 * Event represents a synchronization primitive that can be used to coordinate execution
 * between different components, such as between the CPU and accelerators.
 */
class Event private constructor(handle: Long) : JniHandle(handle) {

  /**
   * Available event types supported by LiteRT.
   */
  enum class Type(val value: Int) {
    NONE(0),
    SYNC_FENCE_FD(1),
    OPEN_CL(2)
  }

  /**
   * Signals the event, which will unblock any waiting operations.
   */
  fun signal() {
    assertNotDestroyed()
    nativeSignal(handle)
  }

  protected override fun destroy() {
    nativeDestroy(handle)
  }

  companion object {
    init {
      System.loadLibrary("litert_jni")
    }

    /**
     * Creates a new managed event of the specified type.
     *
     * @param type The type of event to create.
     * @return A new managed event.
     */
    @JvmStatic
    fun createManaged(type: Type): Event {
      return Event(nativeCreateManaged(type.value))
    }

    /**
     * Creates an event from an Android sync fence file descriptor.
     *
     * @param fd The file descriptor for the sync fence.
     * @param ownsFd Whether the event takes ownership of the file descriptor.
     * @return A new event wrapping the sync fence.
     */
    @JvmStatic
    fun createFromSyncFenceFd(fd: Int, ownsFd: Boolean): Event {
      return Event(nativeCreateFromSyncFenceFd(fd, ownsFd))
    }

    @JvmStatic private external fun nativeCreateManaged(type: Int): Long
    @JvmStatic private external fun nativeCreateFromSyncFenceFd(fd: Int, ownsFd: Boolean): Long
    @JvmStatic private external fun nativeSignal(handle: Long)
    @JvmStatic private external fun nativeDestroy(handle: Long)
  }
}