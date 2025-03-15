package org.tensorflow.tflite.experimental.litert.sample;

import android.annotation.SuppressLint;
import java.lang.reflect.Field;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Helper methods for creating ByteBuffers aligned to a 64-byte boundary. This is required if LiteRT
 * is built with LITERT_HOST_MEMORY_BUFFER_ALIGNMENT = 64.
 *
 * <p>WARNING: Reflection on the "address" field can break on certain Android versions or future
 * devices that block reflection. You may need alternative methods.
 */
public final class AlignedBufferUtils {

  private static final int LITERT_ALIGNMENT = 64; // Must match LITERT_HOST_MEMORY_BUFFER_ALIGNMENT

  private AlignedBufferUtils() {
    // Prevent instantiation
  }

  /**
   * Creates a 64-byte aligned ByteBuffer with a capacity of exactly {@code capacityBytes}. The
   * returned ByteBuffer: - is direct - ordered native - starts at a multiple-of-64 address
   */
  public static ByteBuffer create64ByteAlignedByteBuffer(int capacityBytes) {
    // Over-allocate by up to 63 bytes, so we can shift to the next 64 boundary.
    ByteBuffer big =
        ByteBuffer.allocateDirect(capacityBytes + (LITERT_ALIGNMENT - 1))
            .order(ByteOrder.nativeOrder());

    long baseAddress = getAddress(big);
    // Round up baseAddress to next multiple of 64:
    long alignedAddress = (baseAddress + (LITERT_ALIGNMENT - 1)) & ~((long) (LITERT_ALIGNMENT - 1));
    int shift = (int) (alignedAddress - baseAddress);

    // Position 'big' to that shift, then slice so the new ByteBuffer is at that alignment.
    big.position(shift);
    ByteBuffer aligned = big.slice().order(ByteOrder.nativeOrder());
    aligned.limit(capacityBytes); // Ensure exactly capacityBytes

    // Double-check alignment:
    long checkAddr = getAddress(aligned);
    if ((checkAddr % LITERT_ALIGNMENT) != 0) {
      throw new RuntimeException(
          "Failed to achieve 64-byte alignment (address=0x" + Long.toHexString(checkAddr) + ")");
    }
    return aligned;
  }

  /**
   * Reflection-based helper to retrieve the native address of a direct ByteBuffer. This may fail if
   * hidden APIs or reflection are blocked on your Android platform.
   */
  private static long getAddress(ByteBuffer buffer) {
    try {
      @SuppressLint("DiscouragedPrivateApi")
      Field addressField = Buffer.class.getDeclaredField("address");
      addressField.setAccessible(true);
      return addressField.getLong(buffer);
    } catch (Exception e) {
      throw new RuntimeException("Could not get native address via reflection", e);
    }
  }
}
