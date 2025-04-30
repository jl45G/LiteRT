# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Image Segmentation using LiteRT.

This script demonstrates how to use LiteRT for image segmentation.
It loads an image, processes it with a segmentation model, and saves
the resulting segmentation mask and blended output image.
"""

import os
import random
import time
import logging
import dataclasses
from typing import Any, Dict, List, Tuple

from absl import flags
import numpy as np
from PIL import Image

from litert.python.litert_wrapper.compiled_model_wrapper.compiled_model import CompiledModel
from google3.third_party.tensorflow.python.platform import resource_loader  # pylint: disable=g-direct-tensorflow-import

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define labels as a constant
_LABELS = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv",
    "------",
]

# Define golden ratio conjugate as a constant
_GOLDEN_RATIO_CONJUGATE = 0.618033988749895

# Define flags
_MODEL_PATH = flags.DEFINE_string(
    "model_path",
    "testdata/selfie_multiclass_256x256.tflite",
    "Path to the TFLite model file for segmentation.",
)
_IMAGE_PATH = flags.DEFINE_string(
    "image_path", "testdata/image.jpg", "Path to the input image."
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", "./output", "Directory to save the output."
)

@dataclasses.dataclass
class ColoredLabel:
  """Represents a colored label for segmentation visualization.
  
  Attributes:
      label: The name of the label.
      display_name: The display name of the label.
      color: The RGB color associated with the label.
  """
  label: str
  display_name: str
  color: Tuple[int, int, int]


class ImageSegmentation:
  """Helper class for image segmentation tasks using LiteRT.

  Provides methods to initialize a segmentation model, preprocess images,
  run inference, and process the segmentation masks.
  """

  def __init__(self, model_path: str):
    """Initialize the segmentation helper.

    Args:
        model_path: Path to the TFLite model file for segmentation
    """
    self.model_path = model_path
    self.model = None
    self.colored_labels = self._create_colored_labels()
    self.input_size = (256, 256)
    self._initialize()

  def _initialize(self) -> None:
    """Initialize the CompiledModel for segmentation."""
    try:
      self.model = CompiledModel.from_file(self.model_path)
      logging.info(
          "Model loaded successfully with"
          f" {self.model.get_num_signatures()} signatures"
      )
    except Exception as e:
      logging.error(f"Failed to create LiteRT model: {str(e)}")
      raise

  def normalize(
      self, image: np.ndarray, mean: float = 127.5, stddev: float = 127.5
  ) -> np.ndarray:
    """Normalize the input image for the model.

    Args:
        image: Input image as numpy array (H, W, 3)
        mean: Mean value for normalization
        stddev: Standard deviation for normalization

    Returns:
        Normalized image as a numpy array
    """
    # Ensure image is in the right format (float32)
    image = image.astype(np.float32)

    # Normalize the image
    normalized = (image - mean) / stddev

    # Ensure the result is contiguous in memory and in float32 format
    return np.ascontiguousarray(normalized, dtype=np.float32)

  def preprocess_image(
      self, image_path: str, rotation_degrees: int = 0
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess an image for segmentation.

    Args:
        image_path: Path to the input image
        rotation_degrees: Optional rotation in degrees

    Returns:
        Tuple of (original image, preprocessed image)
    """
    image = Image.open(image_path)

    # Resize to expected input dimensions
    width, height = self.input_size
    resized_image = image.resize((width, height))

    # Apply rotation if needed
    if rotation_degrees != 0:
      resized_image = resized_image.rotate(-rotation_degrees)

    # Convert to numpy array
    img_array = np.array(resized_image)

    # Normalize the image
    normalized = self.normalize(img_array)

    return np.array(image), normalized

  def segment(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
    """Perform segmentation on a preprocessed image.

    Args:
        image: Preprocessed image as numpy array

    Returns:
        Tuple of (segmentation mask, inference time in ms)

    Raises:
        ValueError: If the model creation failed
    """
    if self.model is None:
      raise ValueError("Model not compiled.")

    # Track inference time
    start_time = time.time()

    # Create input and output buffers
    sig_idx = 0
    input_buffers = self.model.create_input_buffers(sig_idx)
    output_buffers = self.model.create_output_buffers(sig_idx)

    # Write the preprocessed image to the input buffer
    # Reshape to match the expected input format
    input_data = image.reshape(-1)
    input_buffers[0].write(input_data)

    # Run inference
    self.model.run_by_index(sig_idx, input_buffers, output_buffers)

    # Get output data (segmentation logits)
    # Assuming the output has shape [height, width, num_classes]
    height, width = self.input_size
    num_classes = 6

    # Read output data
    output_size = height * width * num_classes
    output_data = output_buffers[0].read(output_size, np.float32)
    output_data = output_data.reshape(height, width, num_classes)

    # Process output to get segmentation mask
    mask = self._process_output(output_data)

    # Calculate inference time
    inference_time = (
        time.time() - start_time
    ) * 1000  # Convert to milliseconds

    # Clean up
    for buf in input_buffers:
      buf.destroy()
    for buf in output_buffers:
      buf.destroy()

    return mask, inference_time

  def _process_output(self, output_data: np.ndarray) -> np.ndarray:
    """Process model output to create a segmentation mask.

    Args:
        output_data: Model output as numpy array [height, width, num_classes]

    Returns:
        Segmentation mask as numpy array
    """
    # Find the class with the highest probability for each pixel
    return np.argmax(output_data, axis=2).astype(np.uint8)

  def _create_colored_labels(self) -> List[ColoredLabel]:
    """Create colored labels for visualization.

    Returns:
        List of ColoredLabel objects containing label information and colors
    """
    # Create a list of colored labels
    colored_labels = []

    # Generate visually distinct colors using golden ratio
    hue = random.random()

    for idx, label in enumerate(_LABELS):
      if idx == 0:
        # Background is black
        color = (0, 0, 0)
      else:
        # Generate colors using golden ratio method
        hue += _GOLDEN_RATIO_CONJUGATE
        hue %= 1.0

        # Convert HSV to RGB
        h = hue * 360
        s = 0.7
        v = 0.8

        # HSV to RGB conversion
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c

        if h < 60:
          r, g, b = c, x, 0
        elif h < 120:
          r, g, b = x, c, 0
        elif h < 180:
          r, g, b = 0, c, x
        elif h < 240:
          r, g, b = 0, x, c
        elif h < 300:
          r, g, b = x, 0, c
        else:
          r, g, b = c, 0, x

        r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
        color = (r, g, b)

      colored_labels.append(ColoredLabel(label, label, color))

    return colored_labels

  def create_colored_mask(self, mask: np.ndarray) -> Tuple[np.ndarray, List[ColoredLabel]]:
    """Create a colored segmentation mask for visualization.

    Args:
        mask: Segmentation mask as numpy array

    Returns:
        Tuple containing:
        - Colored mask as numpy array
        - List of ColoredLabel objects
    """
    height, width = mask.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Apply colors to each class
    for label_idx, label_info in enumerate(self.colored_labels):
      # Create a mask for this class
      class_mask = mask == label_idx

      # Apply color to this class
      color = label_info.color
      colored_mask[class_mask] = color

    return colored_mask, self.colored_labels

  def blend_images(
      self, image: np.ndarray, colored_mask: np.ndarray, alpha: float = 0.5
  ) -> np.ndarray:
    """Blend the original image with the colored segmentation mask.

    Args:
        image: Original image as numpy array
        colored_mask: Colored segmentation mask as numpy array
        alpha: Blending factor (0.0 to 1.0)

    Returns:
        Blended image as numpy array
    """
    # Resize colored mask to match original image if necessary
    if image.shape[:2] != colored_mask.shape[:2]:
      colored_mask_pil = Image.fromarray(colored_mask)
      colored_mask_pil = colored_mask_pil.resize(
          (image.shape[1], image.shape[0])
      )
      colored_mask = np.array(colored_mask_pil)

    # Blend images
    blended = (image * (1 - alpha) + colored_mask * alpha).astype(np.uint8)

    return blended


def main():
  """Main function to run the image segmentation demo."""

  # Get the paths from the flags
  model_path = _MODEL_PATH.value
  image_path = _IMAGE_PATH.value
  output_dir = _OUTPUT_DIR.value

  # Create output directory if it doesn't exist
  os.makedirs(output_dir, exist_ok=True)

  # Load model and image files
  model_path = resource_loader.get_path_to_datafile(model_path)
  image_path = resource_loader.get_path_to_datafile(image_path)

  logging.info(f"Using model: {model_path}")
  logging.info(f"Using image: {image_path}")

  # Initialize the segmentation helper
  segmentation = ImageSegmentation(model_path)

  # Preprocess the image
  original_image, preprocessed_image = segmentation.preprocess_image(image_path)

  # Run segmentation
  mask, inference_time = segmentation.segment(preprocessed_image)

  logging.info(f"Segmentation finished in {inference_time:.2f} ms")

  # Create a colored mask
  colored_mask, label_info = segmentation.create_colored_mask(mask)

  # Blend with original image
  blended_image = segmentation.blend_images(original_image, colored_mask)

  # Save results
  output_base = os.path.splitext(os.path.basename(image_path))[0]

  # Save mask as image for debugging
  mask_image = Image.fromarray(colored_mask)
  mask_path = os.path.join(output_dir, f"{output_base}_mask.png")
  mask_image.save(mask_path)

  # Save blended image
  blended_image_pil = Image.fromarray(blended_image)
  blended_path = os.path.join(output_dir, f"{output_base}_blended.png")
  blended_image_pil.save(blended_path)

  logging.info(f"Results saved to: {output_dir}")
  logging.info(f"Mask: {mask_path}")
  logging.info(f"Blended image: {blended_path}")


if __name__ == "__main__":
  main()
