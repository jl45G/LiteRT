#!/bin/bash

set -e

# Build and run the model generator
blaze build //third_party/odml/litert/tensor/examples:multi_head_attention
blaze-bin/third_party/odml/litert/tensor/examples/multi_head_attention --output_path=/tmp/mha.tflite

# Build the C++ runner
blaze build //third_party/odml/litert/tensor/examples:run_mha_cc

# Run on CPU
blaze-bin/third_party/odml/litert/tensor/examples/run_mha_cc --model_path=/tmp/mha.tflite

# Run on GPU
blaze-bin/third_party/odml/litert/tensor/examples/run_mha_cc --model_path=/tmp/mha.tflite --use_gpu=true

echo "Test passed."