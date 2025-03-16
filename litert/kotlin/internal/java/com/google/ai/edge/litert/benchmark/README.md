# LiteRT Kotlin API Benchmark Activity

## Build and Install

```sh
blaze build -c opt --android_platforms=//buildenv/platforms/android:arm64-v8a \
  litert/kotlin/internal/java/com/google/ai/edge/litert/benchmark \
  && adb install blaze-bin/litert/kotlin/internal/java/com/google/ai/edge/litert/benchmark/benchmark.apk
```

## Example

```sh
# Parameters
MODEL_LOCAL_PATH=speech/tts/engine/tensorflow/testdata/simple_double_add.tflite
MODEL_REMOTE_PATH=/data/local/tmp/model.tflite
NUM_ITERATIONS=100

# Prepare the model
adb push ${MODEL_LOCAL_PATH} ${MODEL_REMOTE_PATH}

# Launch the Benchmark activity
adb shell am start -n com.google.ai.edge.litert.benchmark/.BenchmarkActivity \
  --es "model_path" "${MODEL_REMOTE_PATH}" \
  --ei "num_iterations" "${NUM_ITERATIONS}"
```
