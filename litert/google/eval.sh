#!/bin/bash

source gbash.sh

DEFINE_string host_dir "/tmp/torch_models" "host storage directory to store all torch models"
DEFINE_bool download_models false "whether to download the models"
DEFINE_bool compile_models false "whether to compile the models"
DEFINE_bool eval_models false "whether to eval the models"
DEFINE_bool benchmark_models false "whether to benchmark the models via npu"
DEFINE_bool benchmark_models_gpu false "whether to benchmark the models via gpu"
DEFINE_string soc_man "Qualcomm" " SoC manufacturer [Qualcomm, MediaTek]"
DEFINE_string soc_model "SM8650" " SoC model"
DEFINE_string eval_result "/tmp/eval_result.csv" "Eval result file name, new results will be appended to the file"
DEFINE_string benchmark_result "/tmp/benchmark_result.csv" "Benchmark result file name, new results will be appended to the file"

MODEL_SET="pytorch_70"
CNS_MODEL_DIR="/cns/md-d/home/mediapipe/odml_models/"${MODEL_SET}

APPLY_MODEL_MAIN_BUILD_TARGET="//third_party/odml/litert/litert/tools:apply_plugin_main"

EVAL_MODEL_MAIN="qualcomm_dispatcher_numeric_test"
EVAL_MODEL_MAIN_BUILD_TARGET="//third_party/odml/litert/litert/google:${EVAL_MODEL_MAIN}"
EVAL_MODE_MAIN_BINARY_NAME="blaze-bin/third_party/odml/litert/litert/google/${EVAL_MODEL_MAIN}"
EVAL_MODEL_BUILD_OPTIONS="--config=android_arm64 --copt=-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1 -c opt"

BENCHMARK_MODEL_MAIN="run_model"
BENCHMARK_MODEL_MAIN_BINARY_NAME="blaze-bin/third_party/odml/litert/litert/tools/${BENCHMARK_MODEL_MAIN}"
BENCHMARK_MODEL_MAIN_BUILD_TARGET="//third_party/odml/litert/litert/tools:${BENCHMARK_MODEL_MAIN}"

DEVICE_PATH="/data/local/tmp/npu_eval"
ADSP_LIBRARY_PATH_ARG=
QNN_SDK_ROOT=third_party/qairt/latest


function download_models() {
  fileutil -parallelism=8 cp -R ${CNS_MODEL_DIR} ${FLAGS_host_dir}
}

function compile_models() {
  NPU_MODEL_DIR=${CPU_MODEL_DIR}_${FLAGS_soc_man}
  mkdir -p ${NPU_MODEL_DIR}
  for item in "$CPU_MODEL_DIR"/*; do
    if [ -f "$item" ]; then
      model_file_name=$(basename "$item")
      model_name=${model_file_name%.*}
      npu_model_file_name=${model_name}_${FLAGS_soc_man}_${FLAGS_soc_model}.tflite
      npu_model_file_path=${NPU_MODEL_DIR}/${npu_model_file_name}
      blaze run -c opt ${APPLY_MODEL_MAIN_BUILD_TARGET} -- \
        --model=${item} \
        --o=${npu_model_file_path} \
        --soc_manufacturer=${FLAGS_soc_man} \
        --soc_model=${FLAGS_soc_model} \
        --cmd=apply
    fi
  done
}

function build_and_push_runtime_c_api() {
  blaze build //third_party/odml/litert/litert/c:litert_runtime_c_api_so ${EVAL_MODEL_BUILD_OPTIONS}
  adb push --sync blaze-bin/third_party/odml/litert/litert/c/libLiteRtRuntimeCApi.so ${DEVICE_PATH}
}

function build_and_push_mediatek_libs() {
  # Build the MediaTek dispatch API and push it to the device.
  blaze build -c opt --config=android_arm64 --android_ndk_min_sdk_version=26 third_party/odml/litert/litert/vendors/mediatek/dispatch:dispatch_api_so
  adb push --sync blaze-bin/third_party/odml/litert/litert/vendors/mediatek/dispatch/libLiteRtDispatch_Mediatek.so $DEVICE_PATH
  # MediaTek NPU SDK is on device.
}

function build_and_push_qnn_libs() {
  # Build the Qualcomm dispatch API and push it to the device.
  blaze build -c opt --config=android_arm64 third_party/odml/litert/litert/vendors/qualcomm/dispatch:dispatch_api_so
  adb push --sync blaze-bin/third_party/odml/litert/litert/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so ${DEVICE_PATH}
  # Push the Qualcomm NPU SDK to the device.
  adb push --sync $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp*Stub.so ${DEVICE_PATH}
  adb push --sync $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so ${DEVICE_PATH}
  adb push --sync $QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so ${DEVICE_PATH}
  adb push --sync $QNN_SDK_ROOT/lib/hexagon-*/unsigned/libQnnHtp*Skel.so ${DEVICE_PATH}
  # TODO(yunandrew): Do we need prepare lib?
  adb push --sync $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so ${DEVICE_PATH}
}

function evaluate_models() {
  NPU_MODEL_DIR=${CPU_MODEL_DIR}_${FLAGS_soc_man}
  # build the eval model binary
  blaze build ${EVAL_MODEL_MAIN_BUILD_TARGET} ${EVAL_MODEL_BUILD_OPTIONS}
  adb push --sync ${EVAL_MODE_MAIN_BINARY_NAME} ${DEVICE_PATH}
  time_now=$(date +%F_%T)

  # check if result file exists
  if [ -f "${FLAGS_eval_result}" ]; then
    echo "Eval result file already exists, will append new results to the file"
  else
    echo "Eval result file does not exist, will create a new file"
    touch "${FLAGS_eval_result}"
    echo time,model_name,soc_man,soc_model,mse >> "${FLAGS_eval_result}"
  fi


  for item in "$CPU_MODEL_DIR"/*; do
    if [ -f "$item" ]; then
      model_file_name=$(basename "$item")
      model_name=${model_file_name%.*}
      npu_model_file_name=${model_name}_${FLAGS_soc_man}_${FLAGS_soc_model}.tflite
      npu_model_file_path=${NPU_MODEL_DIR}/${npu_model_file_name}
    fi
      adb push --sync $item ${DEVICE_PATH}
      adb push --sync $npu_model_file_path ${DEVICE_PATH}
      mse=$(adb shell "${ADSP_LIBRARY_PATH_ARG} \
        LD_LIBRARY_PATH=${DEVICE_PATH} ${DEVICE_PATH}/${EVAL_MODEL_MAIN} \
        --cpu_model=${DEVICE_PATH}/${model_file_name} --npu_model=${DEVICE_PATH}/${npu_model_file_name}" | grep "MSE")
      mse=$(echo "${mse}" | tr -d "MSE: ")
      mse=$(echo "${mse}" | tr "\n" ",")
      echo "$time_now,${model_name},${FLAGS_soc_man},${FLAGS_soc_model},${mse}" >> "${FLAGS_eval_result}"
  done
}

function benchmark_models() {
  NPU_MODEL_DIR=${CPU_MODEL_DIR}_${FLAGS_soc_man}
  # build the benchmark model binary
  blaze build ${BENCHMARK_MODEL_MAIN_BUILD_TARGET} ${EVAL_MODEL_BUILD_OPTIONS}
  adb push --sync ${BENCHMARK_MODEL_MAIN_BINARY_NAME} ${DEVICE_PATH}
  time_now=$(date +%F_%T)

  # check if result file exists
  if [ -f "${FLAGS_benchmark_result}" ]; then
    echo "Benchmark result file already exists, will append new results to the file"
  else
    echo "Benchmark result file does not exist, will create a new file"
    touch "${FLAGS_benchmark_result}"
    echo time,model_name,soc_man,soc_model,first_time,avg_time,max_time,min_time,gpu_avg_time >> "${FLAGS_benchmark_result}"
  fi

  for item in "$CPU_MODEL_DIR"/*; do
    if [ -f "$item" ]; then
      model_file_name=$(basename "$item")
      model_name=${model_file_name%.*}
      npu_model_file_name=${model_name}_${FLAGS_soc_man}_${FLAGS_soc_model}.tflite
      npu_model_file_path=${NPU_MODEL_DIR}/${npu_model_file_name}
    fi
      adb push --sync $npu_model_file_path ${DEVICE_PATH}
      adb shell "${ADSP_LIBRARY_PATH_ARG} \
        LD_LIBRARY_PATH=${DEVICE_PATH} ${DEVICE_PATH}/${BENCHMARK_MODEL_MAIN} \
        --graph=${DEVICE_PATH}/${npu_model_file_name} \
        --iterations=20 \
        --signature_index=0 \
        --dispatch_library_dir=${DEVICE_PATH}" 2>/tmp/benchmark_log.txt
      result=$(cat "/tmp/benchmark_log.txt")
      min_time=$(echo "${result}" | grep "Fastest run took")
      min_time=$(echo "$min_time" | sed -n 's/.*took \([0-9]\+\) microseconds.*/\1/p')
      max_time=$(echo "${result}" | grep "Slowest run took")
      max_time=$(echo "$max_time" | sed -n 's/.*took \([0-9]\+\) microseconds.*/\1/p')
      first_time=$(echo "${result}" | grep "First run took")
      first_time=$(echo "$first_time" | sed -n 's/.*took \([0-9]\+\) microseconds.*/\1/p')
      avg_time=$(echo "${result}" | grep "All runs took average")
      avg_time=$(echo "$avg_time" | sed -n 's/.*took average \([0-9]\+\) microseconds.*/\1/p')

      if (( ${FLAGS_benchmark_models_gpu})) ; then
      echo "Benchmarking models via GPU"
        adb push --sync ${item} ${DEVICE_PATH}
        adb shell "${DEVICE_PATH}/${BENCHMARK_MODEL_MAIN} \
          --graph=${DEVICE_PATH}/${model_file_name} \
          --iterations=20 \
          --use_gpu=true \
          --signature_index=0" 2>/tmp/benchmark_log.txt
        result_gpu=$(cat "/tmp/benchmark_log.txt")
        avg_time_gpu=$(echo "${result_gpu}" | grep "All runs took average")
        avg_time_gpu=$(echo "$avg_time_gpu" | sed -n 's/.*took average \([0-9]\+\) microseconds.*/\1/p')
      fi
      echo "${time_now},${model_name},${FLAGS_soc_man},${FLAGS_soc_model},${first_time},${avg_time},${max_time},${min_time},${avg_time_gpu}" >> "${FLAGS_benchmark_result}"
  done
}

function main() {
  set -x
  # Configure script
  CPU_MODEL_DIR=${FLAGS_host_dir}/${MODEL_SET}

  # Build and push the runtime and dispatch libs.
  adb shell "mkdir -p ${DEVICE_PATH}"
  if [[ ${FLAGS_soc_man} == "MediaTek" ]]; then
    echo "Building and pushing MediaTek libs"
    EVAL_MODEL_BUILD_OPTIONS=${EVAL_MODEL_BUILD_OPTIONS}" --android_ndk_min_sdk_version=26"
    build_and_push_mediatek_libs
  elif [[ ${FLAGS_soc_man} == "Qualcomm" ]]; then
    ADSP_LIBRARY_PATH_ARG="ADSP_LIBRARY_PATH=${DEVICE_PATH}"
    echo "Building and pushing Qualcomm libs"
    build_and_push_qnn_libs
  fi
  build_and_push_runtime_c_api

  #-----------------------------  Download models ---------------------------
  if (( ${FLAGS_download_models} )); then
    echo "Downloading models to ${FLAGS_host_dir}"
    mkdir -p ${FLAGS_host_dir}
    download_models
  else
    echo "Skipping model download"
  fi

  #----------------------------- Compile models -----------------------------
  if (( ${FLAGS_compile_models} )); then
    echo "Compiling models to ${FLAGS_host_dir}"
    compile_models
  else
    echo "Skipping model compilation"
  fi

  #------------------------ Compare numeric with CPU -------------------------
  if (( ${FLAGS_eval_models} )); then
    echo "Evaluating models to ${FLAGS_host_dir}"
    evaluate_models
  else
    echo "Skipping model evaluation"
  fi

  #----------------------------- Benchmark ------------------------------------
  if (( ${FLAGS_benchmark_models} )); then
    echo "Benchmarking models to ${FLAGS_host_dir}"
    benchmark_models
  else
    echo "Skipping model benchmarking"
  fi

}

gbash::main "$@"
