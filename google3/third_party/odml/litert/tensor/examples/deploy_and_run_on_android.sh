#!/bin/bash

# Script to deploy and run examples on an Android device via ADB

# --- Default values ---
USE_GPU=false # Default to CPU

# --- Usage ---
usage() {
    echo "Usage: $0 [--use_gpu] <binary_name> <binary_build_path>"
    echo "  --use_gpu : Whether to use GPU for execution. Defaults to false."
    echo "  <binary_name> : The name of the binary to run (e.g., run_mha_cc, run_gemma_gqa)."
    echo "  <binary_build_path> : The path to the binary build directory (e.g., bazel-bin/)."
    exit 1
}

# --- Argument Parsing ---
if [ "$#" -eq 0 ]; then
    echo "Error: No arguments provided."
    usage
fi

TEMP=$(getopt -o '' --long use_gpu -- "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing options." >&2
    usage
fi

eval set -- "$TEMP"
unset TEMP

while true; do
    case "$1" in
        '--use_gpu')
            USE_GPU=true
            shift
            ;;
        '--')
            shift
            break
            ;;
        *) 
            break
            ;;
    esac
done

echo "Use GPU: $USE_GPU"

if [ "$#" -ne 2 ]; then
    echo "Error: Incorrect number of arguments or invalid option."
    usage
fi

BINARY_NAME="$1"
BINARY_BUILD_PATH="$2"

if [ ! -d "$BINARY_BUILD_PATH" ]; then
    echo "Error: The provided binary_build_path ($BINARY_BUILD_PATH) is not a valid directory."
    exit 1
fi

# --- Configuration ---
ROOT_DIR="third_party/odml/litert/tensor/examples"
PACKAGE_LOCATION="${ROOT_DIR}"
C_LIBRARY_LOCATION="${BINARY_BUILD_PATH}/third_party/odml/litert/litert/c"

case "$BINARY_NAME" in
    "run_mha_cc")
        MODEL_NAME="mha"
        ;;
    "run_gemma_gqa")
        MODEL_NAME="gemma_attention"
        ;;
    *)
        echo "Error: Invalid binary name provided."
        usage
        ;;
esac

PACKAGE_NAME="${BINARY_NAME}"
OUTPUT_PATH="${BINARY_BUILD_PATH}/${PACKAGE_LOCATION}/${PACKAGE_NAME}"
MODEL_PATH="/tmp/${MODEL_NAME}.tflite"

# Device paths
DEVICE_BASE_DIR="/data/local/tmp/${MODEL_NAME}_android"
DEVICE_EXEC_NAME="${BINARY_NAME}_executable"
DEVICE_MODEL_PATH="${DEVICE_BASE_DIR}/${MODEL_NAME}.tflite"

# Host paths
HOST_GPU_LIBRARY_DIR="${BINARY_BUILD_PATH}/${PACKAGE_LOCATION}/${PACKAGE_NAME}.runfiles/google3/third_party/odml/litert/litert/runtime/accelerators/gpu"

LD_LIBRARY_PATH="${DEVICE_BASE_DIR}/"

# --- Script Logic ---
echo "Starting deployment to Android device..."

# Determine executable path
HOST_EXEC_PATH="${OUTPUT_PATH}"
echo "Using output path: ${HOST_EXEC_PATH}"

if [ ! -f "${HOST_EXEC_PATH}" ]; then
    echo "Error: Executable not found at ${HOST_EXEC_PATH}"
    echo "Please ensure the project has been built and the correct path is provided."
    exit 1
fi

echo "Target device directory: ${DEVICE_BASE_DIR}"

# Create directories on device
adb shell "mkdir -p ${DEVICE_BASE_DIR}"
echo "Created directories on device."

# Push executable
adb push "${HOST_EXEC_PATH}" "${DEVICE_BASE_DIR}/${DEVICE_EXEC_NAME}"
echo "Pushed executable."

# Push model
adb push "${MODEL_PATH}" "${DEVICE_MODEL_PATH}"
echo "Pushed model."

# Push c api shared library
adb push "${C_LIBRARY_LOCATION}/libLiteRtRuntimeCApi.so" "${DEVICE_BASE_DIR}/"
echo "Pushed c api shared library."

# Push gpu accelerator shared library
if [ "$USE_GPU" = true ]; then
    adb push "${HOST_GPU_LIBRARY_DIR}/libLiteRtGpuAccelerator.so" "${DEVICE_BASE_DIR}/"
    echo "Pushed gpu accelerator shared library."
fi

# Set execute permissions
adb shell "chmod +x ${DEVICE_BASE_DIR}/${DEVICE_EXEC_NAME}"
echo "Set execute permissions on device."

echo ""
echo "Deployment complete."
echo "To run the example on the device, use a command like this:"

RUN_COMMAND="./${DEVICE_EXEC_NAME} ${DEVICE_MODEL_PATH}"
if [ "$USE_GPU" = true ]; then
    RUN_COMMAND="${RUN_COMMAND} use_gpu"
fi

FULL_COMMAND="cd ${DEVICE_BASE_DIR} && LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\" ${RUN_COMMAND}"

echo "  adb shell \"${FULL_COMMAND}\""
adb shell "${FULL_COMMAND}"

