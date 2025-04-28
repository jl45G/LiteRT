#!/bin/bash

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running"
    exit 1
fi

echo "Building Docker image..."
docker build -t litert_build_env -f ./hermetic_build.Dockerfile .

if [ $? -ne 0 ]; then
  echo "Error: Docker build failed."
  exit 1
fi

echo "Running build in Docker container..."
docker run --rm -v $(pwd):/litert_build litert_build_env

if [ $? -ne 0 ]; then
  echo "Error: Build failed inside Docker container."
  exit 1
fi

echo "Build completed successfully!"