#!/bin/bash
# Build and run the LiteRT test app

# Build the application
echo "Building LiteRTClassify app with test activity..."
bazel build //litert/kotlin/sample_apps/LiteRTClassify:LiteRTClassify

# Check if build was successful
if [ $? -ne 0 ]; then
  echo "Build failed!"
  exit 1
fi

# Install the app on the device
echo "Installing app on device..."
adb install -r bazel-bin/litert/kotlin/sample_apps/LiteRTClassify/LiteRTClassify.apk

# Check if installation was successful
if [ $? -ne 0 ]; then
  echo "Installation failed!"
  exit 1
fi

echo "App installed successfully!"
echo ""
echo "To run the main app:"
echo "adb shell am start -n org.tensorflow.lite.examples.classification/.MainActivity"
echo ""
echo "To run the LiteRT GPU test activity directly:"
echo "adb shell am start -n org.tensorflow.lite.examples.classification/org.tensorflow.lite.examples.classification.test.LiteRtTestActivity"
echo ""
echo "To view logs:"
echo "adb logcat -s LiteRTTestApp"
