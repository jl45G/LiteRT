package org.tensorflow.tflite.experimental.litert.sample;

import android.app.Activity;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;
import androidx.annotation.Nullable;
import com.google.ai.edge.litert.Accelerator;
import com.google.ai.edge.litert.CompiledModel;
import com.google.ai.edge.litert.Environment;
import com.google.ai.edge.litert.TensorBuffer;
import com.google.ai.edge.litert.acceleration.BuiltinNpuAcceleratorProvider;
import com.google.ai.edge.litert.acceleration.NpuAcceleratorProvider;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;

/** Main activity for the test app. */
public class MainActivity extends Activity {

  private static final String TAG = "MainActivity";

  // Sample input
  private static final List<float[]> testInputTensor =
      Arrays.asList(new float[] {1, 2}, new float[] {10, 20});

  private TextView logView;

  @Override
  protected void onCreate(@Nullable Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    logView = findViewById(R.id.log_text);
    logView.setMovementMethod(new ScrollingMovementMethod());

    //     final Bundle bundle = getIntent().getExtras();
    //     final boolean useNpuAccelerator =
    //         bundle != null && bundle.getBoolean("use_npu_accelerator", false);
    //
    //     findViewById(R.id.run_scenario_btn).setOnClickListener(v ->
    // runScenario(useNpuAccelerator));
    //     runScenario(useNpuAccelerator);
    Button scenarioButton = findViewById(R.id.run_scenario_btn);
    scenarioButton.setOnClickListener(v -> runScenario());

    // Optionally auto-run on start:
    runScenario();
  }

  /** Runs tests with all available delegates. */
  private void runScenario(boolean useNpuAccelerator) {
    logView.setText("Start scenario\n");

    logEvent("Running inference with LiteRt API (sync + zero-copy example)");

    // 1) Load TFLite model.
    //     CompiledModel compiledModel = useNpuAccelerator ? simpleNpuModel() : simpleCpuGpuModel();
    Model model = Model.load(getAssets(), "simple_model.tflite");
    CompiledModel.Options options = new CompiledModel.Options(Accelerator.NONE);
    CompiledModel compiledModel = CompiledModel.create(model, options);

    // 2) Create input buffers the usual way
    List<TensorBuffer> inputBuffers = compiledModel.createInputBuffers();
    logEvent("Input buffers size: " + inputBuffers.size());
    for (int i = 0; i < inputBuffers.size(); ++i) {
      inputBuffers.get(i).writeFloat(testInputTensor.get(i));
      logEvent("Input[" + i + "]: " + Arrays.toString(testInputTensor.get(i)));
    }

    // 3) Run synchronous
    List<TensorBuffer> outputBuffers = compiledModel.run(inputBuffers);
    logEvent("Output buffers size: " + outputBuffers.size());
    for (int i = 0; i < outputBuffers.size(); ++i) {
      float[] output = outputBuffers.get(i).readFloat();
      logEvent("Output[" + i + "]: " + Arrays.toString(output));
    }

    // 4) Demonstrate asynchronous run usage
    //    We'll reuse the same input buffers, just for demonstration:
    boolean wasAsync = compiledModel.runAsync(inputBuffers, outputBuffers);
    logEvent("runAsync => " + (wasAsync ? "Truly Async" : "Fallback to Sync"));

    // 5) Demonstrate zero-copy usage with a direct ByteBuffer
    logEvent("Now trying Zero-Copy creation of a TensorBuffer from a direct ByteBuffer");
    int capacity = 8; // e.g. 2 floats = 8 bytes
    ByteBuffer directBuf = AlignedBufferUtils.create64ByteAlignedByteBuffer(capacity);
    // Put 2 floats => [99.0f, 123.0f]
    directBuf.putFloat(99.0f);
    directBuf.putFloat(123.0f);
    directBuf.rewind();

    // shape => [1,2]
    TensorBuffer zeroCopyBuffer =
        TensorBuffer.createFromDirectBuffer(
            /* elementTypeCode= */ 0, // 0 => Float32 in our mapping
            new int[] {1, 2},
            directBuf);

    // Show that we can read from zeroCopyBuffer with normal readFloat
    float[] readBack = zeroCopyBuffer.readFloat();
    logEvent("Zero-Copy buffer read: " + Arrays.toString(readBack));

    // 6) Demonstrate event usage (though it might not do anything if run was synchronous)
    // Typically events are set by the runtime or accelerator. We'll just check it:
    if (outputBuffers.size() > 0) {
      TensorBuffer out0 = outputBuffers.get(0);
      boolean hasEv = out0.hasEvent();
      logEvent("Output[0] hasEvent => " + hasEv);
      if (hasEv) {
        long evHandle = out0.getEventHandle();
        logEvent("Event handle => " + evHandle + "; waiting...");
        out0.waitOnEvent(-1L); // indefinite
        logEvent("Event wait done; clearing...");
        out0.clearEvent();
      }
    }

    // 7) Cleanup
    for (TensorBuffer buffer : inputBuffers) {
      buffer.destroy();
    }
    for (TensorBuffer buffer : outputBuffers) {
      buffer.destroy();
    }
    zeroCopyBuffer.destroy();
    compiledModel.destroy();
  }

  private CompiledModel simpleCpuGpuModel() {
    return CompiledModel.create(
        getAssets(),
        "simple_model.tflite",
        new CompiledModel.Options(Accelerator.CPU, Accelerator.GPU));
  }

  private CompiledModel simpleNpuModel() {
    NpuAcceleratorProvider npuAcceleratorProvider = new BuiltinNpuAcceleratorProvider(this);
    Environment env = Environment.create(npuAcceleratorProvider);

    return CompiledModel.create(
        getAssets(), "simple_model_npu.tflite", new CompiledModel.Options(Accelerator.NPU), env);
  }

  // TODO(niuchl): Use ModelSelector when the accelerator registry could return NPU.
  /*
  private CompiledModel createCompiledModelWithNpu() {
    NpuAcceleratorProvider npuAcceleratorProvider = new BuiltinNpuAcceleratorProvider(this);
    Environment env = Environment.create(npuAcceleratorProvider);

    ModelProvider cpuGpuModelProvider =
        ModelProvider.staticModel(
            ModelProvider.Type.ASSET, "simple_model.tflite", Accelerator.CPU, Accelerator.GPU);
    ModelProvider npuModelProvider =
        ModelProvider.staticModel(
            ModelProvider.Type.ASSET, "simple_model_npu.tflite", Accelerator.NPU);
    ModelSelector modelSelector = new ModelSelector(cpuGpuModelProvider, npuModelProvider);
    return CompiledModel.create(modelSelector, env, getAssets());
  }
  */

  private void logEvent(String message) {
    logEvent(message, null);
  }

  private void logEvent(String message, Throwable throwable) {
    Log.e(TAG, message, throwable);
    logView.append("• ");
    logView.append(String.valueOf(message));
    logView.append("\n");
    if (throwable != null) {
      logView.append(throwable.getClass().getCanonicalName() + ": " + throwable.getMessage());
      logView.append("\n");
      logView.append(Arrays.toString(throwable.getStackTrace()));
      logView.append("\n");
    }
  }
}
