package org.tensorflow.tflite.experimental.litert.sample;

import android.app.Activity;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.widget.TextView;
import androidx.annotation.Nullable;
import com.google.ai.edge.litert.Accelerator;
import com.google.ai.edge.litert.CompiledModel;
import com.google.ai.edge.litert.Environment;
import com.google.ai.edge.litert.TensorBuffer;
import com.google.ai.edge.litert.acceleration.BuiltinNpuAcceleratorProvider;
import com.google.ai.edge.litert.acceleration.NpuAcceleratorProvider;
import java.util.Arrays;
import java.util.List;

/** Main activity for the test app. */
public class MainActivity extends Activity {

  private static final String TAG = "MainActivity";

  private static final List<float[]> testInputTensor =
      Arrays.asList(new float[] {1, 2}, new float[] {10, 20});

  private TextView logView;

  @Override
  protected void onCreate(@Nullable Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    logView = findViewById(R.id.log_text);
    logView.setMovementMethod(new ScrollingMovementMethod());

    final Bundle bundle = getIntent().getExtras();
    final boolean useNpuAccelerator =
        bundle != null && bundle.getBoolean("use_npu_accelerator", false);

    findViewById(R.id.run_scenario_btn).setOnClickListener(v -> runScenario(useNpuAccelerator));
    runScenario(useNpuAccelerator);
  }

  /** Runs tests with all available delegates. */
  private void runScenario(boolean useNpuAccelerator) {
    logView.setText("Start scenario\n");

    logEvent("Running inference with LiteRt API");

    try (CompiledModel compiledModel = useNpuAccelerator ? simpleNpuModel() : simpleCpuGpuModel()) {
      List<TensorBuffer> inputBuffers = compiledModel.createInputBuffers();
      logEvent("Input buffers size: " + inputBuffers.size());
      for (int i = 0; i < inputBuffers.size(); ++i) {
        inputBuffers.get(i).writeFloat(testInputTensor.get(i));
        logEvent("Input[" + i + "]: " + Arrays.toString(testInputTensor.get(i)));
      }

      List<TensorBuffer> outputBuffers = compiledModel.run(inputBuffers);
      logEvent("Output buffers size: " + outputBuffers.size());
      for (int i = 0; i < outputBuffers.size(); ++i) {
        float[] output = outputBuffers.get(i).readFloat();
        logEvent("Output[" + i + "]: " + Arrays.toString(output));
      }

      for (TensorBuffer buffer : inputBuffers) {
        buffer.close();
      }
      for (TensorBuffer buffer : outputBuffers) {
        buffer.close();
      }
    }
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

  private void logEvent(String message, @Nullable Throwable throwable) {
    Log.e(TAG, message, throwable);
    logView.append("• ");
    logView.append(String.valueOf(message)); // valueOf converts null to "null"
    logView.append("\n");
    if (throwable != null) {
      logView.append(throwable.getClass().getCanonicalName() + ": " + throwable.getMessage());
      logView.append("\n");
      logView.append(Arrays.toString(throwable.getStackTrace()));
      logView.append("\n");
    }
  }
}
