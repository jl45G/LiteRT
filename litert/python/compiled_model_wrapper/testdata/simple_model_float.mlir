// This model takes two float32 input tensors of shape [4], adds them, and returns
// the result. That aligns with your test scenario that expects, e.g. arg0=[1,2,3,4] + arg1=[10,20,30,40] => [11,22,33,44].
module {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    // The TFLite add operation in the TFL dialect:
    %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<4xf32>
    // Return the result
    func.return %0 : tensor<4xf32>
  }
}


