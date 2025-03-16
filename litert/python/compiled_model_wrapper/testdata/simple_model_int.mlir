// simple_model_float.mlir
// A minimal TFL model that takes two f32 inputs (each shape [4]) and performs an add.

// Example usage in your test code:
//  - input0: [1.0, 2.0, 3.0, 4.0]
//  - input1: [10.0, 20.0, 30.0, 40.0]
//  => output: [11.0, 22.0, 33.0, 44.0]


module {
  func.func @main(%arg0: tensor<4xi32>) -> tensor<4xi32> {
    %cst = arith.constant dense<1> : tensor<4xi32>
    %0 = tfl.add %arg0, %cst {fused_activation_function = "NONE"} : tensor<4xi32>
    return %0 : tensor<4xi32>
  }
}
