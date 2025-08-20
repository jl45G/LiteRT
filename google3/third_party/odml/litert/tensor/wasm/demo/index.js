/**
 * Computes the factorial of a non-negative integer using LiteRT tensors.
 * @param {number} n The non-negative integer.
 * @return {!Module.Tensor} A LiteRT tensor containing the factorial result.
 */
function factorial(n) {
  if (n === 0 || n === 1) {
    return Module.createTensor({
      name: 'result',
      type: Module.Type.kFP32,
      shape: [1],
      buffer: new Float32Array([1.0])
    });
  }

  let result = Module.createTensor({
    name: 'result',
    type: Module.Type.kFP32,
    shape: [1],
    buffer: new Float32Array([1.0])
  });

  for (let i = 2; i <= n; i++) {
    const tensor_i = Module.createTensor({
      name: `t_${i}`,
      type: Module.Type.kFP32,
      shape: [1],
      buffer: new Float32Array([i])
    });
    result = Module.multiply(result, tensor_i);
  }

  return result;
}

const input_val = 5;
const fact_5 = factorial(input_val);

// To run the graph, we need to create a vector of output tensors.
const outputs = new Module.VectorTensor();
outputs.push_back(fact_5);

// Now, run the computation.
const status = Module.run(outputs);

if (status) {
  console.error(`Error: ${status}`);
} else {
  const result = fact_5.getBuffer();
  for (const value of result) {
    console.log(value);
  }
  const outputElement = document.getElementById('output');
  if (outputElement) {
    outputElement.textContent = `Factorial of ${input_val} is ${result[0]}`;
  }
}

// Clean up the vector
outputs.delete();