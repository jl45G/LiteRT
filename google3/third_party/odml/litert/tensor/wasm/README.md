# LiterT Tensor WASM Demo

This directory contains a web demo for the LiterT tensor library.

## How to build and run

1.  Build the wasm module:

    ```bash
    blaze build -c opt //third_party/odml/litert/tensor/wasm:litert_tensor_wasm
    ```

2.  Copy the wasm module and the demo files to a directory:

    ```bash
    mkdir /tmp/litert_demo
    cp blaze-bin/third_party/odml/litert/tensor/wasm/litert_tensor_wasm/litert_tensor_wasm_cc.js /tmp/litert_demo/
    cp blaze-bin/third_party/odml/litert/tensor/wasm/litert_tensor_wasm/litert_tensor_wasm_cc.wasm /tmp/litert_demo/
    cp third_party/odml/litert/tensor/wasm/demo/index.html /tmp/litert_demo/
    cp third_party/odml/litert/tensor/wasm/demo/index.js /tmp/litert_demo/
    ```

3.  Start a web server in the directory:

    ```bash
    python3 -m http.server --directory /tmp/litert_demo
    ```

4.  Open a web browser and navigate to `http://localhost:8000`.
