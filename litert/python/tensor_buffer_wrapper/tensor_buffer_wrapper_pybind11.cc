// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "litert/python/tensor_buffer_wrapper/tensor_buffer_wrapper.h"
#include "third_party/pybind11/include/pybind11/functional.h"
#include "third_party/pybind11/include/pybind11/pybind11.h"
#include "third_party/pybind11/include/pybind11/stl.h"

namespace py = pybind11;

using litert::tensor_buffer_wrapper::TensorBufferWrapper;

PYBIND11_MODULE(_pywrap_litert_tensor_buffer_wrapper, m) {
  m.doc() = R"pbdoc(
    _pywrap_litert_tensor_buffer_wrapper
    Python bindings for LiteRT TensorBuffer.
  )pbdoc";

  // Factory methods to create TensorBufferWrapper objects
  m.def(
      "CreateManagedBuffer",
      [](int buffer_type, int element_type, py::object dimensions, 
         size_t buffer_size) {
        std::string error;
        TensorBufferWrapper* wrapper = TensorBufferWrapper::CreateManagedBuffer(
            buffer_type, element_type, dimensions.ptr(), buffer_size, &error);
        if (!wrapper) {
          throw std::runtime_error(error);
        }
        return wrapper;  // Ownership transferred to pybind11
      },
      py::arg("buffer_type"), py::arg("element_type"), 
      py::arg("dimensions"), py::arg("buffer_size"));

  m.def(
      "CreateFromHostMemory",
      [](int element_type, py::object dimensions, py::object host_memory) {
        std::string error;
        TensorBufferWrapper* wrapper = TensorBufferWrapper::CreateFromHostMemory(
            element_type, dimensions.ptr(), host_memory.ptr(), &error);
        if (!wrapper) {
          throw std::runtime_error(error);
        }
        return wrapper;
      },
      py::arg("element_type"), py::arg("dimensions"), py::arg("host_memory"));

  // Bindings for the TensorBufferWrapper class
  py::class_<TensorBufferWrapper>(m, "TensorBufferWrapper")
      // Buffer information methods
      .def("GetBufferType",
           [](TensorBufferWrapper& self) {
             PyObject* r = self.GetBufferType();
             if (!r) throw py::error_already_set();
             return py::reinterpret_steal<py::object>(r);
           })
      .def("GetTensorType",
           [](TensorBufferWrapper& self) {
             PyObject* r = self.GetTensorType();
             if (!r) throw py::error_already_set();
             return py::reinterpret_steal<py::object>(r);
           })
      .def("GetSize",
           [](TensorBufferWrapper& self) {
             PyObject* r = self.GetSize();
             if (!r) throw py::error_already_set();
             return py::reinterpret_steal<py::object>(r);
           })
      // Tensor data I/O methods
      .def("WriteTensor",
           [](TensorBufferWrapper& self, py::object data, const std::string& dtype) {
             PyObject* r = self.WriteTensor(data.ptr(), dtype);
             if (!r) throw py::error_already_set();
             return py::none();
           },
           py::arg("data"), py::arg("dtype"))
      .def("ReadTensor",
           [](TensorBufferWrapper& self, int num_elements, const std::string& dtype) {
             PyObject* r = self.ReadTensor(num_elements, dtype);
             if (!r) throw py::error_already_set();
             return py::reinterpret_steal<py::object>(r);
           },
           py::arg("num_elements"), py::arg("dtype"));
}