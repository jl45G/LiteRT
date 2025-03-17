/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "litert/python/compiled_model_wrapper/compiled_model_wrapper.h"
#include "third_party/pybind11/include/pybind11/functional.h"
#include "third_party/pybind11/include/pybind11/pybind11.h"
#include "third_party/pybind11/include/pybind11/stl.h"

namespace py = pybind11;

using litert::compiled_model_wrapper::CompiledModelWrapper;

PYBIND11_MODULE(_pywrap_litert_compiled_model_wrapper, m) {
  m.doc() = R"pbdoc(
    _pywrap_litert_compiled_model_wrapper
    Python bindings for LiteRT CompiledModel.
  )pbdoc";

  // Factory method to create a CompiledModelWrapper from a model file.
  m.def(
      "CreateCompiledModelFromFile",
      [](const std::string& model_path, const std::string& compiler_plugin_path,
         const std::string& dispatch_library_path, int hardware_accel) {
        std::string error;
        CompiledModelWrapper* wrapper =
            CompiledModelWrapper::CreateWrapperFromFile(
                model_path.c_str(),
                compiler_plugin_path.empty() ? nullptr
                                             : compiler_plugin_path.c_str(),
                dispatch_library_path.empty() ? nullptr
                                              : dispatch_library_path.c_str(),
                hardware_accel, &error);
        if (!wrapper) {
          throw std::runtime_error(error);
        }
        return wrapper;  // Ownership transferred to pybind11
      },
      py::arg("model_path"), py::arg("compiler_plugin_path") = "",
      py::arg("dispatch_library_path") = "", py::arg("hardware_accel") = 0);

  // Factory method to create a CompiledModelWrapper from a model buffer.
  m.def(
      "CreateCompiledModelFromBuffer",
      [](py::bytes model_data, const std::string& compiler_plugin_path,
         const std::string& dispatch_library_path, int hardware_accel) {
        std::string error;
        PyObject* data_obj = model_data.ptr();
        CompiledModelWrapper* wrapper =
            CompiledModelWrapper::CreateWrapperFromBuffer(
                data_obj,
                compiler_plugin_path.empty() ? nullptr
                                             : compiler_plugin_path.c_str(),
                dispatch_library_path.empty() ? nullptr
                                              : dispatch_library_path.c_str(),
                hardware_accel, &error);
        if (!wrapper) {
          throw std::runtime_error(error);
        }
        return wrapper;
      },
      py::arg("model_data"), py::arg("compiler_plugin_path") = "",
      py::arg("dispatch_library_path") = "", py::arg("hardware_accel") = 0);

  // Bindings for the CompiledModelWrapper class.
  py::class_<CompiledModelWrapper>(m, "CompiledModelWrapper")
      .def("GetSignatureList",
           [](CompiledModelWrapper& self) {
             PyObject* r = self.GetSignatureList();
             if (!r) throw py::error_already_set();
             return py::reinterpret_steal<py::object>(r);
           })
      .def("GetSignatureByIndex",
           [](CompiledModelWrapper& self, int index) {
             PyObject* r = self.GetSignatureByIndex(index);
             if (!r) throw py::error_already_set();
             return py::reinterpret_steal<py::object>(r);
           })
      .def("GetNumSignatures",
           [](CompiledModelWrapper& self) {
             PyObject* r = self.GetNumSignatures();
             if (!r) throw py::error_already_set();
             return py::reinterpret_steal<py::object>(r);
           })
      .def("GetSignatureIndex",
           [](CompiledModelWrapper& self, const std::string& key) {
             PyObject* r = self.GetSignatureIndex(key.c_str());
             if (!r) throw py::error_already_set();
             return py::reinterpret_steal<py::object>(r);
           })
      .def("GetInputBufferRequirements",
           [](CompiledModelWrapper& self, int sig_idx, int in_idx) {
             PyObject* r = self.GetInputBufferRequirements(sig_idx, in_idx);
             if (!r) throw py::error_already_set();
             return py::reinterpret_steal<py::object>(r);
           })
      .def("GetOutputBufferRequirements",
           [](CompiledModelWrapper& self, int sig_idx, int out_idx) {
             PyObject* r = self.GetOutputBufferRequirements(sig_idx, out_idx);
             if (!r) throw py::error_already_set();
             return py::reinterpret_steal<py::object>(r);
           })
      .def("CreateInputBufferByName",
           [](CompiledModelWrapper& self, const std::string& sig_key,
              const std::string& input_name) {
             PyObject* r = self.CreateInputBufferByName(sig_key.c_str(),
                                                        input_name.c_str());
             if (!r) throw py::error_already_set();
             return py::reinterpret_steal<py::object>(r);
           })
      .def("CreateOutputBufferByName",
           [](CompiledModelWrapper& self, const std::string& sig_key,
              const std::string& out_name) {
             PyObject* r = self.CreateOutputBufferByName(sig_key.c_str(),
                                                         out_name.c_str());
             if (!r) throw py::error_already_set();
             return py::reinterpret_steal<py::object>(r);
           })
      .def("CreateInputBuffers",
           [](CompiledModelWrapper& self, int sig_index) {
             PyObject* r = self.CreateInputBuffers(sig_index);
             if (!r) throw py::error_already_set();
             return py::reinterpret_steal<py::object>(r);
           })
      .def("CreateOutputBuffers",
           [](CompiledModelWrapper& self, int sig_index) {
             PyObject* r = self.CreateOutputBuffers(sig_index);
             if (!r) throw py::error_already_set();
             return py::reinterpret_steal<py::object>(r);
           })

      .def("CreateTensorBufferFromMemory",
           [](CompiledModelWrapper& self, const std::string& sig_key,
              const std::string& tensor_name, py::object py_data,
              const std::string& dtype) {
             PyObject* r = self.CreateTensorBufferFromMemory(
                 sig_key.c_str(), tensor_name.c_str(), py_data.ptr(), dtype);
             if (!r) throw py::error_already_set();
             return py::reinterpret_steal<py::object>(r);
           })
      .def("RunByName",
           [](CompiledModelWrapper& self, const std::string& sig_key,
              py::object input_map, py::object output_map) {
             PyObject* r = self.RunByName(sig_key.c_str(), input_map.ptr(),
                                          output_map.ptr());
             if (!r) throw py::error_already_set();
             return py::none();
           })
      .def("RunByIndex",
           [](CompiledModelWrapper& self, int sig_index, py::object in_list,
              py::object out_list) {
             PyObject* r =
                 self.RunByIndex(sig_index, in_list.ptr(), out_list.ptr());
             if (!r) throw py::error_already_set();
             return py::none();
           })

      // Methods for tensor data I/O operations
      .def("WriteFloatTensor",
           [](CompiledModelWrapper& self, py::object buffer_capsule,
              py::object float_list) {
             PyObject* r =
                 self.WriteFloatTensor(buffer_capsule.ptr(), float_list.ptr());
             if (!r) throw py::error_already_set();
             return py::none();
           })
      .def("ReadFloatTensor",
           [](CompiledModelWrapper& self, py::object buffer_capsule,
              int num_floats) {
             PyObject* r =
                 self.ReadFloatTensor(buffer_capsule.ptr(), num_floats);
             if (!r) throw py::error_already_set();
             return py::reinterpret_steal<py::object>(r);
           })

      // Generic tensor I/O methods supporting multiple data types
      .def("WriteTensor",
           [](CompiledModelWrapper& self, py::object capsule, py::object data,
              const std::string& dtype) {
             PyObject* r = self.WriteTensor(capsule.ptr(), data.ptr(), dtype);
             if (!r) throw py::error_already_set();
             return py::none();
           })
      .def("ReadTensor",
           [](CompiledModelWrapper& self, py::object capsule, int num_elems,
              const std::string& dtype) {
             PyObject* r = self.ReadTensor(capsule.ptr(), num_elems, dtype);
             if (!r) throw py::error_already_set();
             return py::reinterpret_steal<py::object>(r);
           })

      .def("CreateTensorBufferFromMemory",
           [](CompiledModelWrapper& self, const std::string& sig_key,
              const std::string& tensor_name, py::object py_data,
              const std::string& dtype) {
             PyObject* r = self.CreateTensorBufferFromMemory(
                 sig_key.c_str(), tensor_name.c_str(), py_data.ptr(), dtype);
             if (!r) throw py::error_already_set();
             return py::reinterpret_steal<py::object>(r);
           })
      .def("DestroyTensorBuffer",
           [](CompiledModelWrapper& self, py::object buffer_capsule) {
             PyObject* r = self.DestroyTensorBuffer(buffer_capsule.ptr());
             if (!r) throw py::error_already_set();
             return py::none();
           });
}
