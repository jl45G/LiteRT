#ifndef THIRD_PARTY_ODML_LITERT_LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_JNI_COMMON_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_JNI_COMMON_H_

namespace litert {
namespace jni {

// Values here must match the values in the Kotlin enum
// com.google.ai.edge.litert.Accelerator.
constexpr int kAccelatorNone = 0;
constexpr int kAccelatorCpu = 1;
constexpr int kAccelatorGpu = 2;
constexpr int kAccelatorNpu = 3;

}  // namespace jni
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_JNI_COMMON_H_
