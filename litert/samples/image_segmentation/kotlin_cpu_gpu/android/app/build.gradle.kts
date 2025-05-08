// This command line flag is necessary to build the app with NPU support because NPU is available
// only on arm64 devices. CPU/GPU is available on all devices.
//
// Example:
//   ./gradlew build -PuseNpu=true
val useNpu = (project.findProperty("useNpu") as? String)?.toBoolean() ?: false

plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.jetbrains.kotlin.android)
  alias(libs.plugins.undercouch.download)
  alias(libs.plugins.compose.compiler)
}

android {
  namespace = "com.google.aiedge.examples.image_segmentation"
  compileSdk = 34

  defaultConfig {
    applicationId = "com.google.aiedge.examples.image_segmentation"
    minSdk = 31
    targetSdk = 33
    versionCode = 1
    versionName = "1.0"

    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    vectorDrawables { useSupportLibrary = true }

    if (useNpu) {
      ndk { abiFilters.add("arm64-v8a") }

      // needed for QNN skel libs
      packaging { jniLibs { useLegacyPackaging = true } }
    }
  }

  buildTypes {
    release {
      isMinifyEnabled = false
      proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
    }
  }
  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_1_8
    targetCompatibility = JavaVersion.VERSION_1_8
  }
  kotlinOptions { jvmTarget = "1.8" }
  buildFeatures { compose = true }
  packaging { resources { excludes += "/META-INF/{AL2.0,LGPL2.1}" } }

  if (useNpu) {
    dynamicFeatures.add(":qnn_runtime_v73")
    dynamicFeatures.add(":qnn_runtime_v75")
    dynamicFeatures.add(":qnn_runtime_v79")
  }

  assetPacks.add(":selfie_multiclass_ai_pack")

  bundle {
    deviceTargetingConfig = file("device_targeting_configuration.xml")
    deviceGroup {
      enableSplit = true // split bundle by #group
      defaultGroup = "other" // group used for standalone APKs
    }
  }

  // Disable lint analysis to avoid build failures due to lint errors.
  lint {
    disable.add("CoroutineCreationDuringComposition")
    disable.add("FlowOperatorInvokedInComposition")
    disable.add("Aligned16KB")
  }
}

// Import DownloadModels task
project.extensions.extraProperties["ASSET_DIR"] = "$projectDir/src/main/assets"

dependencies {
  if (useNpu) {
    implementation(project(":runtime_strings"))
  }

  // TODO(b/414723246): Replace with maven package.
  implementation(files("libs/litert_kotlin_api_aar.aar"))
  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.lifecycle.runtime.ktx)
  implementation(libs.androidx.lifecycle.runtime.compose)
  implementation(libs.androidx.lifecycle.viewmodel.compose)
  implementation(libs.androidx.activity.compose)
  implementation(platform(libs.androidx.compose.bom))
  implementation(libs.androidx.ui)
  implementation(libs.androidx.ui.graphics)
  implementation(libs.androidx.ui.tooling.preview)
  implementation(libs.androidx.material.icons.core)
  implementation(libs.androidx.material.icons.extended)
  implementation(libs.androidx.material2)
  implementation(libs.litert)
  implementation(libs.litert.gpu)
  implementation(libs.litert.gpu.api)
  implementation(libs.androidx.camera.core)
  implementation(libs.androidx.camera.lifecycle)
  implementation(libs.androidx.camera.view)
  implementation(libs.androidx.camera.camera2)
  implementation(libs.coil.compose)
  implementation(libs.androidx.compose.runtime.livedata)
  implementation(libs.android.play.ai.delivery)

  testImplementation(libs.junit)
  androidTestImplementation(libs.androidx.junit)
  androidTestImplementation(libs.androidx.espresso.core)
  androidTestImplementation(platform(libs.androidx.compose.bom))
  androidTestImplementation(libs.androidx.ui.test.junit4)
  debugImplementation(libs.androidx.ui.tooling)
  debugImplementation(libs.androidx.ui.test.manifest)
}
