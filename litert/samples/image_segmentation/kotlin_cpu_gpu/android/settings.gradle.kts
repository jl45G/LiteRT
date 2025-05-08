pluginManagement {
  repositories {
    google {
      content {
        includeGroupByRegex("com\\.android.*")
        includeGroupByRegex("com\\.google.*")
        includeGroupByRegex("androidx.*")
      }
    }
    mavenCentral()
    gradlePluginPortal()
  }
}

dependencyResolutionManagement {
  repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
  repositories {
    google()
    mavenCentral()
  }
}

rootProject.name = "Image Segmentation"

include(":app")

include(":selfie_multiclass_ai_pack")

// TODO: b/391631148 - Put these feature modules for NPU in a shared top-level directory, e.g.
// npu_runtime_libraries/.
include(":runtime_strings")

include(":qnn_runtime_v73")

include(":qnn_runtime_v75")

include(":qnn_runtime_v79")
