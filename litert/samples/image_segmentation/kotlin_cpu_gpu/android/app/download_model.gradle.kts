// Use simpler approach that doesn't require importing the Download class
tasks.register("downloadDeeplab") {
    doLast {
        // Use ant.get as a simpler alternative to the Download plugin
        ant.invokeMethod("get", mapOf(
            "src" to "https://storage.googleapis.com/ai-edge/interpreter-samples/image_segmentation/android/deeplab_v3.tflite",
            "dest" to "${project.extensions.extraProperties["ASSET_DIR"]}/deeplab_v3.tflite",
            "skipexisting" to "true"
        ))
    }
}

tasks.named("preBuild").configure {
    dependsOn("downloadDeeplab")
}