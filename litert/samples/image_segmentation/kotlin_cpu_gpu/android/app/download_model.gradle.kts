// Demonstrate how to download a model using gradle.
tasks.register("downloadDeeplab") {
    doLast {
        val downloadTask = de.undercouch.gradle.tasks.download.Download()
        downloadTask.src("https://storage.googleapis.com/ai-edge/interpreter-samples/image_segmentation/android/deeplab_v3.tflite")
        downloadTask.dest("${project.extra["ASSET_DIR"]}/deeplab_v3.tflite")
        downloadTask.overwrite(false)
        downloadTask.execute()
    }
}

tasks.named("preBuild") {
    dependsOn("downloadDeeplab")
}
