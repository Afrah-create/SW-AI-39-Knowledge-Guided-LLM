pluginManagement {
    val flutterSdkPath: String? = run {
        val localPropertiesFile = file("local.properties")
        if (!localPropertiesFile.exists()) {
            // If local.properties doesn't exist, try to get Flutter SDK from environment
            System.getenv("FLUTTER_ROOT")?.let { 
                println("Using FLUTTER_ROOT from environment: $it")
                return@run it
            }
            // Fallback: try common Flutter installation paths
            val commonPaths = listOf(
                System.getProperty("user.home") + "/flutter",
                "/opt/flutter",
                System.getenv("HOME") + "/flutter"
            )
            for (path in commonPaths) {
                val flutterBin = java.io.File(path, "bin/flutter")
                if (flutterBin.exists()) {
                    println("Found Flutter SDK at: $path")
                    return@run path
                }
            }
            println("Warning: local.properties not found and Flutter SDK not detected")
            null
        } else {
            try {
                val properties = java.util.Properties()
                localPropertiesFile.inputStream().use { properties.load(it) }
                val sdkPath = properties.getProperty("flutter.sdk")
                if (sdkPath != null && sdkPath.isNotBlank()) {
                    println("Using Flutter SDK from local.properties: $sdkPath")
                    sdkPath
                } else {
                    null
                }
            } catch (e: Exception) {
                println("Error reading local.properties: ${e.message}")
                null
            }
        }
    }

    if (flutterSdkPath != null) {
        val flutterGradlePath = file("$flutterSdkPath/packages/flutter_tools/gradle")
        if (flutterGradlePath.exists()) {
            includeBuild(flutterGradlePath.absolutePath)
        } else {
            println("Warning: Flutter Gradle plugin not found at: $flutterGradlePath")
        }
    } else {
        println("Warning: Flutter SDK path not found. Build may fail.")
    }

    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

plugins {
    id("dev.flutter.flutter-plugin-loader") version "1.0.0"
    id("com.android.application") version "8.3.0" apply false
    id("org.jetbrains.kotlin.android") version "1.9.22" apply false
}

include(":app")
