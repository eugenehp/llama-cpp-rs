plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.llama"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.llama"
        // minSdk 28 matches ANDROID_PLATFORM=android-28 used by llama-cpp-sys-4's
        // build.rs when cross-compiling the native library.
        minSdk = 28
        targetSdk = 34
        versionCode = 1
        versionName = "0.5.1"
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    // `libllama_jni.so` is produced by `cargo ndk` and dropped here; see
    // ../README.md. Kept out of the repo (built artifact).
    sourceSets["main"].jniLibs.srcDirs("src/main/jniLibs")
}

dependencies {
    implementation("androidx.appcompat:appcompat:1.7.0")
}
