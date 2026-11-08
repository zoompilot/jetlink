import java.io.File
import java.security.MessageDigest
import javax.inject.Inject

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
}

/**
 * jetlink's version, from the Python package as the Mac and iPhone apps read
 * it: the release workflow's check ties the tag to the same line.
 */
val jetlinkVersion: String =
    Regex("""__version__ = '([^']+)'""")
        .find(rootDir.resolve("../jetlink/__init__.py").readText())
        ?.groupValues?.get(1) ?: error("no __version__ = '...' in jetlink/__init__.py")

/** 0.5.0 is 500: releases only go up. */
val jetlinkVersionCode: Int =
    jetlinkVersion.split(".").map { it.takeWhile(Char::isDigit).toIntOrNull() ?: 0 }
        .let { (it.getOrElse(0) { 0 } * 10000) + (it.getOrElse(1) { 0 } * 100) + it.getOrElse(2) { 0 } }

// All options are build-time opt-ins; ordinary releases retain the Maven runtime.
fun qnnFlag(name: String): Boolean = providers.gradleProperty(name).orNull?.let {
    it.toBooleanStrictOrNull() ?: error("$name must be true or false")
} ?: false
val qnnSharedIO = qnnFlag("jetlink.qnnSharedIO")
val qnnStrict = qnnFlag("jetlink.qnnStrict")
val qnnNoActivationFusion = qnnFlag("jetlink.qnnNoActivationFusion")
val qnnJniDir = providers.gradleProperty("jetlink.qnnJniDir").orNull
val qnnJniRoot = qnnJniDir?.let { file(it) }
val qnnRuntimeIdentity = if (qnnNoActivationFusion) {
    check(qnnJniDir != null) { "no-fusion requires one matching SDK; see docs/android-qnn-npu.md" }
    val directory = File(qnnJniRoot!!, "arm64-v8a")
    val libraries = directory.listFiles()?.filter { it.isFile && it.extension == "so" }?.sortedBy { it.name }.orEmpty()
    val names = libraries.map { it.name }.toSet()
    check(names.containsAll(listOf("libJetlinkQnnHtpNoFusion.so", "libQnnHtp.so", "libQnnGpu.so", "libQnnSystem.so",
        "libQnnHtpPrepare.so", "libQnnIr.so", "libQnnSaver.so"))) { "incomplete SDK jniLibs directory: $directory" }
    val stub = names.filter { it.matches(Regex("libQnnHtpV\\d+Stub\\.so")) }.map { it.removeSuffix("Stub.so") }.toSet()
    val skel = names.filter { it.matches(Regex("libQnnHtpV\\d+Skel\\.so")) }.map { it.removeSuffix("Skel.so") }.toSet()
    check(stub.isNotEmpty() && stub == skel) { "matching HTP stub/skel pairs are required" }
    // Actual runtime bytes form a cache namespace, not a preselected SDK/SoC pin.
    val identity = MessageDigest.getInstance("SHA-256")
    for (library in libraries) {
        val digest = MessageDigest.getInstance("SHA-256")
        library.inputStream().use { stream ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        identity.update(library.name.toByteArray(Charsets.UTF_8))
        identity.update(0.toByte())
        identity.update(digest.digest())
    }
    identity.digest().joinToString("") { "%02x".format(it) }
} else {
    check(qnnJniDir == null) { "jetlink.qnnJniDir requires jetlink.qnnNoActivationFusion=true" }
    ""
}

android {
    namespace = "io.zoompilot.jetlink"
    compileSdk = 37
    // The NDK the Swift SDK for Android was built with; AGP strips the
    // native libraries with it.
    ndkVersion = "30.0.16248370"

    defaultConfig {
        applicationId = "io.zoompilot.jetlink.android"
        // Android 12: Build.SOC_MODEL and the performance hint API.
        minSdk = 31
        targetSdk = 36
        versionCode = jetlinkVersionCode
        versionName = jetlinkVersion
        buildConfigField("boolean", "QNN_SHARED_IO", qnnSharedIO.toString())
        buildConfigField("boolean", "QNN_STRICT", qnnStrict.toString())
        buildConfigField("boolean", "QNN_NO_ACTIVATION_FUSION", qnnNoActivationFusion.toString())
        buildConfigField("String", "QNN_RUNTIME_IDENTITY", "\"$qnnRuntimeIdentity\"")
        // The QNN runtime is arm64 only, and so is every Snapdragon with an NPU
        // worth driving on. The emulator on an Apple silicon Mac is arm64 too.
        ndk { abiFilters += "arm64-v8a" }
    }

    signingConfigs {
        // The key the APKs on the releases page are signed with, which the
        // Release workflow writes from its secrets (docs/publishing.md).
        // Android installs an update only over an app signed with the same
        // key, so every release has to use this one.
        fun env(name: String) = providers.environmentVariable(name).orNull?.takeIf(String::isNotEmpty)
        env("JETLINK_ANDROID_KEYSTORE")?.let { keystore ->
            create("release") {
                storeFile = file(keystore)
                storePassword = env("JETLINK_ANDROID_KEYSTORE_PASSWORD") ?: error("JETLINK_ANDROID_KEYSTORE_PASSWORD is not set")
                // keytool's PKCS12 keystore has one password, the key's too
                keyPassword = storePassword
                keyAlias = env("JETLINK_ANDROID_KEY_ALIAS") ?: error("JETLINK_ANDROID_KEY_ALIAS is not set")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            // Sideloaded, never on a store. Without the release key, the debug
            // key: a fork or a local build still builds, and a local release
            // build installs over a debug one.
            signingConfig = signingConfigs.findByName("release") ?: signingConfigs.getByName("debug")
        }
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    packaging {
        jniLibs {
            // The NPU's skel libraries are loaded by the DSP from a real file,
            // so native libraries are extracted to nativeLibraryDir.
            useLegacyPackaging = true
            // onnxruntime's Java binding; the server calls the C API.
            excludes += "**/libonnxruntime4j_jni.so"
        }
    }

    if (qnnNoActivationFusion) sourceSets.getByName("main").jniLibs.srcDir(qnnJniRoot!!)

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    testOptions {
        unitTests.isReturnDefaultValues = true
    }
}

/** The onnxruntime AAR on its own, for its C headers. */
val onnxruntimeAar: Configuration by configurations.creating {
    isCanBeConsumed = false
    isTransitive = false
}

/**
 * The LiteRT AAR on its own, for its native libraries alone: libLiteRt.so and
 * its GPU accelerator, which the server opens from nativeLibraryDir. Its
 * Kotlin API and what that pulls in (litert-api, Guava, Play's AI delivery)
 * are not wanted.
 */
val litertAar: Configuration by configurations.creating {
    isCanBeConsumed = false
    isTransitive = false
}

/** LiteRT's NPU runtime libraries, a zip from its GitHub release (settings.gradle.kts). */
val litertNpu: Configuration by configurations.creating {
    isCanBeConsumed = false
    isTransitive = false
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.lifecycle.service)
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.graphics)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.compose.material3)
    implementation(libs.compose.material.icons)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.kotlinx.serialization.json)
    // libonnxruntime.so with the QNN provider, and the QNN runtime's libraries
    implementation(libs.onnxruntime.qnn) {
        if (qnnNoActivationFusion) exclude(group = "com.qualcomm.qti", module = "qnn-runtime")
    }
    debugImplementation(libs.compose.ui.tooling)

    onnxruntimeAar(variantOf(libs.onnxruntime.qnn) { artifactType("aar") })
    litertAar(variantOf(libs.litert) { artifactType("aar") })
    litertNpu(variantOf(libs.litert.npu) { artifactType("zip") })

    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
}

/**
 * Builds libjetlink.so, the server, with the Swift SDK for Android
 * (scripts/swift-build.sh). Always optimized: the frame path in an
 * unoptimized build is too slow to judge anything by.
 *
 * -Pjetlink.prebuiltSwift=DIR skips the Swift build and packages DIR/arm64-v8a
 * instead, for working on the Kotlin side without the Swift toolchain.
 */
abstract class SwiftBuild @Inject constructor(private val exec: ExecOperations) : DefaultTask() {
    @get:OutputDirectory abstract val outputDir: DirectoryProperty
    @get:InputFiles @get:PathSensitive(PathSensitivity.RELATIVE) abstract val sources: ConfigurableFileCollection
    @get:InputFiles @get:PathSensitive(PathSensitivity.NONE) abstract val onnxruntime: ConfigurableFileCollection
    @get:Internal abstract val script: RegularFileProperty
    @get:Input @get:Optional abstract val prebuilt: Property<String>
    /** The prebuilt libraries' contents, so a new copy there is picked up. */
    @get:InputFiles @get:PathSensitive(PathSensitivity.RELATIVE) abstract val prebuiltFiles: ConfigurableFileCollection

    @TaskAction
    fun build() {
        val out = outputDir.get().asFile
        val prebuiltDir = prebuilt.orNull
        if (prebuiltDir != null) {
            check(File(prebuiltDir, "arm64-v8a/libjetlink.so").isFile) { "$prebuiltDir has no arm64-v8a/libjetlink.so" }
            File(prebuiltDir).copyRecursively(out, overwrite = true)
            return
        }
        exec.exec {
            commandLine("bash", script.get().asFile.absolutePath, out.absolutePath, onnxruntime.singleFile.absolutePath, "release")
        }
    }
}

val swiftBuild = tasks.register<SwiftBuild>("swiftBuild") {
    val kit = rootDir.resolve("../JetlinkKit")
    sources.from(kit.resolve("Package.swift"), fileTree(kit.resolve("Sources")))
    onnxruntime.from(onnxruntimeAar)
    script.set(rootDir.resolve("scripts/swift-build.sh"))
    prebuilt.set(providers.gradleProperty("jetlink.prebuiltSwift"))
    providers.gradleProperty("jetlink.prebuiltSwift").orNull?.let { prebuiltFiles.from(fileTree(it)) }
    outputDir.set(layout.buildDirectory.dir("swift/jniLibs"))
}

/**
 * LiteRT's runtime and GPU accelerator out of its AAR, and Google Tensor's
 * dispatch library and compiler plugin out of the same release's NPU zip,
 * for arm64 as the rest. LiteRT opens them all from nativeLibraryDir. The
 * zip comes from GitHub rather than Maven, so its hash is checked.
 */
abstract class LiteRtLibraries @Inject constructor(
    private val files: FileSystemOperations,
    private val archives: ArchiveOperations,
) : DefaultTask() {
    @get:InputFiles @get:PathSensitive(PathSensitivity.NONE) abstract val aar: ConfigurableFileCollection
    @get:InputFiles @get:PathSensitive(PathSensitivity.NONE) abstract val npu: ConfigurableFileCollection
    @get:Input abstract val npuSha256: Property<String>
    @get:OutputDirectory abstract val outputDir: DirectoryProperty

    @TaskAction
    fun extract() {
        val zip = npu.singleFile
        val sha256 = MessageDigest.getInstance("SHA-256").digest(zip.readBytes()).joinToString("") { "%02x".format(it) }
        check(sha256 == npuSha256.get()) { "${zip.name} has SHA-256 $sha256, not the ${npuSha256.get()} jetlink was built against" }
        files.sync {
            from(archives.zipTree(aar.singleFile)) {
                include("jni/arm64-v8a/libLiteRt.so", "jni/arm64-v8a/libLiteRtClGlAccelerator.so")
                eachFile { path = "arm64-v8a/$name" }
            }
            from(archives.zipTree(zip)) {
                include(
                    "google_tensor_runtime/src/main/jni/arm64-v8a/libLiteRtDispatch_GoogleTensor.so",
                    "google_tensor_runtime/src/main/jni/arm64-v8a/libLiteRtCompilerPlugin_google_tensor.so",
                )
                eachFile { path = "arm64-v8a/$name" }
                // the zip has them read-only, which the next sync could not replace
                filePermissions { unix("rw-r--r--") }
            }
            includeEmptyDirs = false
            into(outputDir)
        }
    }
}

val liteRtLibraries = tasks.register<LiteRtLibraries>("liteRtLibraries") {
    aar.from(litertAar)
    npu.from(litertNpu)
    // litert_npu_runtime_libraries_jit.zip of LiteRT v2.2.0
    npuSha256.set("d6d160104f110e690f1c9b0c54ab3855af1a12e405dcc6e644dd7e3e1955aac4")
    outputDir.set(layout.buildDirectory.dir("litert/jniLibs"))
}

androidComponents {
    onVariants { variant ->
        variant.sources.jniLibs?.addGeneratedSourceDirectory(swiftBuild, SwiftBuild::outputDir)
        variant.sources.jniLibs?.addGeneratedSourceDirectory(liteRtLibraries, LiteRtLibraries::outputDir)
    }
}
