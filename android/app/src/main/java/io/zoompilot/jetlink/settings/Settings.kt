package io.zoompilot.jetlink.settings

import android.content.Context
import android.content.SharedPreferences
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** The server's backends, as its start config names them. */
enum class Backend(val id: String) {
    Ort("ort"),
    LiteRt("litert"),
}

/**
 * Where the model runs: a backend of the server's and its device, an
 * OrtProfile or a LiteRtProfile, or Automatic, which picks one of those for
 * the phone. The device is also what Settings stores, and no two choices
 * share one.
 */
enum class Processor(val backend: Backend, val device: String, val title: String) {
    /** The phone's NPU where Jetlink has one for it, else the GPU ([automatic]). */
    Auto(Backend.LiteRt, "auto", "Automatic"),

    /**
     * The whole model on the NPU, prepared as the iPhone's. QNN, a Snapdragon's: the fastest
     * choice measured on an 8+ Gen 1, several times the GPU's speed.
     */
    Npu(Backend.Ort, "htp-whole", "NPU"),

    /**
     * A Google Tensor's NPU, which LiteRT compiles the model for on the phone
     * through the compiler in its system; the GPU runs a model that compiler
     * cannot take.
     */
    TensorNpu(Backend.LiteRt, "npu", "NPU"),

    /** The whole model on the GPU through LiteRT, which drives any phone's: Adreno, Mali, PowerVR. */
    Gpu(Backend.LiteRt, "gpu", "GPU"),

    /**
     * The vision trunk on the NPU, the rest on the GPU: the Mac's split. QNN, a Snapdragon's.
     * Listed after the GPU: the slowest of the three measured on an 8+ Gen 1.
     */
    NpuGpu(Backend.Ort, "htp", "NPU + GPU"),

    /** Same-frame causal work in parallel; exact model contracts only, otherwise whole NPU. */
    NpuGpuParallel(Backend.Ort, "htp-parallel", "NPU + GPU (experimental)"),

    /** The CPU: the emulator and tests. Seconds a frame with a real model. */
    Cpu(Backend.Ort, "cpu", "CPU");

    /** Runs on QNN, which needs a Snapdragon: onnxruntime anywhere but its CPU. */
    val usesQnn: Boolean get() = backend == Backend.Ort && device != Cpu.device

    /** The choice the server runs: Automatic's pick for this phone, or this one. */
    fun resolved(automatic: Processor = Chip.automatic): Processor = if (this == Auto) automatic else this

    /**
     * Whether the loaded model runs on the GPU though this choice is a Tensor
     * NPU: the NPU took no model. [accelerator] is the ready engine's.
     */
    fun fellBack(accelerator: String?, automatic: Processor = Chip.automatic): Boolean =
        when (resolved(automatic)) {
            TensorNpu -> hardwareOf(accelerator).let { it != null && it != "NPU" }
            NpuGpuParallel -> hardwareOf(accelerator) == "NPU"
            else -> false
        }

    /** What the Processor row says: the choice, and with Automatic what it runs on. */
    fun summary(accelerator: String?, automatic: Processor = Chip.automatic): String {
        val runsOn = if (fellBack(accelerator, automatic)) hardwareOf(accelerator)!! else resolved(automatic).title
        return when {
            this == Auto -> "Automatic ($runsOn)"
            fellBack(accelerator, automatic) -> "$title ($runsOn)"
            else -> title
        }
    }

    companion object {
        /**
         * A stored choice. Older apps stored the GPU as `litert-gpu`, which
         * reads as [Gpu].
         */
        fun of(device: String?): Processor? = if (device == "litert-gpu") Gpu else entries.firstOrNull { it.device == device }

        /** "NPU", "GPU" or "CPU": the hardware an engine's accelerator names, "NPU(Tensor G5)". */
        fun hardwareOf(accelerator: String?): String? = accelerator?.substringBefore('(')?.trim()?.takeIf(String::isNotEmpty)

        /**
         * What a phone can choose, Automatic first. The Snapdragon NPU
         * choices run QNN, which needs a Snapdragon: anywhere else it leaves
         * every op to one CPU thread, minutes a frame. A Google Tensor G3 or
         * later has its own NPU choice. The GPU is LiteRT's, on any phone. The
         * CPU is offered on a Snapdragon only once chosen; the emulator offers
         * everything, for testing.
         */
        fun choices(qualcomm: Boolean, tensorNpu: Boolean, emulator: Boolean, current: Processor, parallel: Boolean = false): List<Processor> = when {
            emulator -> entries.filter { it != NpuGpuParallel || parallel }
            qualcomm -> entries.filter { it != TensorNpu && (it != Cpu || current == Cpu) && (it != NpuGpuParallel || parallel) }
            tensorNpu -> listOf(Auto, TensorNpu, Gpu, Cpu)
            else -> listOf(Auto, Gpu, Cpu)
        }

        /**
         * Automatic's pick: a Google Tensor's NPU, which runs on the GPU any
         * model it cannot compile; the CPU on the emulator; else the GPU. A
         * Snapdragon stays on the GPU: QNN's NPU has run on no phone, and 20
         * of Cinque Terre V3's 44 vision LayerNorms overflow float16 when
         * computed step by step, which would give wrong outputs rather than
         * an error to fall back on.
         */
        fun automatic(tensorNpu: Boolean, emulator: Boolean): Processor = when {
            emulator -> Cpu
            tensorNpu -> TensorNpu
            else -> Gpu
        }

    }
}

/** The few things worth changing on a phone, kept in SharedPreferences. */
data class SettingsValues(
    /** Where bench tools such as `bench_link.py --host` reach the phone, with [developer] on. */
    val port: Int = 5599,
    val processor: Processor = Processor.Auto,
    /** The NPU held in burst mode between frames rather than let it settle. */
    val keepNpuAwake: Boolean = true,
    /** A CPU core kept busy between frames. */
    val keepCpuAwake: Boolean = false,
    /** The screen stays on while Jetlink is on screen. */
    val keepScreenOn: Boolean = true,
    /**
     * Seven taps on Version turn it on: the Developer section, and the server
     * listening on [port] for bench tools. The comma needs neither. The
     * emulator, with no USB host, starts with it on.
     */
    val developer: Boolean = false,
)

class Settings(context: Context) {
    private val prefs: SharedPreferences = context.getSharedPreferences("settings", Context.MODE_PRIVATE).also(::migrate)
    private val state = MutableStateFlow(read())
    val values: StateFlow<SettingsValues> = state.asStateFlow()

    fun update(change: (SettingsValues) -> SettingsValues) {
        val next = change(state.value)
        prefs.edit()
            .putInt(PORT, next.port)
            .putString(PROCESSOR, next.processor.device)
            .putBoolean(KEEP_NPU_AWAKE, next.keepNpuAwake)
            .putBoolean(KEEP_CPU_AWAKE, next.keepCpuAwake)
            .putBoolean(KEEP_SCREEN_ON, next.keepScreenOn)
            .putBoolean(DEVELOPER, next.developer)
            .apply()
        state.value = next
    }

    private fun read(): SettingsValues {
        val defaults = SettingsValues()
        val port = prefs.getInt(PORT, defaults.port)
        return SettingsValues(
            port = if (port in 1..65535) port else defaults.port,
            // a stored QNN choice on a phone without a Snapdragon falls back to the default
            processor = Processor.of(prefs.getString(PROCESSOR, null))
                ?.takeIf { it in Chip.processors(it) }
                ?: defaults.processor,
            keepNpuAwake = prefs.getBoolean(KEEP_NPU_AWAKE, defaults.keepNpuAwake),
            keepCpuAwake = prefs.getBoolean(KEEP_CPU_AWAKE, defaults.keepCpuAwake),
            keepScreenOn = prefs.getBoolean(KEEP_SCREEN_ON, defaults.keepScreenOn),
            developer = prefs.getBoolean(DEVELOPER, Chip.isEmulator),
        )
    }

    companion object {
        private const val VERSION = "version"
        private const val PORT = "port"
        private const val PROCESSOR = "processor"
        private const val KEEP_NPU_AWAKE = "keepNpuAwake"
        private const val KEEP_CPU_AWAKE = "keepCpuAwake"
        private const val KEEP_SCREEN_ON = "keepScreenOn"
        private const val DEVELOPER = "developer"

        /** Settings from before Automatic are version 1. */
        private const val CURRENT = 2

        /** Brings settings stored by an older app up to [CURRENT], once. */
        private fun migrate(prefs: SharedPreferences) {
            if (prefs.getInt(VERSION, 1) >= CURRENT) return
            val edit = prefs.edit().putInt(VERSION, CURRENT)
            migratedProcessor(prefs.getString(PROCESSOR, null))?.let { edit.putString(PROCESSOR, it) }
            edit.apply()
        }

        /**
         * A processor stored before Automatic existed: `gpu` was the default
         * then, so it becomes Automatic. Null leaves what is stored.
         */
        fun migratedProcessor(stored: String?): String? = if (Processor.of(stored) == Processor.Gpu) Processor.Auto.device else null
    }
}
