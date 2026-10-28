package io.zoompilot.jetlink.settings

import android.os.Build
import io.zoompilot.jetlink.BuildConfig

/**
 * The phone's SoC, and whether its NPU can run the model. QNN runs the
 * model in fp16 on the Hexagon NPU, which only v69 and later have: the
 * Snapdragon 8 Gen 1 and newer 8-series. The list is by Build.SOC_MODEL.
 * On a Google Tensor, LiteRT compiles the model for the NPU on the phone,
 * from the G3 on.
 */
object Chip {
    /** "SM8650", or the board name on older builds. */
    val model: String
        get() = Build.SOC_MODEL.takeIf { it.isNotBlank() && it != Build.UNKNOWN } ?: Build.BOARD

    val manufacturer: String get() = Build.SOC_MANUFACTURER

    val isQualcomm: Boolean
        get() = manufacturer.equals("QTI", ignoreCase = true) || manufacturer.equals("Qualcomm", ignoreCase = true)

    /** A Google Tensor whose NPU LiteRT compiles for: "Tensor G3" (the Pixel 8) or later. */
    val hasTensorNpu: Boolean by lazy { (tensorGeneration(model) ?: 0) >= 3 }

    private val tensorModel = Regex("^Tensor G(\\d+)$")

    /** 5 for "Tensor G5", as Build.SOC_MODEL names a Pixel's SoC; null for any other. */
    fun tensorGeneration(model: String): Int? = tensorModel.find(model.trim())?.groupValues?.get(1)?.toIntOrNull()

    /** What this phone's Settings offer. */
    fun processors(current: Processor): List<Processor> = Processor.choices(
        isQualcomm, hasTensorNpu, isEmulator, current, parallel = BuildConfig.EXPERIMENTAL_CAUSAL_PREFILL)

    /** What Automatic runs on here. */
    val automatic: Processor by lazy { Processor.automatic(hasTensorNpu, isEmulator) }

    val isEmulator: Boolean
        get() = Build.HARDWARE.contains("ranchu") || Build.HARDWARE.contains("goldfish") || Build.PRODUCT.contains("sdk")

    /** Known Snapdragons, by SoC model: name, Hexagon generation. */
    private val known = mapOf(
        "SM8350" to ("Snapdragon 888" to 68),
        "SM8450" to ("Snapdragon 8 Gen 1" to 69),
        "SM8475" to ("Snapdragon 8+ Gen 1" to 69),
        "SM8550" to ("Snapdragon 8 Gen 2" to 73),
        "SM8650" to ("Snapdragon 8 Gen 3" to 75),
        "SM8635" to ("Snapdragon 8s Gen 3" to 73),
        "SM8750" to ("Snapdragon 8 Elite" to 79),
        "SM8850" to ("Snapdragon 8 Elite Gen 5" to 81),
    )

    /** "Snapdragon 8 Gen 3", or the SoC model when this list does not know it. */
    val name: String get() = known[model]?.first ?: model

    /** The Hexagon NPU's generation, when known. */
    val hexagon: Int? get() = known[model]?.second

    /** How well this phone should do, before a benchmark says for sure. */
    enum class Expectation { Recommended, Possible, TooOld, Unmeasured }

    /**
     * From the NPU's generation, for QNN's choices: Qualcomm's published
     * numbers for similar models. No phone's GPU or Tensor NPU has been
     * measured.
     */
    fun expectation(processor: Processor): Expectation {
        if (!processor.usesQnn || !isQualcomm) return Expectation.Unmeasured
        val v = hexagon ?: return Expectation.Possible
        return when {
            v >= 75 -> Expectation.Recommended
            v >= 69 -> Expectation.Possible
            else -> Expectation.TooOld
        }
    }
}
