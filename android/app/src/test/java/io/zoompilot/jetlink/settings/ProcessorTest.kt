package io.zoompilot.jetlink.settings

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** What each kind of phone may choose to run the model on. */
class ProcessorTest {
    @Test
    fun aSnapdragonOffersQnn() {
        assertEquals(
            listOf(Processor.Auto, Processor.Npu, Processor.Gpu, Processor.NpuGpu),
            Processor.choices(qualcomm = true, tensorNpu = false, emulator = false, current = Processor.NpuGpu),
        )
        // the CPU once chosen, so it can be chosen away from
        assertEquals(
            listOf(Processor.Auto, Processor.Npu, Processor.Gpu, Processor.NpuGpu, Processor.Cpu),
            Processor.choices(qualcomm = true, tensorNpu = false, emulator = false, current = Processor.Cpu),
        )
    }

    @Test
    fun aGoogleTensorOffersItsNpu() {
        assertEquals(
            listOf(Processor.Auto, Processor.TensorNpu, Processor.Gpu, Processor.Cpu),
            Processor.choices(qualcomm = false, tensorNpu = true, emulator = false, current = Processor.Auto),
        )
    }

    @Test
    fun anyOtherPhoneOffersTheGpuAndCpu() {
        // QNN on a Google Tensor leaves every op to one CPU thread; LiteRT drives its GPU
        assertEquals(
            listOf(Processor.Auto, Processor.Gpu, Processor.Cpu),
            Processor.choices(qualcomm = false, tensorNpu = false, emulator = false, current = Processor.Gpu),
        )
    }

    @Test
    fun automaticPicksATensorsNpuAndElseTheGpu() {
        assertEquals(Processor.TensorNpu, Processor.automatic(tensorNpu = true, emulator = false))
        // a Snapdragon too, until QNN's NPU outputs are checked on a phone
        assertEquals(Processor.Gpu, Processor.automatic(tensorNpu = false, emulator = false))
        assertEquals(Processor.Cpu, Processor.automatic(tensorNpu = false, emulator = true))
        assertEquals(Processor.TensorNpu, Processor.Auto.resolved(Processor.TensorNpu))
        assertEquals(Processor.Gpu, Processor.Gpu.resolved(Processor.TensorNpu))
    }

    @Test
    fun theOldDefaultBecomesAutomatic() {
        // every phone's default before Automatic; chosen choices stay
        assertEquals("auto", Settings.migratedProcessor("gpu"))
        assertEquals("auto", Settings.migratedProcessor("litert-gpu"))
        assertEquals(null, Settings.migratedProcessor("htp"))
        assertEquals(null, Settings.migratedProcessor("cpu"))
        assertEquals(null, Settings.migratedProcessor(null))
    }

    @Test
    fun theQnnGpuChoiceBecomesLiteRtsGpu() {
        assertEquals(Processor.Gpu, Processor.of("gpu"))
        assertEquals(Processor.Gpu, Processor.of("litert-gpu"))
        assertEquals(Processor.NpuGpu, Processor.of("htp"))
        assertEquals(Processor.TensorNpu, Processor.of("npu"))
        assertEquals(Processor.Auto, Processor.of("auto"))
        assertEquals(null, Processor.of("tpu"))
    }

    @Test
    fun onlyOnnxruntimesNpuAndGpuChoicesNeedQnn() {
        assertEquals(listOf(Processor.Npu, Processor.NpuGpu, Processor.NpuGpuParallel), Processor.entries.filter { it.usesQnn })
        assertEquals(
            listOf(Backend.Ort, Backend.LiteRt, Backend.LiteRt, Backend.Ort, Backend.Ort, Backend.Ort),
            Processor.entries.filter { it != Processor.Auto }.map { it.backend },
        )
    }

    @Test
    fun theEmulatorOffersEverything() {
        assertEquals(Processor.entries.filter { it != Processor.NpuGpuParallel }, Processor.choices(false, false, true, Processor.Cpu))
        assertEquals(Processor.entries, Processor.choices(false, false, true, Processor.Cpu, parallel = true))
    }

    @Test
    fun theProcessorRowSaysWhatRuns() {
        assertEquals("Automatic (NPU)", Processor.Auto.summary("NPU(Tensor G5)", automatic = Processor.TensorNpu))
        assertEquals("Automatic (NPU)", Processor.Auto.summary(null, automatic = Processor.TensorNpu))
        assertEquals("Automatic (GPU)", Processor.Auto.summary(null, automatic = Processor.Gpu))
        assertEquals("GPU", Processor.Gpu.summary(null, automatic = Processor.TensorNpu))
    }

    @Test
    fun aTensorNpuThatTookNoModelSaysTheGpuRunsIt() {
        // what a Tensor phone off Google's EdgeTPU allowlist reports
        assertTrue(Processor.Auto.fellBack("GPU(fp16)", automatic = Processor.TensorNpu))
        assertEquals("Automatic (GPU)", Processor.Auto.summary("GPU(fp16)", automatic = Processor.TensorNpu))
        assertEquals("NPU (GPU)", Processor.TensorNpu.summary("GPU(fp16)", automatic = Processor.TensorNpu))
        assertFalse(Processor.Auto.fellBack("NPU(Tensor G5)", automatic = Processor.TensorNpu))
        // only a Tensor NPU falls back; nothing loaded says nothing
        assertFalse(Processor.Gpu.fellBack("GPU(fp16)", automatic = Processor.TensorNpu))
        assertFalse(Processor.Auto.fellBack(null, automatic = Processor.TensorNpu))
        assertEquals("GPU", Processor.hardwareOf("GPU(fp16)"))
        assertEquals(null, Processor.hardwareOf(""))
    }

    @Test
    fun aTensorsGenerationComesFromItsSocModel() {
        assertEquals(5, Chip.tensorGeneration("Tensor G5"))
        assertEquals(3, Chip.tensorGeneration("Tensor G3"))
        assertEquals(null, Chip.tensorGeneration("SM8650"))
        assertEquals(null, Chip.tensorGeneration("Tensor"))
    }

    @Test
    fun theParallelExperimentIsBuildOptInForQualcommAndReportsItsFallback() {
        assertFalse(Processor.choices(true, false, false, Processor.Npu).contains(Processor.NpuGpuParallel))
        assertTrue(Processor.choices(true, false, false, Processor.Npu, parallel = true).contains(Processor.NpuGpuParallel))
        assertFalse(Processor.choices(false, true, false, Processor.TensorNpu, parallel = true).contains(Processor.NpuGpuParallel))
        assertEquals(Processor.NpuGpuParallel, Processor.of("htp-parallel"))
        assertTrue(Processor.NpuGpuParallel.fellBack("NPU", automatic = Processor.Gpu))
        assertFalse(Processor.NpuGpuParallel.fellBack("NPU + GPU", automatic = Processor.Gpu))
        assertEquals("NPU + GPU (experimental) (NPU)", Processor.NpuGpuParallel.summary("NPU", automatic = Processor.Gpu))
    }
}
