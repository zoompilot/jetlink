package io.zoompilot.jetlink.ui.settings

import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import io.zoompilot.jetlink.AppGraph
import io.zoompilot.jetlink.BuildConfig
import io.zoompilot.jetlink.server.RunState
import io.zoompilot.jetlink.server.ServerService
import io.zoompilot.jetlink.server.Snapshot
import io.zoompilot.jetlink.settings.Chip
import io.zoompilot.jetlink.settings.Processor
import io.zoompilot.jetlink.settings.SettingsValues
import io.zoompilot.jetlink.ui.Format
import io.zoompilot.jetlink.ui.JetlinkTheme
import io.zoompilot.jetlink.ui.PreviewData
import io.zoompilot.jetlink.ui.components.ActionRow
import io.zoompilot.jetlink.ui.components.CardSpacing
import io.zoompilot.jetlink.ui.components.FormFooter
import io.zoompilot.jetlink.ui.components.FormSection
import io.zoompilot.jetlink.ui.components.RowDivider
import io.zoompilot.jetlink.ui.components.SwitchRow
import io.zoompilot.jetlink.ui.components.ValueRow
import io.zoompilot.jetlink.ui.components.rememberWifiAddress
import io.zoompilot.jetlink.usb.UsbState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** What the Settings rows show beyond the settings themselves. */
data class SettingsInfo(
    val snapshot: Snapshot,
    val runState: RunState,
    val usb: UsbState,
    val wifi: String?,
    /** onnxruntime's version, from the server's info. */
    val runtime: String?,
    val chip: String,
    val version: String,
    /** What the Processor row offers. */
    val processors: List<Processor>,
    /** A Snapdragon, whose NPU and GPU QNN can drive. */
    val snapdragon: Boolean = true,
    /** What Automatic runs on here. */
    val automatic: Processor = Processor.Gpu,
    /** What runs the loaded model, when its engine says ("GPU(fp16)"). */
    val accelerator: String? = null,
)

/** What the Settings rows do. */
class SettingsActions(
    val update: ((SettingsValues) -> SettingsValues) -> Unit = {},
    val openConnect: () -> Unit = {},
    val openLogs: () -> Unit = {},
    val stopServer: () -> Unit = {},
    val startServer: () -> Unit = {},
    val restartServer: () -> Unit = {},
)

/**
 * Connection, performance, display, storage, help and versions. A change the
 * server runs with restarts it (ServerService does that).
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(graph: AppGraph, openConnect: () -> Unit, openLogs: () -> Unit) {
    val context = LocalContext.current
    val values by graph.settings.values.collectAsStateWithLifecycle()
    val snapshot by graph.server.snapshot.collectAsStateWithLifecycle()
    val runState by graph.server.runState.collectAsStateWithLifecycle()
    val usb by graph.usb.usb.collectAsStateWithLifecycle()
    val wifi = rememberWifiAddress()
    val runtime by produceState<String?>(null, runState) {
        value = withContext(Dispatchers.IO) { graph.server.runtimeVersion() }
    }
    val info = SettingsInfo(
        snapshot = snapshot,
        runState = runState,
        usb = usb,
        wifi = wifi,
        runtime = snapshot.server?.runtimeVersion ?: runtime,
        chip = Chip.name,
        version = BuildConfig.VERSION_NAME,
        processors = Chip.processors(values.processor),
        snapdragon = Chip.isQualcomm || Chip.isEmulator,
        automatic = Chip.automatic,
        accelerator = snapshot.engine.takeIf { it.state == "ready" }?.accelerator,
    )
    val actions = SettingsActions(
        update = graph.settings::update,
        openConnect = openConnect,
        openLogs = openLogs,
        stopServer = { ServerService.start(context, ServerService.ACTION_STOP) },
        startServer = { ServerService.start(context) },
        restartServer = { graph.restartServer(context) },
    )
    Scaffold(
        containerColor = JetlinkTheme.colors.grouped,
        topBar = {
            TopAppBar(
                title = { Text("Settings", fontWeight = FontWeight.Bold) },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = JetlinkTheme.colors.grouped,
                    scrolledContainerColor = JetlinkTheme.colors.grouped,
                ),
            )
        },
    ) { padding ->
        SettingsContent(values, info, actions, Modifier.padding(padding).consumeWindowInsets(padding))
    }
}

@Composable
fun SettingsContent(values: SettingsValues, info: SettingsInfo, actions: SettingsActions, modifier: Modifier = Modifier) {
    Column(
        modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = CardSpacing)
            .padding(top = 8.dp, bottom = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(20.dp),
    ) {
        Connection(info)
        Performance(values, info, actions)
        FormSection("Display", footer = { FormFooter("Jetlink keeps serving with the screen off.") }) {
            SwitchRow("Keep Screen On", values.keepScreenOn, { on -> actions.update { it.copy(keepScreenOn = on) } })
        }
        info.snapshot.disk?.let { disk ->
            FormSection("Storage") {
                ValueRow("Downloaded", Format.bytes(disk.modelsBytes))
                RowDivider()
                ValueRow("Prepared", Format.bytes(disk.enginesBytes))
                disk.freeBytes?.let {
                    RowDivider()
                    ValueRow("Available", Format.bytes(it))
                }
            }
        }
        FormSection("Help") {
            ActionRow("Connecting the Comma", actions.openConnect)
            RowDivider()
            ActionRow("Logs", actions.openLogs)
        }
        if (values.developer) {
            Developer(values, info, actions)
        }
        About(values, info, actions)
    }
}

@Composable
private fun Connection(info: SettingsInfo) {
    val link = info.snapshot.medium?.title
        ?: if (info.usb is UsbState.Attached) "Connecting" else "Not Connected"
    FormSection("Connection") {
        ValueRow("Link", link)
    }
}

/**
 * The port bench tools reach the server on, and where on Wi-Fi. Shown, and
 * the port open, only with the developer setting on.
 */
@Composable
private fun Developer(values: SettingsValues, info: SettingsInfo, actions: SettingsActions) {
    val colors = JetlinkTheme.colors
    val focus = LocalFocusManager.current
    var portText by remember(values.port) { mutableStateOf(values.port.toString()) }
    val typed = portText.toIntOrNull()?.takeIf { it in 1..65535 }
    val apply = {
        if (typed != null && typed != values.port) {
            actions.update { it.copy(port = typed) }
        } else {
            portText = values.port.toString()
        }
        focus.clearFocus()
    }
    val footer = "For bench tools on a Mac over Wi-Fi or adb. The comma does not use this port. Tap Version 7 times to hide this."
    FormSection("Developer", footer = { FormFooter(footer) }) {
        Row(
            Modifier.fillMaxWidth().heightIn(min = 48.dp).padding(horizontal = 16.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Port", style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f))
            BasicTextField(
                value = portText,
                onValueChange = { text -> portText = text.filter(Char::isDigit).take(5) },
                singleLine = true,
                textStyle = MaterialTheme.typography.bodyLarge.copy(color = colors.secondaryText, textAlign = TextAlign.End),
                cursorBrush = SolidColor(MaterialTheme.colorScheme.primary),
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number, imeAction = ImeAction.Done),
                keyboardActions = KeyboardActions(onDone = { apply() }),
                modifier = Modifier.width(96.dp),
            )
        }
        if (typed != null && typed != values.port) {
            RowDivider()
            ActionRow("Use Port $typed", { apply() }, color = MaterialTheme.colorScheme.primary, chevron = false)
        }
        info.wifi?.let { address ->
            RowDivider()
            Row(
                Modifier.fillMaxWidth().heightIn(min = 48.dp).padding(horizontal = 16.dp, vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("Wi-Fi", style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f))
                SelectionContainer {
                    Text("$address:${info.snapshot.port ?: values.port}", style = MaterialTheme.typography.bodyLarge, color = colors.secondaryText)
                }
            }
        }
    }
}

@Composable
private fun Performance(values: SettingsValues, info: SettingsInfo, actions: SettingsActions) {
    val colors = JetlinkTheme.colors
    var choosing by remember { mutableStateOf(false) }
    val choices = info.processors
    val footer = when {
        values.processor.resolved(info.automatic) == Processor.NpuGpuParallel &&
            values.processor.fellBack(info.accelerator, info.automatic) ->
            "This model has no validated causal split, so it runs on the whole NPU. " +
                "Run Benchmark and check accuracy on this phone before use."
        values.processor.fellBack(info.accelerator, info.automatic) ->
            "The NPU did not take this model, so it runs on the GPU. Logs (in Help) say why. " +
                "Changing the processor prepares models again."
        info.automatic == Processor.TensorNpu ->
            "Automatic runs the model on the Tensor NPU, which prepares it the first time. " +
                "A model the NPU cannot take runs on the GPU. Changing the processor prepares models again."
        info.snapdragon -> "Automatic runs the model on the GPU for now. Changing the processor prepares models again."
        else -> "This phone has no NPU Jetlink can use, so the model runs on its GPU. Run Benchmark to see whether it keeps up."
    }
    FormSection("Performance", footer = { FormFooter(footer) }) {
        Box {
            Row(
                Modifier
                    .fillMaxWidth()
                    .clickable(role = Role.DropdownList) { choosing = true }
                    .heightIn(min = 48.dp)
                    .padding(horizontal = 16.dp, vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("Processor", style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f))
                Spacer(Modifier.width(12.dp))
                Text(
                    values.processor.summary(info.accelerator, info.automatic),
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.primary,
                )
            }
            DropdownMenu(expanded = choosing, onDismissRequest = { choosing = false }, modifier = Modifier.background(colors.card)) {
                choices.forEach { processor ->
                    DropdownMenuItem(
                        text = { Text(processor.title, fontWeight = if (processor == values.processor) FontWeight.SemiBold else null) },
                        onClick = {
                            choosing = false
                            if (processor != values.processor) actions.update { it.copy(processor = processor) }
                        },
                    )
                }
            }
        }
        if (values.processor.usesQnn) {
            RowDivider()
            SwitchRow(
                "Keep NPU Awake", values.keepNpuAwake, { on -> actions.update { it.copy(keepNpuAwake = on) } },
                supporting = "Holds the NPU at full speed between frames.",
            )
        }
        RowDivider()
        SwitchRow(
            "Keep CPU Awake", values.keepCpuAwake, { on -> actions.update { it.copy(keepCpuAwake = on) } },
            supporting = "Holds the CPU's clocks up between frames.",
        )
    }
}

@Composable
private fun About(values: SettingsValues, info: SettingsInfo, actions: SettingsActions) {
    val colors = JetlinkTheme.colors
    val context = LocalContext.current
    // seven taps on Version show or hide the Developer section, as Android's build number does
    var taps by remember { mutableIntStateOf(0) }
    val tapVersion = {
        taps += 1
        if (taps >= DEVELOPER_TAPS) {
            taps = 0
            val on = !values.developer
            actions.update { it.copy(developer = on) }
            Toast.makeText(context, if (on) "Developer settings on" else "Developer settings off", Toast.LENGTH_SHORT).show()
        }
    }
    val server = when (info.runState) {
        RunState.Stopped -> "Stopped"
        RunState.Starting -> "Starting"
        RunState.Serving -> "Running"
        is RunState.Failed -> "Failed"
    }
    FormSection("About") {
        ValueRow("Version", info.version, Modifier.clickable(onClick = tapVersion))
        RowDivider()
        ValueRow("Runtime", info.runtime?.let { "onnxruntime $it" } ?: "onnxruntime")
        RowDivider()
        ValueRow("Chip", info.chip)
        RowDivider()
        ValueRow("Server", server, valueColor = if (info.runState is RunState.Failed) colors.bad else null)
        RowDivider()
        if (info.runState == RunState.Stopped) {
            ActionRow("Start Server", actions.startServer, color = MaterialTheme.colorScheme.primary, chevron = false)
        } else {
            // a failed server's service still runs: it starts again, it is not started
            ActionRow("Restart Server", actions.restartServer, color = MaterialTheme.colorScheme.primary, chevron = false)
            RowDivider()
            ActionRow("Stop Server", actions.stopServer, color = colors.bad, chevron = false)
        }
    }
}

private const val DEVELOPER_TAPS = 7

@Preview(showBackground = true, heightDp = 1500)
@Composable
private fun SettingsPreview() {
    JetlinkTheme {
        Column(Modifier.background(JetlinkTheme.colors.grouped)) {
            SettingsContent(
                SettingsValues(developer = true),
                SettingsInfo(
                    snapshot = PreviewData.serving,
                    runState = RunState.Serving,
                    usb = UsbState.Attached,
                    wifi = "192.168.1.23",
                    runtime = "1.29.0",
                    chip = "Snapdragon 8 Gen 3",
                    version = "0.5.0",
                    processors = listOf(Processor.Auto, Processor.Npu, Processor.Gpu, Processor.NpuGpu),
                ),
                SettingsActions(),
            )
        }
    }
}
