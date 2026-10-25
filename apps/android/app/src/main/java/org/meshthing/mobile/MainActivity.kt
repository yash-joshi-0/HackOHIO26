package org.meshthing.mobile

import android.app.ActivityManager
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.content.Context
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.res.painterResource
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.Alignment
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.withContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.CancellationException
import java.text.DateFormat
import java.util.Date

/** Status-label/tint accent. Matches the web "Brand accent" token; distinct from colorScheme.primary, which drives button fills. */
private val BrandAccent: Color @Composable get() = if (isSystemInDarkTheme()) Color(0xFFEDB7A7) else Color(0xFF925648)
private val Moss: Color @Composable get() = BrandAccent
private val Sage: Color @Composable get() = MaterialTheme.colorScheme.secondaryContainer
private val Canvas: Color @Composable get() = MaterialTheme.colorScheme.background

@Composable private fun LayercakeTheme(content: @Composable () -> Unit) {
    val colors = if (isSystemInDarkTheme()) darkColorScheme(
        primary = Color(0xFFEDB7A7), onPrimary = Color(0xFF35201C),
        secondaryContainer = Color(0xFF3B2C28), onSecondaryContainer = Color(0xFFFFF3E7),
        background = Color(0xFF211B19), surface = Color(0xFF2D2421),
        onBackground = Color(0xFFFFF3E7), onSurface = Color(0xFFFFF3E7),
        onSurfaceVariant = Color(0xFFCCB7A9), surfaceVariant = Color(0xFF3B2C28), outlineVariant = Color(0xFF574139)
    ) else lightColorScheme(
        primary = Color(0xFF784B3E), onPrimary = Color.White,
        secondaryContainer = Color(0xFFEAE1D9), onSecondaryContainer = Color(0xFF713D31),
        background = Color(0xFFFAF9F7), surface = Color(0xFFFFFFFF),
        onBackground = Color(0xFF292725), onSurface = Color(0xFF292725),
        onSurfaceVariant = Color(0xFF706C66), surfaceVariant = Color(0xFFF5F3EF), outlineVariant = Color(0xFFE6E3DE)
    )
    val typography = Typography(
        headlineLarge = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.Bold, fontSize = 34.sp, lineHeight = 40.sp),
        headlineMedium = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.Bold, fontSize = 28.sp, lineHeight = 34.sp),
        titleLarge = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.Bold, fontSize = 24.sp, lineHeight = 30.sp)
    )
    MaterialTheme(colorScheme = colors, typography = typography, content = content)
}

@Composable fun CakeSprinkles() {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(vertical = 6.dp)) {
        repeat(5) { index -> Box(Modifier.width(if (index == 2) 18.dp else 5.dp).height(4.dp).background(if (index % 2 == 0) Color(0xFFBD525B) else BrandAccent.copy(alpha = 0.45f))) }
    }
}
private fun bytes(value: Double?) = value?.let { "%.1f GiB".format(it / 1_073_741_824) } ?: "Not reported"

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val store = ViewModelProvider(this)[MeshStore::class.java]
        setContent { LayercakeTheme { MeshApp(store) } }
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.RESUMED) {
                try {
                    store.enrollment.start()
                    while (isActive) { store.enrollment.refresh(); store.refresh(); delay(3000) }
                } finally { withContext(NonCancellable) { store.enrollment.stop() } }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable fun MeshApp(store: MeshStore) {
    var tab by rememberSaveable { mutableIntStateOf(0) }
    var pairing by rememberSaveable { mutableStateOf(false) }
    var startWithJoin by rememberSaveable { mutableStateOf(false) }
    var selected by remember { mutableStateOf<MeshDevice?>(null) }
    val titles = listOf("Mesh", "This device", "Model", "Settings")
    val icons = listOf(Icons.Default.Home, Icons.Default.Person, Icons.Default.Star, Icons.Default.Settings)
    Scaffold(
        topBar = { TopAppBar(title = { Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) { if (tab == 0) Image(painterResource(R.drawable.layercake), contentDescription = null, modifier = Modifier.size(38.dp)); Text(if (tab == 0) "layercake" else titles[tab], fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.Bold) } }) },
        bottomBar = { NavigationBar(containerColor = Canvas) { titles.forEachIndexed { index, title -> NavigationBarItem(selected = tab == index, onClick = { tab = index }, icon = { Icon(icons[index], contentDescription = null) }, label = { Text(title) }) } } }
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize().background(Canvas).verticalScroll(rememberScrollState()).padding(20.dp), verticalArrangement = Arrangement.spacedBy(18.dp)) {
            when (tab) {
                0 -> MeshScreen(store, { joining -> startWithJoin = joining; pairing = true }, { selected = it })
                1 -> DeviceScreen(store)
                2 -> ModelScreen(store)
                3 -> SettingsScreen(store)
            }
        }
    }
    if (pairing) PairingDialog(store, startWithJoin) { pairing = false }
    selected?.let { device ->
        AlertDialog(onDismissRequest = { selected = null }, title = {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Icon(Icons.Default.Info, contentDescription = null, tint = BrandAccent)
                Column { Text(device.name, fontWeight = FontWeight.SemiBold); Text(device.hardware, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
            }
        }, text = {
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                DetailRow("Layers", device.range)
                DetailRow("Observation", "${if (!store.connected) "Unavailable" else if (store.snapshot?.active == true) "Current run" else "Last reported run"} · ${device.executions} executions")
                DetailRow("Native compute", device.computeMs?.let { "%.1f ms".format(it) } ?: "Not reported")
                DetailRow("Observed buffers", bytes(device.bufferBytes))
                Text("Buffers do not describe total memory. Trust and model benchmark eligibility are not reported.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 8.dp))
            }
        }, confirmButton = { TextButton(onClick = { selected = null }) { Text("Done") } })
    }
}

@Composable private fun DetailRow(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.padding(start = 12.dp))
    }
}

@Composable private fun MeshScreen(store: MeshStore, pair: (Boolean) -> Unit, detail: (MeshDevice) -> Unit) {
    val scope = rememberCoroutineScope()
    Text(store.enrollment.state?.meshName?.let { "● Joined $it" } ?: if (store.enrollment.state?.joiningName != null) "Waiting for approval" else "Create or join your mesh", color = Moss, style = MaterialTheme.typography.labelMedium)
    Card(colors = CardDefaults.cardColors(containerColor = Sage), shape = RoundedCornerShape(24.dp)) {
        Column(Modifier.fillMaxWidth().padding(24.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            CakeSprinkles()
            Text("YOUR DEVICES. ONE MODEL.", style = MaterialTheme.typography.labelSmall)
            Text(if (store.enrollment.state?.meshName == null) "A little from each.\nMore together." else "Your mesh,\na slice from each.", style = MaterialTheme.typography.headlineLarge)
            Text("Welcome, ${store.deviceName}. Give your devices a shared purpose.", style = MaterialTheme.typography.bodyMedium)
            Button(onClick = { pair(false) }, enabled = store.enrollment.state?.joiningName == null) { Icon(Icons.Default.Add, null); Spacer(Modifier.width(7.dp)); Text(if (store.enrollment.state?.meshName == null) "Create a mesh" else "Invite a device") }
            if (store.enrollment.state?.meshName == null && store.enrollment.state?.joiningName == null) OutlinedButton(onClick = { pair(true) }) { Text("Join a mesh") }
        }
    }
    MembershipSummary(store.enrollment)
    InfoCard("One shared model", "Your trusted devices form the mesh. Model execution is still being connected.")
    if (store.snapshot?.devices?.isNotEmpty() == true) {
    if (store.error.isNotEmpty()) Text(store.error, color = MaterialTheme.colorScheme.error)
    if (store.snapshot?.connected == false) Text(store.snapshot?.message ?: "Runtime unavailable")
    Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Metric("Configured stages", "${store.snapshot?.devices?.size ?: 0}", Modifier.weight(1f))
        Metric("Latest steps / sec", store.snapshot?.rate?.let { "%.2f".format(it) } ?: "—", Modifier.weight(1f))
    }
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text("Runtime stages", style = MaterialTheme.typography.titleLarge, modifier = Modifier.weight(1f))
        IconButton(onClick = { scope.launch { store.refresh() } }, enabled = store.endpoint.isNotEmpty() && !store.refreshing) { Icon(Icons.Default.Refresh, "Refresh mesh") }
    }
    if (store.snapshot?.devices.isNullOrEmpty()) InfoCard("No runtime connected", "Connect a runtime dashboard in Settings to view model execution. Your mesh membership is shown above.")
    store.snapshot?.devices?.forEach { device ->
        Card(onClick = { detail(device) }, colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
            Column(Modifier.fillMaxWidth().padding(18.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text(device.name, fontWeight = FontWeight.SemiBold)
                Text(device.hardware, style = MaterialTheme.typography.bodySmall)
                Text("Layers ${device.range} · ${if (!store.connected) "Unavailable" else if (store.snapshot?.active == true) "Current run" else "Last reported"}", style = MaterialTheme.typography.labelMedium, color = Moss)
            }
        }
    }
    Text("Configured stages are not a liveness or trust check. Idle measurements describe the last reported run.", style = MaterialTheme.typography.bodySmall)
    }
}

@Composable private fun Metric(title: String, value: String, modifier: Modifier) {
    Card(modifier, colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) { Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) { Text(title, style = MaterialTheme.typography.labelMedium); Text(value, style = MaterialTheme.typography.headlineMedium) } }
}
@Composable private fun InfoCard(title: String, message: String) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) { Column(Modifier.fillMaxWidth().padding(20.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) { Text(title, style = MaterialTheme.typography.titleMedium); Text(message, style = MaterialTheme.typography.bodyMedium) } }
}

@Composable private fun DeviceScreen(store: MeshStore) {
    val context = LocalContext.current
    var checked by remember { mutableStateOf(Date()) }
    var name by rememberSaveable { mutableStateOf(store.deviceName) }
    var budget by rememberSaveable { mutableIntStateOf(store.budgetMiB) }
    var power by rememberSaveable { mutableStateOf(store.powerOnly) }
    var notice by rememberSaveable { mutableStateOf("") }
    val memory = remember(checked) { ActivityManager.MemoryInfo().also { (context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(it) } }
    val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager
    InfoCard("${Build.MANUFACTURER} ${Build.MODEL}", "Android ${Build.VERSION.RELEASE}\nPhysical memory: ${bytes(memory.totalMem.toDouble())}\nAvailable system memory: ${bytes(memory.availMem.toDouble())}\nProcessors: ${Runtime.getRuntime().availableProcessors()}\nPower saving: ${if (powerManager.isPowerSaveMode) "On" else "Off"}")
    Text("Available system memory is not an app memory budget.", style = MaterialTheme.typography.bodySmall)
    OutlinedButton(onClick = { checked = Date() }) { Text("Refresh device check") }
    Text("Checked ${DateFormat.getTimeInstance().format(checked)}", style = MaterialTheme.typography.labelSmall)
    OutlinedTextField(value = name, onValueChange = { if (it.length <= 48) name = it }, label = { Text("Device name") }, modifier = Modifier.fillMaxWidth(), singleLine = true)
    Text("Memory contribution preference", style = MaterialTheme.typography.titleSmall)
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(5.dp)) { listOf(512, 1024, 2048, 4096).forEach { value -> FilterChip(selected = budget == value, onClick = { budget = value }, label = { Text(if (value == 512) "512M" else "${value / 1024}G") }) } }
    Row(verticalAlignment = Alignment.CenterVertically) { Text("Only while plugged in", Modifier.weight(1f)); Switch(checked = power, onCheckedChange = { power = it }) }
    Button(onClick = { try { store.saveDevice(name, budget, power); notice = "Preferences saved on this device." } catch (e: Exception) { notice = e.message.orEmpty() } }) { Text("Save preferences") }
    if (notice.isNotEmpty()) Text(notice)
    InfoCard("Compute integration pending", "These preferences are not yet enforced by a runtime. Model-specific benchmarks and a managed native worker are required before this device can contribute.")
}

@Composable private fun ModelScreen(store: MeshStore) {
    val scope = rememberCoroutineScope()
    var error by remember { mutableStateOf("") }
    var removing by remember { mutableStateOf(false) }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) scope.launch { try { store.selectModel(uri); error = "" } catch (cancelled: CancellationException) { throw cancelled } catch (e: Exception) { error = e.message ?: "Cannot read this file." } }
    }
    InfoCard("One model", store.snapshot?.model ?: "Connect your dashboard to see the mesh model.")
    Text("Local model file", style = MaterialTheme.typography.titleLarge)
    if (store.modelName.isEmpty()) {
        Button(onClick = { picker.launch(arrayOf("*/*")) }, enabled = !store.selectingModel) { Text(if (store.selectingModel) "Checking file…" else "Choose a GGUF file") }
    } else {
        InfoCard(store.modelName, "${if (store.modelBytes > 0) bytes(store.modelBytes.toDouble()) else "Size not reported"}\n${store.modelNote}")
        OutlinedButton(onClick = { scope.launch { store.checkModel() } }) { Text("Check file access") }
        TextButton(onClick = { removing = true }) { Text("Remove selection", color = MaterialTheme.colorScheme.error) }
    }
    if (error.isNotEmpty()) Text(error, color = MaterialTheme.colorScheme.error)
    Text("The file stays in its original location. No weights are copied, uploaded, loaded or shared. Full GGUF validation, download and placement need the native core.", style = MaterialTheme.typography.bodyMedium)
    if (removing) AlertDialog(onDismissRequest = { removing = false }, title = { Text("Remove this selection?") }, text = { Text("The original file stays on disk.") }, confirmButton = { TextButton(onClick = { store.removeModel(); removing = false }) { Text("Remove selection") } }, dismissButton = { TextButton(onClick = { removing = false }) { Text("Cancel") } })
}

@Composable private fun SettingsScreen(store: MeshStore) {
    var endpoint by rememberSaveable { mutableStateOf(store.endpoint) }
    var notice by rememberSaveable { mutableStateOf("") }
    val scope = rememberCoroutineScope()
    Text("Dashboard connection", style = MaterialTheme.typography.titleLarge)
    OutlinedTextField(value = endpoint, onValueChange = { endpoint = it }, label = { Text("Dashboard address") }, placeholder = { Text("https://dashboard.example") }, singleLine = true, modifier = Modifier.fillMaxWidth())
    Button(onClick = { try { store.saveEndpoint(endpoint); notice = "Connection saved."; scope.launch { store.refresh() } } catch (e: Exception) { notice = e.message.orEmpty() } }) { Text("Save connection") }
    if (notice.isNotEmpty()) Text(notice)
    Text("Use the web dashboard address, not the raw core port. Reads /api/mesh. Leave empty to disconnect. Development uses http://127.0.0.1:3210 after an adb port forward.")
    InfoCard("Private by default", "Polling pauses in the background. No prompt history is stored. Dashboard access does not join a mesh; membership is managed on the Mesh tab. Compute integration is still pending.")
}
