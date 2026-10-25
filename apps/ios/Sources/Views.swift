import SwiftUI
import UniformTypeIdentifiers

private func bytes(_ value: Double?) -> String {
    guard let value, value.isFinite, value >= 0, value < Double(Int64.max) else { return "Not reported" }
    return ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .binary)
}

private enum Destination: String, CaseIterable {
    case mesh = "Mesh", device = "This device", model = "Model", settings = "Settings"
    var symbol: String {
        switch self { case .mesh: "circle.hexagongrid"; case .device: "ipad"; case .model: "square.stack.3d.up"; case .settings: "slider.horizontal.3" }
    }
}

struct RootView: View {
    @State private var selection = Destination.mesh
    @AppStorage("deviceName") private var deviceName = "My device"
    var body: some View {
        GeometryReader { geometry in
            if geometry.size.width >= 760 {
                HStack(spacing: 0) {
                    sidebar.frame(width: 196)
                    Rectangle().fill(MeshPalette.line).frame(width: 1)
                    NavigationStack { destination(selection) }.frame(maxWidth: .infinity, maxHeight: .infinity)
                }.background(MeshPalette.canvas)
            } else {
                TabView(selection: $selection) {
                    ForEach(Destination.allCases, id: \.self) { item in
                        NavigationStack { destination(item) }.tabItem { Label(item.rawValue, systemImage: item.symbol) }.tag(item)
                    }
                }
            }
        }.tint(MeshPalette.brandAccent)
    }
    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 36) {
            HStack(spacing: 9) {
                Image("BrandCake").resizable().interpolation(.none).scaledToFit().frame(width: 38, height: 38).accessibilityHidden(true)
                Text("layercake").font(.system(size: 20, weight: .bold, design: .rounded))
            }.padding(.top, 24).accessibilityElement(children: .combine)
            VStack(alignment: .leading, spacing: 6) {
                Eyebrow(text: "Workspace").padding(.leading, 12).padding(.bottom, 9)
                ForEach(Destination.allCases, id: \.self) { item in
                    Button { selection = item } label: {
                        HStack(spacing: 12) {
                            Image(systemName: item.symbol).font(.system(size: 18)).frame(width: 23)
                            Text(item.rawValue).font(.subheadline.weight(selection == item ? .semibold : .regular))
                            Spacer(minLength: 0)
                        }.padding(.horizontal, 12).padding(.vertical, 14)
                            .foregroundStyle(selection == item ? MeshPalette.ink : MeshPalette.muted)
                            .background(selection == item ? MeshPalette.surface : Color.clear, in: RoundedRectangle(cornerRadius: 13))
                    }.buttonStyle(.plain).accessibilityAddTraits(selection == item ? .isSelected : [])
                }
            }
            Spacer(minLength: 30)
            VStack(alignment: .leading, spacing: 14) {
                Rectangle().fill(MeshPalette.line).frame(height: 1)
                Eyebrow(text: "On this device")
                Label(deviceName, systemImage: "ipad").font(.subheadline).lineLimit(2)
                Text("Your devices.\nOne shared model.").font(.caption).foregroundStyle(MeshPalette.muted).lineSpacing(4)
            }.padding(.bottom, 24)
        }.padding(.horizontal, 16).background(MeshPalette.sidebar).foregroundStyle(MeshPalette.ink)
    }
    @ViewBuilder private func destination(_ item: Destination) -> some View {
        switch item {
        case .mesh: MeshView(openModel: { selection = .model })
        case .device: DeviceView()
        case .model: ModelView(openSettings: { selection = .settings })
        case .settings: SettingsView()
        }
    }
}

struct MeshView: View {
    @EnvironmentObject private var enrollment: EnrollmentStore
    @EnvironmentObject private var store: MeshStore
    var openModel: () -> Void = {}
    @State private var pairing = false
    @State private var pairingMode = "create"
    private var joined: Bool { enrollment.state?.mesh != nil }
    var body: some View {
        Page {
            HStack {
                VStack(alignment: .leading, spacing: 8) { CakeSprinkles(); Eyebrow(text: "Your workspace") }
                Spacer()
                Label(joined ? "Private mesh" : "Connect anywhere", systemImage: joined ? "lock" : "globe")
                    .font(.caption).foregroundStyle(MeshPalette.muted)
            }
            if joined { joinedHeader } else if enrollment.state?.joining == nil { welcome }
            MembershipSummary()
            if joined {
                ViewThatFits(in: .horizontal) {
                    HStack(alignment: .top, spacing: 18) { invitationCard.frame(minWidth: 225); modelCard.frame(minWidth: 225) }
                    VStack(spacing: 18) { invitationCard; modelCard }
                }
            } else if enrollment.state?.joining == nil {
                gettingStarted
            }
        }
        .toolbar(.hidden, for: .navigationBar)
        .refreshable { await enrollment.refresh() }
        .sheet(isPresented: $pairing) { PairingView(mode: pairingMode) }
    }
    private var joinedHeader: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(enrollment.state?.mesh?.name ?? "Your mesh").font(MeshType.display(42)).fixedSize(horizontal: false, vertical: true)
            Text("A shared space for the devices you trust.").font(.subheadline).foregroundStyle(MeshPalette.muted)
        }.padding(.vertical, 8)
    }
    private var welcome: some View {
        VStack(alignment: .leading, spacing: 26) {
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 20) { welcomeText.frame(minWidth: 270); MeshIllustration().frame(width: 205) }
                VStack(alignment: .leading, spacing: 20) { welcomeText; MeshIllustration().frame(maxWidth: 300).frame(maxWidth: .infinity) }
            }
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 12) { createButton; joinButton }
                VStack(alignment: .leading, spacing: 12) { createButton; joinButton }
            }
        }.padding(.vertical, 18)
    }
    private var welcomeText: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Good things\nwork together.").font(MeshType.display(42)).fixedSize(horizontal: false, vertical: true)
            Text("Give your devices a place to connect.\nStart a mesh, then invite them in.")
                .font(.body).foregroundStyle(MeshPalette.muted).lineSpacing(5).fixedSize(horizontal: false, vertical: true)
        }
    }
    private var createButton: some View {
        Button { pairingMode = "create"; pairing = true } label: { Label("Create a mesh", systemImage: "plus") }.buttonStyle(MeshButtonStyle())
    }
    private var joinButton: some View {
        Button { pairingMode = "join"; pairing = true } label: { Label("Join a mesh", systemImage: "arrow.up.right") }.buttonStyle(MeshButtonStyle(primary: false))
    }
    private var invitationCard: some View {
        Surface {
            VStack(alignment: .leading, spacing: 16) {
                Image(systemName: "person.badge.plus").font(.title2).foregroundStyle(MeshPalette.brandAccent)
                Text("Make room for more.").font(MeshType.display(23))
                Text("One invitation. A shared mesh.\nBring another device into your mesh.").font(.subheadline).foregroundStyle(MeshPalette.muted).lineSpacing(3)
                Button { pairingMode = "create"; pairing = true } label: { Label("Invite a device", systemImage: "plus") }.buttonStyle(MeshButtonStyle())
            }
        }
    }
    private var modelCard: some View {
        Surface {
            VStack(alignment: .leading, spacing: 16) {
                Image(systemName: "square.stack.3d.up").font(.title2).foregroundStyle(MeshPalette.brandAccent)
                Text("One shared model.").font(MeshType.display(23))
                Text(store.snapshot?.model ?? "Membership is ready. Model execution is still being connected.")
                    .font(.subheadline).foregroundStyle(MeshPalette.muted).lineSpacing(3)
                Button(action: openModel) { Label("View model", systemImage: "arrow.up.right") }.buttonStyle(MeshButtonStyle(primary: false))
            }
        }
    }
    private var gettingStarted: some View {
        Surface {
            VStack(alignment: .leading, spacing: 22) {
                Eyebrow(text: "A simple way in")
                setupStep("01", title: "Start with this device", detail: "Create a mesh and give it a name.", icon: "ipad")
                Divider().overlay(MeshPalette.line)
                setupStep("02", title: "Share an emoji code", detail: "Your other devices enter the same eight emojis.", icon: "face.smiling")
                Divider().overlay(MeshPalette.line)
                setupStep("03", title: "Let them in", detail: "Approve the request from any trusted device.", icon: "checkmark.shield")
            }
        }
    }
    private func setupStep(_ number: String, title: String, detail: String, icon: String) -> some View {
        HStack(alignment: .top, spacing: 18) {
            Text(number).font(MeshType.label).foregroundStyle(MeshPalette.muted).padding(.top, 4)
            VStack(alignment: .leading, spacing: 6) { Text(title).font(.subheadline.weight(.semibold)); Text(detail).font(.subheadline).foregroundStyle(MeshPalette.muted) }
            Spacer(minLength: 0)
            Image(systemName: icon).font(.title3).foregroundStyle(MeshPalette.brandAccent).padding(.top, 3).accessibilityHidden(true)
        }
    }
}

struct StageView: View {
    let device: MeshDevice
    let active: Bool
    let connected: Bool
    var body: some View {
        Page {
            PageHeading(eyebrow: "Runtime stage", title: device.name, subtitle: device.hardware)
            Surface {
                VStack(spacing: 4) {
                    DetailRow(label: "Layers", value: device.range)
                    DetailRow(label: "Observation", value: !connected ? "Unavailable" : active ? "Current run" : "Last reported run")
                    DetailRow(label: "Executions", value: "\(device.executions)")
                    DetailRow(label: "Native compute", value: device.computeMs.map { String(format: "%.1f ms", $0) } ?? "Not reported")
                    DetailRow(label: "Observed buffers", value: bytes(device.bufferBytes))
                }
            }
            StatusNote(symbol: "info.circle", text: "Buffer measurements do not include all system or GPU memory.")
        }.navigationBarTitleDisplayMode(.inline)
    }
}

struct DeviceView: View {
    @AppStorage("deviceName") private var name = "My device"
    @AppStorage("contributionMiB") private var budget = 1024
    @AppStorage("powerOnly") private var powerOnly = true
    @State private var checked = Date()
    var body: some View {
        Page {
            PageHeading(eyebrow: "This device", title: "Your part of the mesh.", subtitle: "A familiar name. A little room to contribute.")
            Surface {
                VStack(alignment: .leading, spacing: 20) {
                    MeshField(title: "Device name", value: $name).onChange(of: name) { _, new in if new.count > 48 { name = String(new.prefix(48)) } }
                    Divider()
                    DetailRow(label: "Platform", value: "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)")
                    DetailRow(label: "Physical memory", value: bytes(Double(ProcessInfo.processInfo.physicalMemory)))
                    DetailRow(label: "Processors", value: "\(ProcessInfo.processInfo.processorCount)")
                    DetailRow(label: "Power saving", value: ProcessInfo.processInfo.isLowPowerModeEnabled ? "On" : "Off")
                    DetailRow(label: "Temperature", value: thermalName)
                    Button { checked = Date() } label: { Label("Refresh device details", systemImage: "arrow.clockwise") }.buttonStyle(MeshButtonStyle(primary: false))
                }
            }
            Surface {
                VStack(alignment: .leading, spacing: 20) {
                    Text("Contribution preferences").font(MeshType.display(24))
                    Picker("Memory contribution", selection: $budget) { ForEach([512, 1024, 2048, 4096], id: \.self) { Text("\($0) MiB").tag($0) } }
                    Toggle("Only while plugged in", isOn: $powerOnly)
                    StatusNote(symbol: "pause.circle", text: "Saved for future compute setup. Benchmarking and contribution limits are not active yet.")
                }
            }
        }.toolbar(.hidden, for: .navigationBar)
    }
    private var thermalName: String {
        _ = checked
        switch ProcessInfo.processInfo.thermalState { case .nominal: return "Normal"; case .fair: return "Warm"; case .serious: return "Hot"; case .critical: return "Critical"; @unknown default: return "Unknown" }
    }
}

struct ModelView: View {
    @EnvironmentObject private var store: MeshStore
    var openSettings: () -> Void = {}
    @State private var importing = false
    @State private var removing = false
    @State private var failure = ""
    var body: some View {
        Page {
            PageHeading(eyebrow: "One model", title: "Shared intelligence.", subtitle: "One model for the whole mesh. Each device plays its part.")
            Surface {
                VStack(alignment: .leading, spacing: 20) {
                    Label("LOCAL MODEL FILE", systemImage: "doc").font(MeshType.label).foregroundStyle(MeshPalette.muted)
                    if store.modelName.isEmpty {
                        Text("Bring your model.").font(MeshType.display(28))
                        Text("Choose a GGUF file to keep ready on this device. It stays in its original location.").font(.subheadline).foregroundStyle(MeshPalette.muted)
                        Button { importing = true } label: { Label("Choose a GGUF file", systemImage: "plus") }.buttonStyle(MeshButtonStyle())
                    } else {
                        Text(store.modelName).font(.title3.weight(.medium))
                        DetailRow(label: "File size", value: bytes(Double(store.modelBytes)))
                        Text(store.modelNote).font(.subheadline).foregroundStyle(MeshPalette.muted)
                        Button("Check file access") { store.verifySelectedFile() }.buttonStyle(MeshButtonStyle(primary: false))
                        Button("Remove selection", role: .destructive) { removing = true }
                    }
                    StatusNote(symbol: "pause.circle", text: "Selecting a file does not start inference. Automatic benchmarks and model execution are still being connected.")
                    if !failure.isEmpty { Text(failure).foregroundStyle(MeshPalette.warning).font(.subheadline) }
                }
            }
            runtime
        }.toolbar(.hidden, for: .navigationBar)
        .fileImporter(isPresented: $importing, allowedContentTypes: [.data], allowsMultipleSelection: false) { result in
            do { guard let url = try result.get().first else { return }; try store.selectModel(url); failure = "" } catch { failure = error.localizedDescription }
        }
        .confirmationDialog("Remove this file selection? The original file will stay on disk.", isPresented: $removing, titleVisibility: .visible) { Button("Remove selection", role: .destructive) { store.removeModel() } }
    }
    private var runtime: some View {
        Surface {
            VStack(alignment: .leading, spacing: 18) {
                HStack { Text("Runtime").font(MeshType.display(24)); Spacer(); if store.refreshing { ProgressView() } }
                if let snapshot = store.snapshot {
                    Text(snapshot.model).font(.headline)
                    DetailRow(label: "Status", value: store.connected ? (snapshot.active ? "Running" : "Idle") : "Unavailable")
                    if let rate = snapshot.decodeRate { DetailRow(label: "Latest steps / sec", value: String(format: "%.2f", rate)) }
                    ForEach(snapshot.devices) { device in
                        NavigationLink { StageView(device: device, active: snapshot.active, connected: store.connected) } label: {
                            HStack { VStack(alignment: .leading, spacing: 5) { Text(device.name).font(.subheadline.weight(.medium)); Text("Layers \(device.range)").font(.caption).foregroundStyle(MeshPalette.muted) }; Spacer(); Image(systemName: "chevron.right").font(.caption) }.padding(.vertical, 10)
                        }.buttonStyle(.plain)
                    }
                    StatusNote(symbol: "clock", text: "Idle measurements describe the last reported run. Runtime stages are separate from trusted membership.")
                } else {
                    Text("Connect a dashboard to see model execution and layer assignments.").font(.subheadline).foregroundStyle(MeshPalette.muted)
                    Button(action: openSettings) { Label("Connection settings", systemImage: "arrow.up.right") }.buttonStyle(MeshButtonStyle(primary: false))
                }
                if !store.error.isEmpty { Text(store.error).foregroundStyle(MeshPalette.warning).font(.footnote) }
            }
        }
    }
}

struct SettingsView: View {
    @EnvironmentObject private var store: MeshStore
    @State private var endpoint = ""
    @State private var notice = ""
    var body: some View {
        Page {
            PageHeading(eyebrow: "Settings", title: "Make yourself at home.", subtitle: "Connections and the small details.")
            Surface {
                VStack(alignment: .leading, spacing: 20) {
                    Text("Dashboard connection").font(MeshType.display(24))
                    Text("Optional. Connect your runtime dashboard to see model activity. Mesh invitations work independently.").font(.subheadline).foregroundStyle(MeshPalette.muted)
                    MeshField(title: "Dashboard address", value: $endpoint, placeholder: "https://dashboard.example")
                        .textInputAutocapitalization(.never).autocorrectionDisabled().keyboardType(.URL).accessibilityIdentifier("dashboard-address")
                    Button("Save connection") { do { try store.saveEndpoint(endpoint); notice = "Connection saved." } catch { notice = error.localizedDescription } }.buttonStyle(MeshButtonStyle())
                    if !notice.isEmpty { Text(notice).font(.footnote).foregroundStyle(MeshPalette.muted) }
                    Text("Use the web dashboard address. Leave blank to disconnect.").font(.footnote).foregroundStyle(MeshPalette.muted)
                }
            }
            Surface {
                VStack(alignment: .leading, spacing: 20) {
                    Text("Quiet by default.").font(MeshType.display(24))
                    StatusNote(symbol: "globe", text: "Pair over the Internet with Iroh. A trusted member approves each new device.")
                    StatusNote(symbol: "moon", text: "Discovery and dashboard updates pause in the background.")
                    StatusNote(symbol: "lock", text: "Trust is saved on this device. No prompt history is stored by the app.")
                }
            }
        }.toolbar(.hidden, for: .navigationBar).onAppear { endpoint = store.endpoint }
    }
}
