import SwiftUI

@main
struct MeshThingProbeApp: App {
    @State private var status = "Preparing native shard test…"

    var body: some Scene {
        WindowGroup {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("MeshThing native probe").font(.title)
                    Text("Local GLM shard correctness and context reset. This test does not connect to Iroh.")
                    Text(status).font(.system(.body, design: .monospaced)).textSelection(.enabled)
                }.padding()
            }.task { runProbe() }
        }
    }

    private func runProbe() {
        guard let model = Bundle.main.path(forResource: "GLM-4.7-Flash-two-layer-fixture", ofType: "gguf") else {
            status = "Missing test fixture"
            return
        }
        let report = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("native-shard-result.json")
        status = "Running on \(ProcessInfo.processInfo.physicalMemory) bytes of physical RAM…"
        DispatchQueue.global(qos: .userInitiated).async {
            let code = model.withCString { modelPath in
                report.path.withCString { reportPath in
                    meshthing_native_smoke(modelPath, reportPath)
                }
            }
            let text = (try? String(contentsOf: report, encoding: .utf8)) ?? "No result file; exit \(code)"
            print("MESHTHING_PROBE_RESULT: \(text)")
            DispatchQueue.main.async { status = text }
        }
    }
}
