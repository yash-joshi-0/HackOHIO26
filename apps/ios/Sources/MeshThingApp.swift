import SwiftUI

@main struct MeshThingApp: App {
    @StateObject private var store = MeshStore()
    @StateObject private var enrollment = EnrollmentStore()
    @Environment(\.scenePhase) private var phase
    var body: some Scene {
        WindowGroup {
            RootView().environmentObject(store).environmentObject(enrollment)
                .tint(MeshPalette.brandAccent)
                .task(id: phase) {
                    guard phase == .active else { await enrollment.stop(); return }
                    await enrollment.start()
                    while !Task.isCancelled {
                        await enrollment.refresh()
                        do { try await Task.sleep(for: .seconds(3)) } catch { return }
                    }
                }
                .task(id: "\(phase)-\(store.endpoint)") {
                    guard phase == .active else { return }
                    while !Task.isCancelled {
                        await store.refresh()
                        do { try await Task.sleep(for: .seconds(5)) } catch { return }
                    }
                }
        }
    }
}
