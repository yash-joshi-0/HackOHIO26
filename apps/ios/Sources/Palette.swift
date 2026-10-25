import SwiftUI
import UIKit

/// Keep custom surfaces and their foregrounds in the same appearance.
/// Forms and lists continue to use the platform's semantic label colors.
enum MeshPalette {
    private static func adaptive(light: UInt32, dark: UInt32) -> Color {
        Color(uiColor: UIColor { traits in
            let rgb = traits.userInterfaceStyle == .dark ? dark : light
            return UIColor(
                red: CGFloat((rgb >> 16) & 255) / 255,
                green: CGFloat((rgb >> 8) & 255) / 255,
                blue: CGFloat(rgb & 255) / 255,
                alpha: 1
            )
        })
    }

    static let strawberry = adaptive(light: 0xBD525B, dark: 0xF49DA4)
    static let canvas = adaptive(light: 0xFAF9F7, dark: 0x211B19)
    static let surface = adaptive(light: 0xFFFFFF, dark: 0x2D2421)
    static let hero = adaptive(light: 0xEAE1D9, dark: 0x3B2C28)
    /// Primary action (button fill). Matches the web "Primary action" token.
    static let accent = adaptive(light: 0x784B3E, dark: 0xEDB7A7)
    /// Focus rings, tint, and status-label accents. Matches the web "Brand accent" token.
    static let brandAccent = adaptive(light: 0x925648, dark: 0xEDB7A7)
    static let onAccent = adaptive(light: 0xFFFFFF, dark: 0x35201C)
    static let warning = adaptive(light: 0x856528, dark: 0xFFD08A)
    static let ink = adaptive(light: 0x292725, dark: 0xFFF3E7)
    static let muted = adaptive(light: 0x706C66, dark: 0xCCB7A9)
    static let line = adaptive(light: 0xE6E3DE, dark: 0x574139)
    static let sidebar = adaptive(light: 0xF5F3EF, dark: 0x261E1B)
}
