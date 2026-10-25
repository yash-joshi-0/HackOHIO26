import SwiftUI

enum MeshType {
    static func display(_ size: CGFloat) -> Font { .system(size: size, weight: .bold, design: .rounded) }
    static let label = Font.system(.caption2, design: .monospaced).weight(.medium)
}

struct Eyebrow: View {
    let text: String
    var body: some View { Text(text.uppercased()).font(MeshType.label).tracking(1.8).foregroundStyle(MeshPalette.muted) }
}

struct MeshButtonStyle: ButtonStyle {
    var primary = true
    @Environment(\.isEnabled) private var enabled
    func makeBody(configuration: Configuration) -> some View {
        configuration.label.font(.subheadline.weight(.semibold))
            .padding(.horizontal, 20).padding(.vertical, 14).frame(minHeight: 48)
            .foregroundStyle(primary ? MeshPalette.onAccent : MeshPalette.ink)
            .background(primary ? MeshPalette.accent : MeshPalette.surface, in: RoundedRectangle(cornerRadius: 14))
            .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(primary ? Color.clear : MeshPalette.line, lineWidth: 1))
            .opacity(enabled ? (configuration.isPressed ? 0.75 : 1) : 0.45)
    }
}

struct Surface<Content: View>: View {
    @ViewBuilder let content: Content
    var body: some View {
        content.padding(24).frame(maxWidth: .infinity, alignment: .leading)
            .background(MeshPalette.surface, in: RoundedRectangle(cornerRadius: 22))
            .overlay(RoundedRectangle(cornerRadius: 22).strokeBorder(MeshPalette.line.opacity(0.7), lineWidth: 1))
    }
}

struct Page<Content: View>: View {
    @Environment(\.horizontalSizeClass) private var sizeClass
    @ViewBuilder let content: Content
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) { content }
                .frame(maxWidth: 1040, alignment: .leading)
                .padding(sizeClass == .regular ? 32 : 22)
                .frame(maxWidth: .infinity, alignment: .topLeading)
        }.background(MeshPalette.canvas).foregroundStyle(MeshPalette.ink)
    }
}

struct PageHeading: View {
    let eyebrow: String
    let title: String
    let subtitle: String
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            CakeSprinkles()
            Eyebrow(text: eyebrow)
            Text(title).font(MeshType.display(36)).fixedSize(horizontal: false, vertical: true)
            if !subtitle.isEmpty { Text(subtitle).font(.subheadline).foregroundStyle(MeshPalette.muted).fixedSize(horizontal: false, vertical: true) }
        }
    }
}

struct DetailRow: View {
    let label: String
    let value: String
    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .firstTextBaseline) { Text(label).foregroundStyle(MeshPalette.muted); Spacer(minLength: 20); Text(value).multilineTextAlignment(.trailing) }
            VStack(alignment: .leading, spacing: 6) { Text(label).foregroundStyle(MeshPalette.muted); Text(value) }
        }.font(.subheadline).padding(.vertical, 8)
    }
}

struct StatusNote: View {
    let symbol: String
    let text: String
    var body: some View {
        Label { Text(text).fixedSize(horizontal: false, vertical: true) } icon: { Image(systemName: symbol) }
            .font(.footnote).foregroundStyle(MeshPalette.muted)
    }
}

struct MeshField: View {
    let title: String
    @Binding var value: String
    var placeholder = ""
    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            Text(title).font(.subheadline.weight(.medium))
            TextField(placeholder.isEmpty ? title : placeholder, text: $value)
                .padding(15).background(MeshPalette.surface, in: RoundedRectangle(cornerRadius: 12))
                .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(MeshPalette.line, lineWidth: 1))
        }
    }
}

/// Decorative illustration, not a topology or a live connectivity indicator.
struct MeshIllustration: View {
    var body: some View {
        GeometryReader { geometry in
            let w = geometry.size.width
            let h = geometry.size.height
            let points = [CGPoint(x: w * 0.5, y: h * 0.25), CGPoint(x: w * 0.18, y: h * 0.75), CGPoint(x: w * 0.82, y: h * 0.75)]
            Canvas { context, _ in
                for x in stride(from: 12.0, to: w, by: 18) {
                    for y in stride(from: 8.0, to: h, by: 18) {
                        context.fill(Path(ellipseIn: CGRect(x: x, y: y, width: 1.5, height: 1.5)), with: .color(MeshPalette.line))
                    }
                }
                var path = Path(); path.move(to: points[0]); path.addLine(to: points[1]); path.addLine(to: points[2]); path.closeSubpath()
                context.stroke(path, with: .color(MeshPalette.brandAccent.opacity(0.35)), style: StrokeStyle(lineWidth: 1, dash: [4, 5]))
            }
            Image("BrandCake").resizable().interpolation(.none).scaledToFit()
                .frame(width: 84, height: 84).position(x: w * 0.5, y: h * 0.58)
            ForEach(0..<3) { index in
                Image(systemName: ["desktopcomputer", "ipad", "iphone"][index])
                    .font(.system(size: 25, weight: .light)).foregroundStyle(MeshPalette.brandAccent)
                    .frame(width: 64, height: 64).background(MeshPalette.surface, in: RoundedRectangle(cornerRadius: 20))
                    .overlay(RoundedRectangle(cornerRadius: 20).strokeBorder(MeshPalette.line, lineWidth: 1))
                    .position(points[index])
            }
        }.frame(height: 180).accessibilityHidden(true)
    }
}

/// The same restrained strawberry/cocoa decoration as the web header.
struct CakeSprinkles: View {
    var body: some View {
        HStack(spacing: 8) {
            ForEach(0..<5) { index in
                Rectangle().fill(index.isMultiple(of: 2) ? MeshPalette.strawberry : MeshPalette.brandAccent.opacity(0.45))
                    .frame(width: index == 2 ? 18 : 5, height: 4)
                    .offset(y: index == 1 ? 3 : 0)
            }
        }.frame(height: 10).accessibilityHidden(true)
    }
}
