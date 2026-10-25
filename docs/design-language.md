# Layercake design language

Layercake is a tool for understanding a distributed model: which devices participate, which layers they run, and where time goes. The interface should make that information easy to scan and compare.

Use a restrained, warm monitoring interface. The Minecraft cake provides a small, recognizable brand accent. Keep the working area calm, readable, and focused on measurements.

## Information hierarchy

1. Name the current view plainly: Overview, Devices, Model distribution, Activity, Settings.
2. Show a compact summary of configured devices, decode rate, and reported stage buffers.
3. Show the actual configured execution order, with device names and layer ranges.
4. Make device measurements visible in aligned rows; support filtering and sorting by native compute time.
5. Put secondary details in a device dialog. Keep measurement definitions close to their values.

Use real telemetry. Keep missing values as an em dash. Distinguish a connected runtime API from an online worker, and distinguish the latest observation from current activity. Native compute time excludes network transfer and waiting; buffers are reported allocations, not total RAM or free memory. Sorting compute times helps compare the latest measurements but does not prove the source of a network stall.

## Color

| Role | Value | Use |
| --- | --- | --- |
| Main background | `#faf9f7` | Warm neutral canvas |
| Surface | `#ffffff` | Panels and top bar |
| Sidebar | `#f5f3ef` | Quiet navigation surface |
| Primary text | `#292725` | Headings and important values |
| Secondary text | `#706c66` | Descriptions and measurement definitions |
| Dividers | `#e6e3de` | Panel outlines and row separators |
| Brand accent | `#925648` | Focus rings and selected mobile navigation |
| Primary action | `#784b3e` | Supported primary actions |
| Selected navigation | `#eae1d9` / `#713d31` | Background / foreground |
| Connected | `#4e8061` | Runtime connection indicator |
| Active compute | `#34714b` | Computing state, accompanied by text |
| Failure | `#a14336` | Failed run, accompanied by text |
| Warning | `#856528` | Disconnected or warning state |
| Compute comparison | `#a77b61` | Small bars on a neutral track |

Device/layer identity colors, in route order: `#42685c`, `#89602f`, `#526a8d`, `#865567`, `#625d8d`. Keep a device's color stable when filtering or sorting. These identify stages; they do not indicate health. Always pair color with names, labels, or values.

## Typography

- **DM Sans Variable:** interface text, labels, navigation, and descriptions.
- **Manrope Variable:** brand, page headings, and summary values.
- **System monospace:** layer ranges, compute timings, buffer values, and endpoint addresses. Use tabular numerals for comparisons.
- Desktop page title: 27px, weight 650, tight tracking; mobile: 24px.
- Panel titles: 14px, weight 650.
- Body and controls: 12–14px. Secondary labels: 11–12px. Reserve 10px for small supporting annotations.
- Summary values: 29px desktop, 24px mobile. Units are smaller, quieter, and explicitly labeled.

Fonts are bundled locally. Do not add remote font dependencies for this theme.

## Layout and components

- Desktop: 216px fixed sidebar, 62px top bar, main area with 36px horizontal padding and a maximum width of 1450px.
- Use consistent 16–24px internal spacing. Separate major panels by 24px.
- Panels use a 1px neutral border and a 7px corner radius. Buttons use a 5px radius. Prefer dividers to shadows.
- Keep the summary as one divided strip rather than many floating cards.
- Execution route stages are small, numbered, clickable blocks connected by simple arrows. Render configured route data, not a decorative mesh illustration.
- Device rows align name/hardware, state, layers, latest native compute, and buffers. Small comparison bars supplement the numeric timings.
- Keep actions specific: Refresh, Add device, Save preferences, Close. Use a stronger button treatment only for a supported primary action.
- Device details use a native dialog with an explicit close control and Escape dismissal.
- Unavailable functionality gets a short explanation. Do not present sample pairing codes or input controls that imply a working enrollment flow.

## Cake branding

Use the exact supplied asset: [Minecraft Wiki Cake_JE4.png](https://minecraft.wiki/images/Cake_JE4.png?7c59e), stored at `apps/web/static/layercake-logo.png`.

Display the small pixelated cake beside **Layercake**, vertically centered with the wordmark. The source PNG has uneven transparent padding: its visible alpha bounds are approximately `(33, 92)` to `(267, 290)` on a 300×300 canvas. Center the painted cake rather than the image canvas. The current 37px image uses `translate(-50%, -63.6667%)` within a 29×28px wrapper. Preserve this correction when scaling the mark.

Use a short, thin cake-colored accent at the top. Keep cake decorations out of data rows. Avoid oversized cakes, repeated mascots, frosting backgrounds, gradients, decorative orbit diagrams, and promotional hero banners.

## Responsive behavior

- At 760px and below, navigation becomes a compact horizontal row above the content.
- Keep device names, state, layers, compute, and buffers visible. Device rows reflow into compact cards with labels; important timings must not require horizontal scrolling.
- The ordered route can scroll horizontally within its panel. Keep the page itself within the viewport.
- Layer labels stay on one line; the legend wraps cleanly.
- Verify at 320px, 390px, and desktop widths. Preserve clear focus states and usable control sizes.

## Copy and accessibility

Use direct descriptions of the data. Prefer “Devices,” “Execution route,” and “Layer allocation” over slogans. Avoid exaggerated claims about capacity, speed, availability, or completed integration.

Use labeled controls, visible keyboard focus, a skip-to-content link, and text alongside status colors. Respect reduced-motion preferences. Loading, disconnected, missing telemetry, and no matching devices are distinct states with useful explanations. Retained observations must remain clearly labeled when refresh fails.

## Platforms

This color table is canonical for iOS and Android as well as web — light mode only; the mobile apps' existing dark-mode palettes are a mobile-only extension with no web equivalent and are not governed by this table. Typography stays per platform: web uses DM Sans Variable and Manrope Variable as specified above; iOS and Android use their platform's system sans font at a bold weight for display text — no serif anywhere in the brand, and no bundled custom font files on mobile. Layout and component shape (sidebar vs. bottom navigation/tabs, corner radii, dialog vs. sheet) stays platform-native; only the color tokens and the content inside overlays are shared.

Overlays and sheets on every platform show one focused task at a time. Do not concatenate unrelated sections (for example invitation creation, the full trusted-members list, and pending-approval decisions) into a single scrolling overlay — if content is already shown on the screen behind the overlay, it does not need to repeat inside it.

## Validation and implementation

The implementation lives in `apps/web/src/app.css` and `apps/web/src/routes/+page.svelte` (web), `apps/ios/Sources/Palette.swift` and `apps/ios/Sources/Design.swift` (iOS), and the `LayercakeTheme` in `apps/android/app/src/main/java/org/meshthing/mobile/MainActivity.kt` (Android). This document records the design intent; keep it aligned with future changes.

Before shipping a UI change:

- Run the web Svelte check, production build, existing unit tests, and applicable launcher integration test.
- Inspect desktop and phone screenshots in Helium.
- Exercise device filtering, compute sorting, dialogs, navigation, and disconnected/unknown states.
- Use clearly identified synthetic fixtures for visual testing when measurements are unavailable. Never display fixture results as live runtime evidence.

## References

[SiteScope on Dribbble](https://dribbble.com/shots/26790985-Web-Monitoring-Dashboard-UI-SiteScope) informed compact summaries and aligned service rows. [Mobbin's dashboard examples](https://mobbin.com/blog/what-good-dashboard-design-looks-like) informed hierarchy and grouping. Treat references as layout inspiration; no screenshot assets are bundled.
