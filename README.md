# bereal

Flutter MVP for an image authenticity checker (AI generation, beauty filter, ELA, and tamper scoring).

## Quick Start

1. Run the app:
   ```bash
   flutter run
   ```
2. Enable GitHub Pages:
   - Settings → Pages → Build from `main` branch → `/docs` folder.
3. Ensure the analyzer page is live at:
   - `https://cyleunggg.github.io/bereal/analyzer/index.html`

The Flutter app loads that analyzer URL and communicates via a JavaScript bridge.

Analyzer assets are sourced from the `image-reality-check` project and live under `docs/analyzer/`.
