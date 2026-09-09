# Media Partners — Synthetic Demo v5

Deploy this version to Streamlit Community Cloud.

Entrypoint:
`streamlit_app.py`

In Streamlit App settings → Secrets add:

```toml
NEON_DATABASE_URL = "YOUR_SYNTHETIC_DEMO_NEON_BRANCH_CONNECTION_STRING"
```

Use the synthetic demo Neon branch, not main/production.

This v5 includes:
- Overview with website views and social reach
- Story Journeys table-first navigation
- story details and cross-partner chronology
- bubble/timeline visualization weighted by reach/views
- topics, people and entities for each story
- citation details enriched with performance/context
- website ↔ Telegram/Facebook/Instagram links
- performance metrics and metric snapshots
