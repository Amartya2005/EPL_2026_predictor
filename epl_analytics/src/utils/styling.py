"""
Shared design system: 'Pitch Dossier' -- a match-report aesthetic (deep
turf green, chalk white, gold tier markings, IBM Plex Mono for data)
carried over from the project's original static dashboard.html so the
Streamlit app and that file share one visual identity rather than two.

Fonts: Oswald (display / headers -- terrace scoreboard character),
Inter (body), IBM Plex Mono (all numeric data -- stat sheets, tables).
"""
from __future__ import annotations

import streamlit as st

DARK = {
    "bg_deep": "#0E1712", "bg_panel": "#16221B", "bg_panel_2": "#1C2A21",
    "chalk": "#EDE9DC", "chalk_dim": "#C6C2B3", "sage": "#8FA290", "sage_dim": "#5E6E60",
    "gold": "#C7A24C", "gold_dim": "#4A3E22", "clay": "#B65C3D", "clay_dim": "#3A241C",
    "blue": "#4C7A93", "blue_dim": "#1E3038", "line": "#2A362D",
}
LIGHT = {
    "bg_deep": "#F4F1E8", "bg_panel": "#FFFFFF", "bg_panel_2": "#ECE7D8",
    "chalk": "#1B2A20", "chalk_dim": "#3E4A40", "sage": "#3F6B47", "sage_dim": "#6F8574",
    "gold": "#8A6A1E", "gold_dim": "#F1E4BE", "clay": "#96422A", "clay_dim": "#F1DCD2",
    "blue": "#2E5A72", "blue_dim": "#DCE9EE", "line": "#DCD6C3",
}

TIER_COLOR = {"Elite": "gold", "Good": "blue", "Average": "sage", "Weak": "clay"}


def _tokens() -> dict:
    theme = st.session_state.get("theme", "dark")
    return DARK if theme == "dark" else LIGHT


def inject_css() -> None:
    t = _tokens()
    st.markdown(f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
      :root {{
        --bg-deep:{t['bg_deep']}; --bg-panel:{t['bg_panel']}; --bg-panel-2:{t['bg_panel_2']};
        --chalk:{t['chalk']}; --chalk-dim:{t['chalk_dim']}; --sage:{t['sage']}; --sage-dim:{t['sage_dim']};
        --gold:{t['gold']}; --gold-dim:{t['gold_dim']}; --clay:{t['clay']}; --clay-dim:{t['clay_dim']};
        --blue:{t['blue']}; --blue-dim:{t['blue_dim']}; --line:{t['line']};
      }}
      .stApp {{ background: var(--bg-deep); color: var(--chalk); }}
      html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
      h1, h2, h3 {{ font-family: 'Oswald', sans-serif !important; letter-spacing: -0.01em; color: var(--chalk); }}
      .eyebrow {{
        font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.14em;
        text-transform: uppercase; color: var(--sage); margin-bottom: 4px;
      }}
      section[data-testid="stSidebar"] {{ background: var(--bg-panel); border-right: 1px solid var(--line); }}
      div[data-testid="stMetric"] {{
        background: var(--bg-panel); border: 1px solid var(--line); border-radius: 8px;
        padding: 14px 16px 10px;
      }}
      div[data-testid="stMetric"] label {{
        font-family: 'IBM Plex Mono', monospace !important; font-size: 10.5px !important;
        text-transform: uppercase; letter-spacing: 0.06em; color: var(--sage) !important;
      }}
      div[data-testid="stMetricValue"] {{ font-family: 'Oswald', sans-serif !important; color: var(--chalk) !important; }}
      div[data-testid="stExpander"] {{ border: 1px solid var(--line) !important; border-radius: 8px !important; background: var(--bg-panel); }}
      .pitch-card {{
        background: var(--bg-panel); border: 1px solid var(--line); border-radius: 8px;
        padding: 16px 18px; margin-bottom: 10px;
      }}
      .tier-pill {{ display:inline-block; padding:3px 10px; border-radius:4px; font-size:12px; font-family:'IBM Plex Mono',monospace; }}
      .tier-Elite {{ background: var(--gold-dim); color: var(--gold); }}
      .tier-Good {{ background: var(--blue-dim); color: var(--blue); }}
      .tier-Average {{ background: var(--bg-panel-2); color: var(--sage); }}
      .tier-Weak {{ background: var(--clay-dim); color: var(--clay); }}
      .honest-box {{
        background: var(--bg-panel); border-left: 3px solid var(--clay); border-radius: 0 6px 6px 0;
        padding: 14px 18px; font-size: 14px; color: var(--chalk-dim);
      }}
      .honest-box b {{ color: var(--chalk); }}
      hr {{ border-color: var(--line); }}
      div[data-testid="stDataFrame"] {{ border: 1px solid var(--line); border-radius: 8px; }}
      .stTabs [data-baseweb="tab"] {{ font-family: 'IBM Plex Mono', monospace; font-size: 13px; }}
    </style>
    """, unsafe_allow_html=True)


def plotly_layout_kwargs() -> dict:
    t = _tokens()
    return dict(
        paper_bgcolor=t["bg_panel"], plot_bgcolor=t["bg_panel"],
        font=dict(family="Inter, sans-serif", color=t["chalk"], size=12),
        title_font=dict(family="Oswald, sans-serif", size=16, color=t["chalk"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=10),
    )


def team_color(index: int) -> str:
    """Deterministic accent color from a small palette, cycling by index --
    used when a team has no dedicated color (no crest/kit data in this
    dataset)."""
    t = _tokens()
    palette = [t["gold"], t["blue"], t["clay"], t["sage"]]
    return palette[index % len(palette)]


def theme_toggle_sidebar() -> None:
    st.sidebar.markdown('<div class="eyebrow">Display</div>', unsafe_allow_html=True)
    current = st.session_state.get("theme", "dark")
    choice = st.sidebar.radio("Theme", ["dark", "light"], index=0 if current == "dark" else 1,
                               horizontal=True, label_visibility="collapsed")
    st.session_state["theme"] = choice
