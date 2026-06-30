"""Small style layer for the Streamlit debug dashboard."""

from __future__ import annotations

import html

import streamlit as st


LIGHT_PALETTE = """
            --foa-bg: #f6f8fb;
            --foa-panel: rgba(255, 255, 255, 0.86);
            --foa-panel-strong: rgba(255, 255, 255, 0.96);
            --foa-ink: #182232;
            --foa-muted: #617084;
            --foa-line: rgba(39, 52, 70, 0.13);
            --foa-blue: #4267f5;
            --foa-teal: #00a89d;
            --foa-amber: #c47a13;
            --foa-rose: #bf4b6b;
            --foa-header-bg: rgba(246, 248, 251, 0.84);
            --foa-shadow: rgba(22, 35, 58, 0.08);
            --foa-metric-shadow: rgba(31, 45, 61, 0.05);
            --foa-button-shadow: rgba(0, 168, 157, 0.12);
            --foa-button-bg: rgba(255, 255, 255, 0.88);
            --foa-button-ink: #1e2c42;
            --foa-code-bg: rgba(255, 255, 255, 0.86);
            --foa-code-ink: #182232;
            --foa-input-bg: rgba(255, 255, 255, 0.96);
            --foa-input-ink: #182232;
            --foa-app-background:
                radial-gradient(circle at top left, rgba(66, 103, 245, 0.12), transparent 34rem),
                radial-gradient(circle at 85% 10%, rgba(0, 168, 157, 0.10), transparent 26rem),
                linear-gradient(180deg, #f9fbff 0%, #f6f8fb 48%, #f4f7fb 100%);
            --foa-hero-background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(255, 255, 255, 0.68)),
                linear-gradient(90deg, rgba(66, 103, 245, 0.08), rgba(0, 168, 157, 0.08));
            --foa-sidebar-bg:
                linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(246, 249, 255, 0.98) 52%, rgba(239, 249, 248, 0.98));
            --foa-sidebar-ink: #17233a;
            --foa-sidebar-muted: #66758b;
            --foa-sidebar-section: #315dde;
            --foa-sidebar-brand-ink: #142136;
            --foa-sidebar-border: rgba(39, 52, 70, 0.12);
            --foa-sidebar-divider:
                linear-gradient(90deg, transparent, rgba(39, 52, 70, 0.16), transparent);
            --foa-sidebar-input-bg: rgba(255, 255, 255, 0.98);
            --foa-sidebar-input-ink: #132033;
            --foa-sidebar-input-border: rgba(66, 103, 245, 0.25);
            --foa-sidebar-placeholder: rgba(19, 32, 51, 0.48);
            --foa-sidebar-code-bg: rgba(66, 103, 245, 0.09);
            --foa-sidebar-code-ink: #274987;
            --foa-sidebar-code-border: rgba(66, 103, 245, 0.18);
            --foa-sidebar-button-bg: rgba(255, 255, 255, 0.72);
            --foa-sidebar-button-border: rgba(39, 52, 70, 0.14);
            --foa-sidebar-button-hover-bg: rgba(66, 103, 245, 0.08);
            --foa-sidebar-button-hover-border: rgba(66, 103, 245, 0.28);
"""

DARK_PALETTE = """
            --foa-bg: #0c111b;
            --foa-panel: rgba(21, 29, 42, 0.88);
            --foa-panel-strong: rgba(25, 35, 51, 0.96);
            --foa-ink: #eef4ff;
            --foa-muted: rgba(207, 218, 233, 0.74);
            --foa-line: rgba(215, 226, 242, 0.14);
            --foa-blue: #8eb2ff;
            --foa-teal: #52d3c8;
            --foa-amber: #ffca74;
            --foa-rose: #ff9ab7;
            --foa-header-bg: rgba(12, 17, 27, 0.88);
            --foa-shadow: rgba(0, 0, 0, 0.32);
            --foa-metric-shadow: rgba(0, 0, 0, 0.22);
            --foa-button-shadow: rgba(82, 211, 200, 0.13);
            --foa-button-bg: rgba(25, 35, 51, 0.92);
            --foa-button-ink: #eef4ff;
            --foa-code-bg: rgba(9, 14, 23, 0.92);
            --foa-code-ink: #e8f0ff;
            --foa-input-bg: rgba(13, 20, 32, 0.96);
            --foa-input-ink: #eef4ff;
            --foa-app-background:
                radial-gradient(circle at top left, rgba(83, 119, 255, 0.16), transparent 34rem),
                radial-gradient(circle at 85% 10%, rgba(38, 208, 193, 0.12), transparent 26rem),
                linear-gradient(180deg, #0c111b 0%, #101725 45%, #0c111b 100%);
            --foa-hero-background:
                linear-gradient(135deg, rgba(20, 29, 43, 0.98), rgba(14, 22, 35, 0.92)),
                linear-gradient(90deg, rgba(83, 119, 255, 0.16), rgba(38, 208, 193, 0.12));
            --foa-sidebar-bg:
                linear-gradient(180deg, rgba(10, 17, 30, 0.99), rgba(14, 24, 41, 0.98) 52%, rgba(18, 29, 46, 0.99));
            --foa-sidebar-ink: rgba(248, 250, 252, 0.95);
            --foa-sidebar-muted: rgba(226, 232, 240, 0.74);
            --foa-sidebar-section: rgba(158, 217, 255, 0.92);
            --foa-sidebar-brand-ink: #ffffff;
            --foa-sidebar-border: rgba(255, 255, 255, 0.08);
            --foa-sidebar-divider:
                linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.22), transparent);
            --foa-sidebar-input-bg: rgba(255, 255, 255, 0.96);
            --foa-sidebar-input-ink: #132033;
            --foa-sidebar-input-border: rgba(117, 195, 255, 0.34);
            --foa-sidebar-placeholder: rgba(19, 32, 51, 0.48);
            --foa-sidebar-code-bg: rgba(117, 195, 255, 0.18);
            --foa-sidebar-code-ink: #ffffff;
            --foa-sidebar-code-border: rgba(117, 195, 255, 0.24);
            --foa-sidebar-button-bg: rgba(255, 255, 255, 0.055);
            --foa-sidebar-button-border: rgba(255, 255, 255, 0.10);
            --foa-sidebar-button-hover-bg: rgba(255, 255, 255, 0.09);
            --foa-sidebar-button-hover-border: rgba(148, 210, 255, 0.44);
"""

SYSTEM_PALETTE = """
            --foa-bg: light-dark(#f6f8fb, #0c111b);
            --foa-panel: light-dark(rgba(255, 255, 255, 0.86), rgba(21, 29, 42, 0.88));
            --foa-panel-strong: light-dark(rgba(255, 255, 255, 0.96), rgba(25, 35, 51, 0.96));
            --foa-ink: light-dark(#182232, #eef4ff);
            --foa-muted: light-dark(#617084, rgba(207, 218, 233, 0.74));
            --foa-line: light-dark(rgba(39, 52, 70, 0.13), rgba(215, 226, 242, 0.14));
            --foa-blue: light-dark(#4267f5, #8eb2ff);
            --foa-teal: light-dark(#00a89d, #52d3c8);
            --foa-amber: light-dark(#c47a13, #ffca74);
            --foa-rose: light-dark(#bf4b6b, #ff9ab7);
            --foa-header-bg: light-dark(rgba(246, 248, 251, 0.84), rgba(12, 17, 27, 0.88));
            --foa-shadow: light-dark(rgba(22, 35, 58, 0.08), rgba(0, 0, 0, 0.32));
            --foa-metric-shadow: light-dark(rgba(31, 45, 61, 0.05), rgba(0, 0, 0, 0.22));
            --foa-button-shadow: light-dark(rgba(0, 168, 157, 0.12), rgba(82, 211, 200, 0.13));
            --foa-button-bg: light-dark(rgba(255, 255, 255, 0.88), rgba(25, 35, 51, 0.92));
            --foa-button-ink: light-dark(#1e2c42, #eef4ff);
            --foa-code-bg: light-dark(rgba(255, 255, 255, 0.86), rgba(9, 14, 23, 0.92));
            --foa-code-ink: light-dark(#182232, #e8f0ff);
            --foa-input-bg: light-dark(rgba(255, 255, 255, 0.96), rgba(13, 20, 32, 0.96));
            --foa-input-ink: light-dark(#182232, #eef4ff);
            --foa-app-background:
                radial-gradient(circle at top left, light-dark(rgba(66, 103, 245, 0.12), rgba(83, 119, 255, 0.16)), transparent 34rem),
                radial-gradient(circle at 85% 10%, light-dark(rgba(0, 168, 157, 0.10), rgba(38, 208, 193, 0.12)), transparent 26rem),
                linear-gradient(180deg, light-dark(#f9fbff, #0c111b) 0%, light-dark(#f6f8fb, #101725) 48%, light-dark(#f4f7fb, #0c111b) 100%);
            --foa-hero-background:
                linear-gradient(135deg, light-dark(rgba(255, 255, 255, 0.92), rgba(20, 29, 43, 0.98)), light-dark(rgba(255, 255, 255, 0.68), rgba(14, 22, 35, 0.92))),
                linear-gradient(90deg, light-dark(rgba(66, 103, 245, 0.08), rgba(83, 119, 255, 0.16)), light-dark(rgba(0, 168, 157, 0.08), rgba(38, 208, 193, 0.12)));
            --foa-sidebar-bg:
                linear-gradient(180deg, light-dark(rgba(255, 255, 255, 0.98), rgba(10, 17, 30, 0.99)), light-dark(rgba(246, 249, 255, 0.98), rgba(14, 24, 41, 0.98)) 52%, light-dark(rgba(239, 249, 248, 0.98), rgba(18, 29, 46, 0.99)));
            --foa-sidebar-ink: light-dark(#17233a, rgba(248, 250, 252, 0.95));
            --foa-sidebar-muted: light-dark(#66758b, rgba(226, 232, 240, 0.74));
            --foa-sidebar-section: light-dark(#315dde, rgba(158, 217, 255, 0.92));
            --foa-sidebar-brand-ink: light-dark(#142136, #ffffff);
            --foa-sidebar-border: light-dark(rgba(39, 52, 70, 0.12), rgba(255, 255, 255, 0.08));
            --foa-sidebar-divider:
                linear-gradient(90deg, transparent, light-dark(rgba(39, 52, 70, 0.16), rgba(255, 255, 255, 0.22)), transparent);
            --foa-sidebar-input-bg: light-dark(rgba(255, 255, 255, 0.98), rgba(255, 255, 255, 0.96));
            --foa-sidebar-input-ink: #132033;
            --foa-sidebar-input-border: light-dark(rgba(66, 103, 245, 0.25), rgba(117, 195, 255, 0.34));
            --foa-sidebar-placeholder: rgba(19, 32, 51, 0.48);
            --foa-sidebar-code-bg: light-dark(rgba(66, 103, 245, 0.09), rgba(117, 195, 255, 0.18));
            --foa-sidebar-code-ink: light-dark(#274987, #ffffff);
            --foa-sidebar-code-border: light-dark(rgba(66, 103, 245, 0.18), rgba(117, 195, 255, 0.24));
            --foa-sidebar-button-bg: light-dark(rgba(255, 255, 255, 0.72), rgba(255, 255, 255, 0.055));
            --foa-sidebar-button-border: light-dark(rgba(39, 52, 70, 0.14), rgba(255, 255, 255, 0.10));
            --foa-sidebar-button-hover-bg: light-dark(rgba(66, 103, 245, 0.08), rgba(255, 255, 255, 0.09));
            --foa-sidebar-button-hover-border: light-dark(rgba(66, 103, 245, 0.28), rgba(148, 210, 255, 0.44));
"""


def apply_dashboard_style(theme: str = "System") -> None:
    """Apply dashboard-wide visual polish."""

    css = """
        <style>
        :root {
__FOA_PALETTE__
        }

        html,
        body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        section.main {
            color: var(--foa-ink) !important;
            background: var(--foa-app-background) !important;
        }

        [data-testid="stAppViewContainer"] > .main,
        [data-testid="stMainBlockContainer"] {
            background: transparent !important;
        }

        [data-testid="stHeader"] {
            background: var(--foa-header-bg) !important;
            backdrop-filter: blur(12px);
        }

        .block-container {
            color: var(--foa-ink);
            padding-top: 4.25rem;
            padding-bottom: 3rem;
        }

        .block-container h1,
        .block-container h2,
        .block-container h3,
        .block-container h4,
        .block-container h5,
        .block-container h6,
        .block-container p,
        .block-container label,
        .block-container span,
        .block-container div[data-testid="stMarkdownContainer"] {
            color: var(--foa-ink);
        }

        .block-container div[data-testid="stCaptionContainer"],
        .block-container .stCaptionContainer,
        .block-container small {
            color: var(--foa-muted) !important;
        }

        pre,
        code,
        div[data-testid="stCodeBlock"] pre {
            color: var(--foa-code-ink) !important;
            background: var(--foa-code-bg) !important;
        }

        pre,
        div[data-testid="stCodeBlock"] pre {
            border: 1px solid var(--foa-line);
            border-radius: 8px;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-top: 4.75rem;
            }
        }

        [data-testid="stSidebar"] {
            background: var(--foa-sidebar-bg);
            border-right: 1px solid var(--foa-sidebar-border);
        }

        [data-testid="stSidebar"] * {
            color: var(--foa-sidebar-ink);
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stCaptionContainer,
        [data-testid="stSidebar"] p {
            color: var(--foa-sidebar-muted);
        }

        [data-testid="stSidebar"] code {
            color: var(--foa-sidebar-code-ink);
            background: var(--foa-sidebar-code-bg) !important;
            border: 1px solid var(--foa-sidebar-code-border);
            border-radius: 4px;
            padding: 0.08rem 0.26rem;
        }

        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] div[data-baseweb="input"] {
            color: var(--foa-sidebar-input-ink) !important;
            -webkit-text-fill-color: var(--foa-sidebar-input-ink) !important;
            caret-color: var(--foa-sidebar-input-ink);
            background: var(--foa-sidebar-input-bg) !important;
            border-color: var(--foa-sidebar-input-border) !important;
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] span {
            color: var(--foa-sidebar-input-ink) !important;
            -webkit-text-fill-color: var(--foa-sidebar-input-ink) !important;
        }

        [data-testid="stSidebar"] input::placeholder,
        [data-testid="stSidebar"] textarea::placeholder {
            color: var(--foa-sidebar-placeholder) !important;
            -webkit-text-fill-color: var(--foa-sidebar-placeholder) !important;
        }

        .sidebar-brand {
            padding: 0.7rem 0.15rem 1.05rem;
            margin-bottom: 0.35rem;
        }

        .sidebar-brand strong {
            display: block;
            font-size: 1.32rem;
            line-height: 1.15;
            color: var(--foa-sidebar-brand-ink);
        }

        .brand-kicker,
        .sidebar-section-title {
            display: block;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
            color: var(--foa-sidebar-section);
        }

        .sidebar-section-title {
            margin: 1rem 0 0.45rem;
        }

        .sidebar-divider {
            height: 1px;
            margin: 1rem 0 0.55rem;
            background: var(--foa-sidebar-divider);
        }

        .dashboard-hero {
            border: 1px solid var(--foa-line);
            background: var(--foa-hero-background);
            border-radius: 8px;
            padding: 1rem 1.15rem;
            margin-bottom: 1rem;
            box-shadow: 0 18px 45px var(--foa-shadow);
        }

        .dashboard-hero .eyebrow {
            color: var(--foa-blue);
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
        }

        .dashboard-hero h1 {
            font-size: 1.72rem;
            line-height: 1.2;
            margin: 0.12rem 0 0.18rem;
            color: var(--foa-ink);
        }

        .dashboard-hero p {
            margin: 0;
            color: var(--foa-muted);
            font-size: 0.95rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid var(--foa-line);
            background: var(--foa-panel);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            box-shadow: 0 10px 26px var(--foa-metric-shadow);
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            color: var(--foa-muted);
        }

        div.stButton > button {
            border-radius: 8px;
            border: 1px solid rgba(66, 103, 245, 0.26);
            background: var(--foa-button-bg) !important;
            color: var(--foa-button-ink) !important;
            font-weight: 650;
            transition: border-color 120ms ease, box-shadow 120ms ease, transform 120ms ease;
        }

        div.stButton > button:hover {
            border-color: rgba(0, 168, 157, 0.55);
            box-shadow: 0 8px 22px var(--foa-button-shadow);
            transform: translateY(-1px);
        }

        [data-testid="stSidebar"] div.stButton > button {
            justify-content: flex-start;
            min-height: 2.65rem;
            border-color: var(--foa-sidebar-button-border);
            background: var(--foa-sidebar-button-bg) !important;
            color: var(--foa-sidebar-ink) !important;
            box-shadow: none;
        }

        [data-testid="stSidebar"] div.stButton > button:hover {
            border-color: var(--foa-sidebar-button-hover-border);
            background: var(--foa-sidebar-button-hover-bg) !important;
            box-shadow: none;
        }

        [data-testid="stSidebar"] div.stButton > button:disabled {
            opacity: 1;
            color: #ffffff !important;
            border-color: rgba(117, 195, 255, 0.60);
            background:
                linear-gradient(90deg, rgba(66, 103, 245, 0.72), rgba(0, 168, 157, 0.58)) !important;
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-baseweb="select"] > div {
            border-radius: 8px;
        }

        .block-container div[data-testid="stTextInput"] input,
        .block-container div[data-testid="stTextArea"] textarea,
        .block-container div[data-baseweb="select"] > div,
        .block-container div[data-baseweb="input"],
        .block-container div[data-baseweb="textarea"] {
            color: var(--foa-input-ink) !important;
            -webkit-text-fill-color: var(--foa-input-ink) !important;
            caret-color: var(--foa-input-ink);
            background: var(--foa-input-bg) !important;
            border-color: var(--foa-line) !important;
        }

        .block-container div[data-baseweb="select"] span,
        .block-container input,
        .block-container textarea {
            color: var(--foa-input-ink) !important;
            -webkit-text-fill-color: var(--foa-input-ink) !important;
        }

        .block-container input::placeholder,
        .block-container textarea::placeholder {
            color: var(--foa-muted) !important;
            -webkit-text-fill-color: var(--foa-muted) !important;
        }
        </style>
        """.replace("__FOA_PALETTE__", _palette_variables(theme))
    st.markdown(css, unsafe_allow_html=True)


def render_dashboard_header(page: str, theme: str = "System") -> None:
    """Render a compact page header."""

    theme_class = _theme_class(theme)
    descriptions = {
        "Runner": "Launch and edit configs, watch runtime logs, and keep the memory view close.",
        "Test Workshop": "Run manual E2E scripts and inspect their image and JSON artifacts.",
        "Live Play": "Follow the freshest persisted turn while a run is active.",
        "Offline Inspector": "Review prior runs, model inputs, and raw redacted state.",
        "Scoring": "Compare memory runs with ARC-AGI human baseline stats.",
    }
    safe_page = html.escape(page)
    safe_description = html.escape(descriptions.get(page, "Local runtime console."))
    st.markdown(
        f"""
        <div class="dashboard-hero {theme_class}">
            <h1>{safe_page}</h1>
            <p>{safe_description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def resolve_dashboard_theme(preference: str = "System") -> str:
    """Resolve dashboard preference against Streamlit's active UI theme."""

    normalized = preference.strip().lower()
    if normalized == "dark":
        return "Dark"
    if normalized == "light":
        return "Light"

    try:
        streamlit_theme = st.context.theme or {}
    except Exception:
        streamlit_theme = {}
    theme_type = str(streamlit_theme.get("type") or "").lower()
    if theme_type == "dark":
        return "Dark"
    return "Light"


def _theme_class(theme: str) -> str:
    normalized = theme.strip().lower()
    if normalized == "dark":
        return "foa-theme-dark"
    return "foa-theme-light"


def _palette_variables(theme: str) -> str:
    normalized = theme.strip().lower()
    if normalized == "dark":
        return DARK_PALETTE
    if normalized == "light":
        return LIGHT_PALETTE
    if normalized == "system":
        return SYSTEM_PALETTE
    return LIGHT_PALETTE
