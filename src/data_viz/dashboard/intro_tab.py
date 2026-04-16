"""
Narrative introduction tab.

Static layout — no callbacks required.
"""
from __future__ import annotations

from dash import dcc, html

# ── Colour palette (mirrors factor_dashboard._FACTOR_COLORS) ──────────────────
_FACTOR_PILLS: list[tuple[str, str]] = [
    ("Momentum",      "#4C72B0"),
    ("Low Volatility","#55A868"),
    ("Value",         "#C44E52"),
    ("Quality",       "#8172B3"),
    ("Growth",        "#DD8452"),
    ("Multi-Factor",  "#1A1A1A"),
]

_NAVY   = "#1e3d6e"
_ACCENT = "#2d6bc4"
_LIGHT  = "#f4f6fb"
_CARD_SHADOW = "0 2px 8px rgba(0,0,0,0.08)"


def _section_card(
    title: str,
    body: list,
    accent_color: str = _ACCENT,
    style_override: dict | None = None,
) -> html.Div:
    style = {
        "backgroundColor": "white",
        "borderRadius": "10px",
        "boxShadow": _CARD_SHADOW,
        "borderTop": f"4px solid {accent_color}",
        "padding": "28px 32px",
        "marginBottom": "24px",
        **(style_override or {}),
    }
    return html.Div(
        [html.H3(title, style={"color": accent_color, "marginTop": "0", "marginBottom": "14px",
                                "fontSize": "17px", "fontWeight": "700", "letterSpacing": "0.3px"})]
        + body,
        style=style,
    )


def _pill(label: str, color: str) -> html.Span:
    return html.Span(
        label,
        style={
            "display": "inline-block",
            "backgroundColor": color,
            "color": "white",
            "borderRadius": "20px",
            "padding": "4px 14px",
            "fontSize": "12px",
            "fontWeight": "600",
            "marginRight": "8px",
            "marginBottom": "8px",
            "letterSpacing": "0.3px",
        },
    )


def _nav_card(
    icon: str,
    title: str,
    subtitle: str,
    description: str,
    tab_hint: str,
    color: str,
) -> html.Div:
    return html.Div(
        [
            html.Div(icon, style={"fontSize": "28px", "marginBottom": "10px"}),
            html.H4(title, style={"color": color, "margin": "0 0 4px 0",
                                   "fontSize": "15px", "fontWeight": "700"}),
            html.P(subtitle, style={"color": "#666", "fontSize": "12px",
                                     "margin": "0 0 10px 0", "fontStyle": "italic"}),
            html.P(description, style={"color": "#333", "fontSize": "13px",
                                        "lineHeight": "1.6", "margin": "0 0 14px 0"}),
            html.Div(
                f"→ {tab_hint}",
                style={"color": color, "fontSize": "12px", "fontWeight": "600",
                       "borderTop": f"1px solid {color}30", "paddingTop": "10px"},
            ),
        ],
        style={
            "flex": "1",
            "backgroundColor": "white",
            "borderRadius": "10px",
            "boxShadow": _CARD_SHADOW,
            "borderTop": f"4px solid {color}",
            "padding": "24px 22px",
        },
    )


def build_intro_layout() -> html.Div:
    return html.Div(
        [
            # ── Hero ────────────────────────────────────────────────────────────
            html.Div(
                [
                    html.Div(
                        [
                            html.H1(
                                "Factor Intelligence Dashboard",
                                style={"color": "white", "fontSize": "28px",
                                       "fontWeight": "800", "margin": "0 0 8px 0",
                                       "letterSpacing": "0.5px"},
                            ),
                            html.P(
                                "Russell 1000  ·  Systematic Equity Research  ·  Regime-Aware Factor Analysis",
                                style={"color": "#a8c0e8", "fontSize": "13px",
                                       "margin": "0 0 20px 0", "letterSpacing": "0.8px"},
                            ),
                            html.P(
                                "A portfolio manager's toolkit for navigating a universe of 1,000+ stocks "
                                "through the lens of systematic factor investing — anchored in academic "
                                "asset pricing research and calibrated for real-world regime dynamics.",
                                style={"color": "#c8d8f0", "fontSize": "14px",
                                       "lineHeight": "1.7", "maxWidth": "720px", "margin": "0"},
                            ),
                        ],
                        style={"maxWidth": "960px", "margin": "0 auto"},
                    )
                ],
                style={
                    "background": f"linear-gradient(135deg, {_NAVY} 0%, #2d5090 100%)",
                    "padding": "48px 32px",
                    "marginBottom": "32px",
                },
            ),

            # ── Body ─────────────────────────────────────────────────────────────
            html.Div(
                [
                    # 1 — The Challenge
                    _section_card(
                        "The Challenge",
                        [
                            html.P(
                                "A Portfolio Manager overseeing the Russell 1000 faces a fundamental constraint: "
                                "with more than 1,000 stocks and limited analyst capacity, rigorous bottom-up "
                                "coverage of every name is simply not feasible. Manually tracking earnings, "
                                "balance sheets, management quality, competitive dynamics, and valuation for "
                                "each stock would require dozens of analysts and months of work — and the "
                                "landscape would have already shifted by the time the analysis was complete.",
                                style={"color": "#333", "lineHeight": "1.75", "margin": "0 0 12px 0"},
                            ),
                            html.P(
                                "Yet each stock behaves differently. It responds differently to macro shocks, "
                                "loads differently on risk premia, and deserves differentiated portfolio weights. "
                                "The question is: how do you scale intelligent, differentiated analysis across "
                                "an entire large-cap universe?",
                                style={"color": "#333", "lineHeight": "1.75", "margin": "0"},
                            ),
                        ],
                        accent_color="#c0392b",
                    ),

                    # 2 — The Academic Anchor
                    _section_card(
                        "The Academic Answer: Systematic Factors",
                        [
                            html.P(
                                "Decades of asset pricing research — from Fama & French (1993) to Carhart (1997), "
                                "Novy-Marx (2013), Frazzini & Pedersen (2014), and beyond — have established that "
                                "a compact set of systematic factors explains a large share of cross-sectional "
                                "return variation. Rather than analysing 1,000 stocks individually, a portfolio "
                                "manager can score each one along these dimensions and construct exposures accordingly.",
                                style={"color": "#333", "lineHeight": "1.75", "margin": "0 0 16px 0"},
                            ),
                            html.P(
                                "This dashboard covers five well-documented factors plus a multi-factor composite:",
                                style={"color": "#555", "fontSize": "13px", "margin": "0 0 10px 0"},
                            ),
                            html.Div(
                                [_pill(label, color) for label, color in _FACTOR_PILLS],
                                style={"marginBottom": "16px"},
                            ),
                            html.P(
                                "Each stock receives a z-score on every factor (winsorised at the 2nd–98th "
                                "percentile). The top decile (D10) forms the long book; the bottom decile (D1) "
                                "forms the short book in a long-short implementation, or the universe "
                                "for a long-only strategy.",
                                style={"color": "#555", "fontSize": "13px",
                                       "lineHeight": "1.65", "margin": "0"},
                            ),
                        ],
                        accent_color=_ACCENT,
                    ),

                    # 2b — Why Not Sectors?
                    _section_card(
                        "Why Not Just Rotate Between Sectors?",
                        [
                            html.P(
                                "A common alternative is to decompose market returns by sector — dynamically "
                                "rotating between Technology, Financials, Energy, Healthcare, and so on based "
                                "on macro views. This approach is intuitive and widely used, but it is "
                                "fundamentally incomplete.",
                                style={"color": "#333", "lineHeight": "1.75", "margin": "0 0 12px 0"},
                            ),
                            html.P(
                                "Sector labels hide factor exposure overlap. Technology and Consumer "
                                "Discretionary names may both exhibit strong momentum and growth characteristics. "
                                "Utilities and Healthcare may share a low-volatility profile regardless of "
                                "their sector classification. Rotating into Technology because you expect "
                                "growth to outperform is, implicitly, a bet on growth and momentum — "
                                "without explicitly measuring or controlling for those exposures.",
                                style={"color": "#333", "lineHeight": "1.75", "margin": "0 0 12px 0"},
                            ),
                            html.P(
                                [
                                    "A factor-based lens makes these exposures ",
                                    html.Strong("explicit, measurable, and manageable"),
                                    ". The ",
                                    html.Strong("Factor–Sector Scores heatmap"),
                                    " in section C of the Factor Dashboard directly "
                                    "visualises this overlap: it shows the average factor z-score of each "
                                    "GICS sector, revealing which sectors cluster together in factor space "
                                    "and where the true diversification actually lies.",
                                ],
                                style={"color": "#333", "lineHeight": "1.75", "margin": "0"},
                            ),
                        ],
                        accent_color="#8172B3",
                    ),

                    # 3 — Regime Dependency
                    _section_card(
                        "The Core Challenge: Time-Varying, Regime-Dependent Performance",
                        [
                            html.P(
                                "Factor performance is not stable through time. A strategy that generates "
                                "consistent alpha in a low-volatility bull market may suffer sharp drawdowns "
                                "during a credit crisis or a violent macro regime shift. The historical record "
                                "is unambiguous:",
                                style={"color": "#333", "lineHeight": "1.75", "margin": "0 0 12px 0"},
                            ),
                            html.Div(
                                [
                                    html.Div([
                                        html.Span("Value", style={"fontWeight": "700", "color": "#C44E52"}),
                                        html.Span(" underperformed severely through the 1990s tech boom, "
                                                   "recovered post-2000, then struggled again in the 2010s "
                                                   "low-rate environment before reversing sharply in 2022.",
                                                   style={"color": "#444"}),
                                    ], style={"marginBottom": "8px", "fontSize": "13px"}),
                                    html.Div([
                                        html.Span("Momentum", style={"fontWeight": "700", "color": "#4C72B0"}),
                                        html.Span(" crashed violently in March–May 2009 as markets "
                                                   "reversed, and again briefly in 2020.",
                                                   style={"color": "#444"}),
                                    ], style={"marginBottom": "8px", "fontSize": "13px"}),
                                    html.Div([
                                        html.Span("Low Volatility", style={"fontWeight": "700", "color": "#55A868"}),
                                        html.Span(" struggled during the 2022 rate shock as "
                                                   "rate-sensitive defensives repriced sharply.",
                                                   style={"color": "#444"}),
                                    ], style={"fontSize": "13px"}),
                                ],
                                style={
                                    "backgroundColor": _LIGHT,
                                    "borderRadius": "8px",
                                    "padding": "16px 20px",
                                    "marginBottom": "16px",
                                    "borderLeft": "3px solid #e67e22",
                                },
                            ),
                            html.P(
                                [
                                    "This creates two urgent practical questions: ",
                                    html.Strong("(1) How do we define and monitor market regimes?"),
                                    " And ",
                                    html.Strong("(2) Which factors are rewarded in each regime?"),
                                    " Answering them rigorously — and dynamically — is the central purpose "
                                    "of this dashboard.",
                                ],
                                style={"color": "#333", "lineHeight": "1.75", "margin": "0"},
                            ),
                        ],
                        accent_color="#e67e22",
                    ),

                    # 4 — Navigation cards
                    html.H3(
                        "What This Dashboard Answers",
                        style={"color": _NAVY, "fontSize": "17px", "fontWeight": "700",
                               "marginBottom": "16px", "letterSpacing": "0.3px"},
                    ),
                    html.Div(
                        [
                            _nav_card(
                                icon="",
                                title="Market Regime Monitor",
                                subtitle="Market Overview tab",
                                description=(
                                    "Track market breadth, cross-sectional return dispersion, "
                                    "and sector performance to identify the current regime. "
                                    "Animated treemap and time-series views allow you to follow "
                                    "regime transitions in real time."
                                ),
                                tab_hint="Market Overview",
                                color="#2d6bc4",
                            ),
                            _nav_card(
                                icon="",
                                title="Factor Performance by Regime",
                                subtitle="Factor Dashboard — sections A & B",
                                description=(
                                    "Analyse cumulative returns, rolling Sharpe ratios, "
                                    "calendar-year bar charts, and cross-factor heatmaps. "
                                    "Compare long-only vs long-short implementations with "
                                    "realistic transaction costs."
                                ),
                                tab_hint="Factor Dashboard › A & B",
                                color="#8172B3",
                            ),
                            _nav_card(
                                icon="",
                                title="Individual Stock Scoring",
                                subtitle="Factor Dashboard — sections C & D",
                                description=(
                                    "Score every stock in the universe on all five factors "
                                    "plus the multi-factor composite. Identify D10 longs and "
                                    "D1 shorts, explore sector-level factor clustering, and "
                                    "drill into any name at any historical date."
                                ),
                                tab_hint="Factor Dashboard › C & D",
                                color="#55A868",
                            ),
                        ],
                        style={"display": "flex", "gap": "20px", "marginBottom": "32px",
                               "flexWrap": "wrap"},
                    ),

                    # Footer / References
                    html.Div(
                        [
                            html.P(
                                "Key references",
                                style={"fontWeight": "700", "color": "#555",
                                       "fontSize": "12px", "marginBottom": "6px"},
                            ),
                            html.P(
                                "Fama & French (1993) · Carhart (1997) · Fama & French (2015) · "
                                "Novy-Marx (2013) · Frazzini & Pedersen (2014, Betting Against Beta) · "
                                "Asness, Moskowitz & Pedersen (2013, Value and Momentum Everywhere)",
                                style={"color": "#999", "fontSize": "11px", "lineHeight": "1.6",
                                       "margin": "0"},
                            ),
                        ],
                        style={
                            "borderTop": "1px solid #e0e0e0",
                            "paddingTop": "20px",
                            "marginTop": "8px",
                        },
                    ),
                ],
                style={"maxWidth": "960px", "margin": "0 auto", "padding": "0 24px 48px"},
            ),
        ],
        style={"backgroundColor": _LIGHT, "minHeight": "100vh",
               "fontFamily": "'Segoe UI', Arial, sans-serif"},
    )