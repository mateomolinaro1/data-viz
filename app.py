"""
Main Dash dashboard — Data Visualization Project
ENSAE Paris / Institut Polytechnique de Paris
"""

import pandas as pd
from pathlib import Path
from dash import Dash, dcc, html, Input, Output, State

from src.data_viz.components.heatmap import (
    build_correlation_heatmap,
    build_returns_heatmap,
)
from src.data_viz.components.treemap import build_treemap

ROOT = Path(__file__).parent
mkt_data   = pd.read_parquet(ROOT / "data" / "wrds_gross_query.parquet")
funda_data = pd.read_parquet(ROOT / "data" / "wrds_funda_gross_query.parquet")

DATE_MIN = mkt_data["date"].min().strftime("%Y-%m-%d")
DATE_MAX = mkt_data["date"].max().strftime("%Y-%m-%d")
N_STOCKS = mkt_data["permno"].nunique()

app = Dash(__name__, title="DataViz · ENSAE", suppress_callback_exceptions=True)

C = {
    "bg":      "#0f1117",
    "surface": "#1c1f2b",
    "border":  "#2d3048",
    "accent":  "#6366f1",
    "text":    "#e2e8f0",
    "muted":   "#64748b",
}

def kpi(label, value):
    return html.Div([
        html.P(label, style={"color": C["muted"], "fontSize": "12px", "margin": "0 0 4px"}),
        html.P(value, style={"color": C["text"], "fontSize": "20px", "fontWeight": "600", "margin": "0"}),
    ], style={
        "background": C["surface"], "border": f"1px solid {C['border']}",
        "borderRadius": "10px", "padding": "16px 20px", "flex": "1", "minWidth": "140px",
    })

def lbl(text):
    return html.P(text, style={
        "color": C["muted"], "fontSize": "11px", "fontWeight": "600",
        "textTransform": "uppercase", "letterSpacing": "0.07em", "margin": "0 0 8px",
    })

TAB_STYLE = {
    "background": C["surface"], "color": C["muted"],
    "border": f"1px solid {C['border']}", "borderRadius": "8px 8px 0 0",
    "padding": "10px 24px", "fontSize": "13px", "fontWeight": "600",
}
TAB_SELECTED = {**TAB_STYLE, "background": C["accent"], "color": "white", "border": f"1px solid {C['accent']}"}

app.layout = html.Div([

    # Header
    html.Div([
        html.H1("Market Data Dashboard", style={"color": C["text"], "fontSize": "20px", "fontWeight": "700", "margin": "0"}),
        html.P("ENSAE Paris · Data Storytelling", style={"color": C["muted"], "fontSize": "12px", "margin": "4px 0 0"}),
    ], style={"background": C["surface"], "borderBottom": f"1px solid {C['border']}", "padding": "18px 32px", "marginBottom": "24px"}),

    html.Div([

        # KPIs
        html.Div([
            kpi("Univers",      f"{N_STOCKS:,} actions"),
            kpi("Période",      f"{DATE_MIN} → {DATE_MAX}"),
            kpi("Observations", f"{len(mkt_data):,}"),
            kpi("Source",       "CRSP · Local"),
        ], style={"display": "flex", "gap": "14px", "marginBottom": "20px", "flexWrap": "wrap"}),

        # Onglets
        dcc.Tabs(id="tabs", value="heatmap", children=[
            dcc.Tab(label="Matrice de corrélation", value="heatmap",
                    style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Treemap S&P 500", value="treemap",
                    style=TAB_STYLE, selected_style=TAB_SELECTED),
        ], style={"marginBottom": "0"}),

        # Contenu des onglets
        html.Div(id="tab-content", style={
            "background": C["surface"], "border": f"1px solid {C['border']}",
            "borderRadius": "0 12px 12px 12px", "padding": "20px",
        }),

    ], style={"maxWidth": "1400px", "margin": "0 auto", "padding": "0 28px 48px"}),

], style={"fontFamily": "Inter, sans-serif", "background": C["bg"], "minHeight": "100vh"})


# ── Contenu des onglets ───────────────────────────────────────────────────────
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
)
def render_tab(tab):
    if tab == "heatmap":
        return html.Div([
            html.Div([
                html.Div([
                    lbl("Période"),
                    dcc.DatePickerRange(
                        id="date-range",
                        min_date_allowed=DATE_MIN, max_date_allowed=DATE_MAX,
                        start_date=DATE_MIN, end_date=DATE_MAX,
                        display_format="YYYY-MM-DD",
                    ),
                ], style={"flex": "1", "minWidth": "280px"}),
                html.Div([
                    lbl("Type"),
                    dcc.RadioItems(
                        id="heatmap-type",
                        options=[
                            {"label": "  Corrélations",       "value": "corr"},
                            {"label": "  Rendements mensuels","value": "ret"},
                        ],
                        value="corr",
                        labelStyle={"display": "flex", "gap": "6px", "marginBottom": "8px",
                                    "color": C["text"], "fontSize": "14px"},
                    ),
                ], style={"flex": "1", "minWidth": "160px"}),
                html.Div([
                    lbl("Nombre d'actions"),
                    dcc.Slider(id="top-n", min=10, max=50, step=5, value=25,
                        marks={i: {"label": str(i), "style": {"color": C["muted"]}} for i in [10,20,30,40,50]},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ], style={"flex": "2", "minWidth": "240px"}),
                html.Div([
                    lbl("Colorscale"),
                    dcc.Dropdown(id="colorscale",
                        options=[{"label": s, "value": s} for s in ["Plasma","RdBu","Viridis","Hot","Magma"]],
                        value="Plasma", clearable=False,
                    ),
                ], style={"flex": "1", "minWidth": "140px"}),
                html.Div([
                    html.Button("Mettre à jour", id="btn-update", n_clicks=0, style={
                        "background": C["accent"], "color": "white", "border": "none",
                        "borderRadius": "8px", "padding": "10px 22px", "fontSize": "13px",
                        "fontWeight": "600", "cursor": "pointer", "marginTop": "20px",
                    }),
                ]),
            ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap",
                      "alignItems": "flex-start", "marginBottom": "20px"}),

            dcc.Loading(type="circle", color=C["accent"],
                children=dcc.Graph(id="heatmap-graph",
                    config={"toImageButtonOptions": {"format": "png", "filename": "heatmap_ensae", "scale": 2}},
                    style={"minHeight": "560px"},
                ),
            ),
        ])

    elif tab == "treemap":
        return html.Div([
            html.Div([
                html.Div([
                    lbl("Période"),
                    dcc.DatePickerRange(
                        id="treemap-date-range",
                        min_date_allowed=DATE_MIN, max_date_allowed=DATE_MAX,
                        start_date=DATE_MIN, end_date=DATE_MAX,
                        display_format="YYYY-MM-DD",
                    ),
                ], style={"flex": "1", "minWidth": "280px"}),
                html.Div([
                    lbl("Rendement affiché"),
                    dcc.RadioItems(
                        id="ret-period",
                        options=[
                            {"label": "  YTD",  "value": "YTD"},
                            {"label": "  1 mois","value": "1M"},
                            {"label": "  3 mois","value": "3M"},
                            {"label": "  MAX",  "value": "MAX"},
                        ],
                        value="YTD",
                        labelStyle={"display": "flex", "gap": "6px", "marginBottom": "8px",
                                    "color": C["text"], "fontSize": "14px"},
                    ),
                ], style={"flex": "1", "minWidth": "160px"}),
                html.Div([
                    lbl("Nombre d'actions"),
                    dcc.Slider(id="treemap-top-n", min=50, max=200, step=25, value=100,
                        marks={i: {"label": str(i), "style": {"color": C["muted"]}} for i in [50,100,150,200]},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ], style={"flex": "2", "minWidth": "240px"}),
                html.Div([
                    html.Button("Mettre à jour", id="btn-treemap", n_clicks=0, style={
                        "background": C["accent"], "color": "white", "border": "none",
                        "borderRadius": "8px", "padding": "10px 22px", "fontSize": "13px",
                        "fontWeight": "600", "cursor": "pointer", "marginTop": "20px",
                    }),
                ]),
            ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap",
                      "alignItems": "flex-start", "marginBottom": "20px"}),

            dcc.Loading(type="circle", color=C["accent"],
                children=dcc.Graph(id="treemap-graph",
                    config={"toImageButtonOptions": {"format": "png", "filename": "treemap_ensae", "scale": 2}},
                    style={"minHeight": "650px"},
                ),
            ),
        ])


# ── Callback heatmap ──────────────────────────────────────────────────────────
@app.callback(
    Output("heatmap-graph", "figure"),
    Input("btn-update", "n_clicks"),
    State("date-range",   "start_date"),
    State("date-range",   "end_date"),
    State("heatmap-type", "value"),
    State("top-n",        "value"),
    State("colorscale",   "value"),
    prevent_initial_call=False,
)
def update_heatmap(_, start, end, htype, top_n, colorscale):
    fn = build_correlation_heatmap if htype == "corr" else build_returns_heatmap
    return fn(mkt_data=mkt_data, date_start=start, date_end=end,
              top_n=top_n, colorscale=colorscale)


# ── Callback treemap ──────────────────────────────────────────────────────────
@app.callback(
    Output("treemap-graph", "figure"),
    Input("btn-treemap", "n_clicks"),
    State("treemap-date-range", "start_date"),
    State("treemap-date-range", "end_date"),
    State("ret-period",         "value"),
    State("treemap-top-n",      "value"),
    prevent_initial_call=False,
)
def update_treemap(_, start, end, ret_period, top_n):
    return build_treemap(
        mkt_data=mkt_data, funda_data=funda_data,
        date_start=start, date_end=end,
        top_n=top_n, ret_period=ret_period,
    )


if __name__ == "__main__":
    app.run(debug=True, port=8050)