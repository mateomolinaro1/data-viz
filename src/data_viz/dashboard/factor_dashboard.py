"""
Factor Dashboard tab.

Sections
--------
A. Cumulative returns + rolling returns + performance table
B. Factor return / volatility / Sharpe heatmap
C. Factor–Sector Scores heatmap (date slider)
D. Stock Picker table (date slider, all stocks, raw + z-score)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import (
    Input, Output, State, callback_context,
    dash_table, dcc, html, no_update,
)

from data_viz.data.factor_engine import FactorEngine, FACTOR_META, _PERF_METRIC_LABELS
from data_viz.data.regime_engine import RegimeEngine, EXTREME_REGIMES, REGIME_COLORS
from data_viz.dashboard.market_overview import add_regime_bands_to_figure

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------

_FACTOR_COLORS: dict[str, str] = {
    "momentum": "#4C72B0",
    "low_vol":  "#55A868",
    "value":    "#C44E52",
    "quality":  "#8172B3",
    "growth":   "#DD8452",
    "multi":    "#1A1A1A",   # black — composite of all factors
    "ew":       "#555555",
    "cw":       "#AAAAAA",
    "tbill":    "#999999",
}

_FACTOR_LABELS: dict[str, str] = {
    "momentum": "Momentum",
    "low_vol":  "Low Vol",
    "value":    "Value",
    "quality":  "Quality",
    "growth":   "Growth",
    "multi":    "Multi-Factor",
    "ew":       "EW Market",
    "cw":       "CW Market",
    "tbill":    "T-bill (rf)",
}

_CRISIS_PERIODS: list[dict] = [
    {"label": "GFC",        "start": "2008-09-01", "end": "2009-03-31"},
    {"label": "COVID",      "start": "2020-02-01", "end": "2020-04-30"},
    {"label": "Rate shock", "start": "2022-01-01", "end": "2022-12-31"},
]
_CRISIS_GREY_FILL = "rgba(160,160,160,0.13)"
_CRISIS_GREY_LINE = "rgba(110,110,110,0.40)"

_PERF_FMT: dict[str, str] = {
    "ann_ret": "{:+.1%}",
    "ann_vol": "{:.1%}",
    "sharpe":  "{:.2f}",
    "beta":    "{:.2f}",
    "alpha":   "{:+.1%}",
    "ir":      "{:.2f}",
    "max_dd":  "{:.1%}",
}


# ---------------------------------------------------------------
# Helper: crisis vrects
# ---------------------------------------------------------------

def _add_crisis(fig: go.Figure, x_min=None, x_max=None):
    for c in _CRISIS_PERIODS:
        t0, t1 = pd.Timestamp(c["start"]), pd.Timestamp(c["end"])
        if x_min and t1 < pd.Timestamp(x_min): continue
        if x_max and t0 > pd.Timestamp(x_max): continue
        fig.add_vrect(
            x0=t0, x1=t1,
            fillcolor=_CRISIS_GREY_FILL, line_color=_CRISIS_GREY_LINE,
            line_width=1, layer="below",
        )
        fig.add_annotation(
            x=t0 + (t1 - t0) / 2, y=1.01, yref="paper",
            text=c["label"], showarrow=False,
            font=dict(size=9, color="#777"), xanchor="center",
        )


# ---------------------------------------------------------------
# Section A builders
# ---------------------------------------------------------------

def build_cumret_figure(
    cumret: pd.DataFrame,
    mode: str,
    regime_series: "pd.Series | None" = None,
) -> go.Figure:
    fig = go.Figure()
    if cumret.empty:
        return fig
    bm_cols = ["ew", "cw"] if mode == "lo" else ["tbill"]
    for col in cumret.columns:
        is_bm = col in bm_cols
        fig.add_trace(go.Scatter(
            x=cumret.index.tolist(),
            y=cumret[col].tolist(),
            mode="lines",
            name=_FACTOR_LABELS.get(col, col),
            line=dict(
                width=1.5 if is_bm else 2.0,
                dash="dot" if is_bm else "solid",
                color=_FACTOR_COLORS.get(col, "#999"),
            ),
            hovertemplate=(
                f"<b>{_FACTOR_LABELS.get(col, col)}</b><br>"
                "%{x|%d %b %Y}<br>%{y:+.2%}<extra></extra>"
            ),
        ))
    if regime_series is not None:
        add_regime_bands_to_figure(fig, regime_series)
    if len(cumret) > 1:
        _add_crisis(fig, cumret.index.min(), cumret.index.max())
    fig.add_hline(y=0, line_dash="dot", line_color="#aaa", line_width=1)
    fig.update_layout(
        yaxis=dict(tickformat="+.0%", title="Cumulative return (rebased to 0)"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=60, b=40),
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        hovermode="x unified", height=400,
    )
    return fig


def build_rolling_figure(
    rolling: pd.DataFrame,
    mode: str,
    window: int,
    metric: str = "return",
    regime_series: "pd.Series | None" = None,
) -> go.Figure:
    fig = go.Figure()
    if rolling.empty:
        return fig
    bm_cols = ["ew", "cw"] if mode == "lo" else ["tbill"]
    for col in rolling.columns:
        is_bm = col in bm_cols
        fig.add_trace(go.Scatter(
            x=rolling.index.tolist(),
            y=rolling[col].tolist(),
            mode="lines",
            name=_FACTOR_LABELS.get(col, col),
            line=dict(
                width=1.5 if is_bm else 2.0,
                dash="dot" if is_bm else "solid",
                color=_FACTOR_COLORS.get(col, "#999"),
            ),
            hovertemplate=(
                f"<b>{_FACTOR_LABELS.get(col, col)}</b><br>"
                "%{x|%b %Y}<br>%{y:+.1%} ann.<extra></extra>"
            ),
        ))
    if regime_series is not None:
        add_regime_bands_to_figure(fig, regime_series)
    if len(rolling) > 1:
        _add_crisis(fig, rolling.index.min(), rolling.index.max())
    fig.add_hline(y=0, line_dash="dot", line_color="#aaa", line_width=1)
    fig.update_layout(
        yaxis=dict(
            tickformat="+.0%" if metric != "sharpe" else ".2f",
            title={
                "return": f"Trailing {window}M ann. return",
                "vol":    f"Trailing {window}M ann. vol",
                "sharpe": f"Trailing {window}M ann. Sharpe",
            }.get(metric, f"Trailing {window}M"),
        ),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=60, b=40),
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        hovermode="x unified", height=360,
    )
    return fig


def build_perf_datatable(perf_df: pd.DataFrame) -> dash_table.DataTable | html.Div:
    if perf_df.empty:
        return html.Div("No performance data.", style={"color": "#999"})

    metrics = [m for m in _PERF_FMT if m in perf_df.columns]
    display = perf_df[metrics].copy().reset_index()
    display.columns = ["Strategy"] + [_PERF_METRIC_LABELS.get(m, m) for m in metrics]

    # Format each metric column
    for raw_col, disp_col in zip(metrics, display.columns[1:]):
        fmt = _PERF_FMT[raw_col]
        display[disp_col] = display[disp_col].apply(
            lambda x: fmt.format(float(x)) if pd.notna(x) else "—"
        )

    columns = [{"name": c, "id": c} for c in display.columns]

    return dash_table.DataTable(
        id="fct-perf-datatable",
        data=display.to_dict("records"),
        columns=columns,
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": "13px", "padding": "5px 12px",
                    "textAlign": "right", "fontFamily": "monospace"},
        style_cell_conditional=[
            {"if": {"column_id": "Strategy"}, "textAlign": "left", "fontWeight": "bold"},
        ],
        style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0",
                      "borderBottom": "2px solid #ccc"},
        style_data_conditional=[
            {"if": {"filter_query": '{Strategy} contains "benchmark"'},
             "backgroundColor": "#f5f5f5", "fontStyle": "italic"},
        ],
    )


# ---------------------------------------------------------------
# Section B builder
# ---------------------------------------------------------------

def build_factor_heatmap(df: pd.DataFrame, sort_by: str, metric: str) -> go.Figure:
    if df.empty:
        return go.Figure()

    # Sort factors
    if sort_by == "avg":
        order = df.mean(axis=0).sort_values(ascending=False).index.tolist()
        df = df[order]

    labels = [_FACTOR_LABELS.get(c, c) for c in df.columns]
    z = df.values.T
    n_periods = len(df)

    if metric in ("return", "roll_return"):
        colorscale, zmid, tickfmt, cb_title = "RdYlGn", 0.0, "+.1%", "Return"
    elif metric in ("vol", "vol_raw"):
        colorscale, zmid, tickfmt, cb_title = "YlOrRd", None, ".1%", "Ann. Vol"
    else:  # sharpe, sharpe_raw
        colorscale, zmid, tickfmt, cb_title = "RdYlGn", 0.0, ".2f", "Sharpe"

    tick_vals, tick_texts, seen = [], [], set()
    for i, idx in enumerate(df.index):
        yr = pd.Timestamp(idx).year
        if yr not in seen:
            seen.add(yr)
            tick_vals.append(i)
            tick_texts.append(str(yr))

    fig = go.Figure(go.Heatmap(
        z=z, x=list(range(n_periods)), y=labels,
        colorscale=colorscale, zmid=zmid,
        colorbar=dict(title=dict(text=cb_title, side="right"),
                      thickness=14, len=0.7, tickformat=tickfmt),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Value: %{z}<extra></extra>",
        xgap=0.5, ygap=1,
    ))
    fig.update_layout(
        xaxis=dict(tickvals=tick_vals, ticktext=tick_texts, showgrid=False),
        yaxis=dict(showgrid=False, autorange="reversed"),
        margin=dict(l=120, r=80, t=20, b=50),
        plot_bgcolor="#fafafa", paper_bgcolor="white", height=280,
    )
    return fig


# ---------------------------------------------------------------
# Section C builder
# ---------------------------------------------------------------

def build_sector_scores_figure(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()

    sectors = df.index.tolist()
    factors = [_FACTOR_LABELS.get(c, c) for c in df.columns]
    z = df.values

    fig = go.Figure(go.Heatmap(
        z=z, x=factors, y=sectors,
        colorscale="RdYlGn", zmid=0.0,   # green = high z-score, red = low
        colorbar=dict(title=dict(text="Avg z-score", side="right"),
                      thickness=14, len=0.8, tickformat=".2f"),
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>Avg z-score: %{z:.2f}<extra></extra>",
        xgap=2, ygap=1,
    ))
    fig.update_layout(
        xaxis=dict(showgrid=False, side="bottom"),
        yaxis=dict(showgrid=False, autorange="reversed"),
        margin=dict(l=200, r=80, t=20, b=60),
        plot_bgcolor="#fafafa", paper_bgcolor="white", height=380,
    )
    return fig


# ---------------------------------------------------------------
# Section D builder
# ---------------------------------------------------------------

def build_stock_table(ranked: pd.DataFrame, factor: str) -> dash_table.DataTable | html.Div:
    if ranked.empty:
        return html.Div("No data available.", style={"color": "#999"})

    _, raw_col, unit = (factor,) + FACTOR_META.get(factor, (f"{factor}_raw", ""))

    # Build display columns
    display = ranked[["decile_group", "rank", "ticker", "gics_sector", "market_cap"]].copy()

    # Add ALL factors: "z-score (raw value [unit])"
    for f in FactorEngine.FACTORS:
        z_col   = f
        raw_col_name, unit_label = FACTOR_META.get(f, (f"{f}_raw", ""))
        if z_col in ranked.columns and raw_col_name in ranked.columns:
            def _fmt(row, _z=z_col, _r=raw_col_name, _u=unit_label):
                z_val  = row[_z]
                r_val  = row[_r]
                z_str  = f"{z_val:+.2f}" if pd.notna(z_val) else "—"
                if pd.notna(r_val):
                    if _u in ("12M ret", "Ann. vol"):
                        r_str = f"{r_val:+.1%}" if _u == "12M ret" else f"{r_val:.1%}"
                    elif _u == "ΔROA pp":
                        r_str = f"{r_val:+.3f}"
                    else:
                        r_str = f"{r_val:.3f}"
                    display_val = f"{z_str} ({r_str} {_u})"
                else:
                    display_val = z_str
                return display_val
            display[_FACTOR_LABELS.get(f, f)] = ranked.apply(_fmt, axis=1)
        elif z_col in ranked.columns:
            display[_FACTOR_LABELS.get(f, f)] = ranked[z_col].apply(
                lambda x: f"{x:+.2f}" if pd.notna(x) else "—"
            )

    # Format market cap
    if "market_cap" in display.columns:
        display["market_cap"] = display["market_cap"].apply(
            lambda x: f"${x / 1e9:.1f}B" if pd.notna(x) else "—"
        )

    display = display.rename(columns={
        "decile_group": "Group", "rank": "Rank",
        "ticker": "Ticker", "gics_sector": "Sector", "market_cap": "Mkt Cap",
    })

    columns = [{"name": c, "id": c} for c in display.columns]

    return dash_table.DataTable(
        data=display.to_dict("records"),
        columns=columns,
        sort_action="native",
        filter_action="native",
        page_size=len(display),       # show all rows
        style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "600px"},
        style_cell={"fontSize": "11px", "padding": "4px 8px",
                    "whiteSpace": "nowrap", "fontFamily": "monospace"},
        style_cell_conditional=[
            {"if": {"column_id": "Ticker"},   "fontWeight": "bold"},
            {"if": {"column_id": "Group"},    "fontWeight": "bold"},
        ],
        style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0",
                      "borderBottom": "2px solid #ccc"},
        style_data_conditional=[
            {"if": {"filter_query": '{Group} = "D10 (Long)"'},
             "backgroundColor": "#efffef"},
            {"if": {"filter_query": '{Group} = "D1 (Short)"'},
             "backgroundColor": "#fff0f0"},
        ],
        fixed_rows={"headers": True},
    )


# ---------------------------------------------------------------
# Section A — annual grouped bar chart
# ---------------------------------------------------------------

_BENCHMARKS = {"ew", "cw", "tbill"}

# ---------------------------------------------------------------
# Section B — Factor Profile radar chart
# ---------------------------------------------------------------

_RADAR_FACTORS: list[str] = ["momentum", "low_vol", "value", "quality", "growth"]

_RADAR_COLORS: dict[str, str] = {
    "ticker1":  "#4C72B0",
    "ticker2":  "#C44E52",
    "sector":   "#55A868",
    "universe": "#AAAAAA",
}


def _make_radar_marks(dates: list[pd.Timestamp]) -> dict[int, str]:
    marks: dict[int, str] = {}
    seen: set[int] = set()
    for i, d in enumerate(dates):
        if d.year not in seen:
            marks[i] = str(d.year)
            seen.add(d.year)
    return marks


def build_radar_figure(
    scores_df: pd.DataFrame,
    ticker1: str | None,
    mode: str,
    ticker2: str | None = None,
) -> go.Figure:
    """
    Radar chart of factor percentile ranks (0–100) for one or two tickers.

    NaN factors are set to 50 (universe median = neutral) and listed
    in an annotation below the chart.
    """
    if scores_df.empty or not ticker1:
        return go.Figure()

    factors   = _RADAR_FACTORS
    theta_lbl = [_FACTOR_LABELS[f] for f in factors]

    # ── Percentile ranks cross-sectionnels (0–100) ──────────────────
    pct = scores_df[factors].rank(pct=True) * 100
    pct = pct.copy()
    pct["ticker"]      = scores_df["ticker"].values
    pct["gics_sector"] = scores_df["gics_sector"].values

    fig = go.Figure()

    # ── Univers médiane (toujours 50 par construction) ───────────────
    fig.add_trace(go.Scatterpolar(
        r=[50.0] * len(factors),
        theta=theta_lbl,
        fill="toself",
        fillcolor="rgba(170,170,170,0.08)",
        line=dict(color="#AAAAAA", dash="dash", width=1.5),
        name="Universe median",
    ))

    # ── Ticker 1 ────────────────────────────────────────────────────
    row1 = pct.loc[pct["ticker"] == ticker1]
    if row1.empty:
        return fig

    nan_idx1: set[int] = set()
    missing1: list[str] = []
    vals1: list[float] = []
    for i, f in enumerate(factors):
        v = row1[f].iloc[0]
        if pd.isna(v):
            nan_idx1.add(i)
            missing1.append(_FACTOR_LABELS[f])
            vals1.append(50.0)
        else:
            vals1.append(float(v))
    text1 = ["n/a" if i in nan_idx1 else str(int(round(vals1[i]))) for i in range(len(factors))]

    fig.add_trace(go.Scatterpolar(
        r=vals1,
        theta=theta_lbl,
        fill="toself",
        fillcolor="rgba(76,114,176,0.25)",
        line=dict(color=_RADAR_COLORS["ticker1"], width=2.5),
        name=ticker1,
        mode="lines+markers+text",
        text=text1,
        textposition="top center",
        textfont=dict(size=11, color=_RADAR_COLORS["ticker1"]),
    ))

    # ── Mode vs_sector ───────────────────────────────────────────────
    if mode == "vs_sector":
        sector_ser = scores_df.loc[scores_df["ticker"] == ticker1, "gics_sector"]
        if not sector_ser.empty and pd.notna(sector_ser.iloc[0]):
            sector = sector_ser.iloc[0]
            mask = pct["gics_sector"] == sector
            sector_med = pct.loc[mask, factors].median()
            nan_idx_s: set[int] = set()
            sector_vals: list[float] = []
            for i, f in enumerate(factors):
                v = sector_med[f]
                if pd.isna(v):
                    nan_idx_s.add(i)
                    sector_vals.append(50.0)
                else:
                    sector_vals.append(float(v))
            text_s = ["n/a" if i in nan_idx_s else str(int(round(sector_vals[i]))) for i in range(len(factors))]
            fig.add_trace(go.Scatterpolar(
                r=sector_vals,
                theta=theta_lbl,
                fill="toself",
                fillcolor="rgba(85,168,104,0.15)",
                line=dict(color=_RADAR_COLORS["sector"], width=2, dash="dot"),
                name=f"{sector} median",
                mode="lines+markers+text",
                text=text_s,
                textposition="top center",
                textfont=dict(size=11, color=_RADAR_COLORS["sector"]),
            ))

    # ── Mode vs_ticker ───────────────────────────────────────────────
    elif mode == "vs_ticker" and ticker2:
        row2 = pct.loc[pct["ticker"] == ticker2]
        if not row2.empty:
            nan_idx2: set[int] = set()
            missing2: list[str] = []
            vals2: list[float] = []
            for i, f in enumerate(factors):
                v = row2[f].iloc[0]
                if pd.isna(v):
                    nan_idx2.add(i)
                    missing2.append(_FACTOR_LABELS[f])
                    vals2.append(50.0)
                else:
                    vals2.append(float(v))
            text2 = ["n/a" if i in nan_idx2 else str(int(round(vals2[i]))) for i in range(len(factors))]
            fig.add_trace(go.Scatterpolar(
                r=vals2,
                theta=theta_lbl,
                fill="toself",
                fillcolor="rgba(196,78,82,0.20)",
                line=dict(color=_RADAR_COLORS["ticker2"], width=2.5),
                name=ticker2,
                mode="lines+markers+text",
                text=text2,
                textposition="top center",
                textfont=dict(size=11, color=_RADAR_COLORS["ticker2"]),
            ))
            if missing2:
                fig.add_annotation(
                    text=f"⚠ {ticker2} — missing factors set to 50th pct: {', '.join(missing2)}",
                    xref="paper", yref="paper", x=0.5, y=-0.18,
                    showarrow=False, font=dict(size=11, color="#888"),
                    xanchor="center",
                )

    # ── Annotation facteurs manquants ticker1 ───────────────────────
    if missing1:
        fig.add_annotation(
            text=f"⚠ {ticker1} — missing factors set to 50th pct: {', '.join(missing1)}",
            xref="paper", yref="paper", x=0.5, y=-0.12,
            showarrow=False, font=dict(size=11, color="#888"),
            xanchor="center",
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                range=[0, 100],
                tickvals=[25, 50, 75],
                ticktext=["25th", "50th", "75th"],
                tickfont=dict(size=10),
                gridcolor="#e0e0e0",
            ),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.08,
                    xanchor="center", x=0.5),
        margin=dict(l=60, r=60, t=40, b=80),
        height=607,
        paper_bgcolor="white",
    )
    return fig


def build_annual_bar_figure(df: pd.DataFrame, metric: str = "ann_ret") -> go.Figure:
    if df.empty:
        return go.Figure()

    years = [str(ts.year) for ts in df.index]
    is_pct = metric in ("ann_ret", "ann_vol")
    y_tick_fmt = "+.0%" if metric == "ann_ret" else (".0%" if metric == "ann_vol" else ".2f")
    hover_fmt  = "+.1%" if metric == "ann_ret" else (".1%" if metric == "ann_vol" else ".2f")
    y_title    = {"ann_ret": "Ann. Return", "ann_vol": "Ann. Vol", "sharpe": "Ann. Sharpe"}.get(metric, metric)

    lower_is_better = metric == "ann_vol"
    factor_cols = [c for c in df.columns if c not in _BENCHMARKS]

    best_by_year: dict[str, str] = {}
    for ts in df.index:
        year = str(ts.year)
        vals = {c: df.loc[ts, c] for c in factor_cols if pd.notna(df.loc[ts, c])}
        if vals:
            best_by_year[year] = min(vals, key=vals.get) if lower_is_better else max(vals, key=vals.get)

    fig = go.Figure()
    for col in df.columns:
        line_colors = ["red" if best_by_year.get(y) == col else "rgba(0,0,0,0)" for y in years]
        line_widths = [2 if best_by_year.get(y) == col else 0 for y in years]
        fig.add_trace(go.Bar(
            name=_FACTOR_LABELS.get(col, col),
            x=years,
            y=df[col].tolist(),
            marker_color=_FACTOR_COLORS.get(col, "#999"),
            marker_line_color=line_colors,
            marker_line_width=line_widths,
            hovertemplate=(
                f"<b>{_FACTOR_LABELS.get(col, col)}</b><br>"
                f"%{{x}}<br>%{{y:{hover_fmt}}}<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="group",
        yaxis=dict(tickformat=y_tick_fmt, title=y_title),
        xaxis=dict(title="Calendar Year", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=60, b=40),
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        hovermode="x unified", height=380,
    )
    if metric != "ann_vol":
        fig.add_hline(y=0, line_dash="dot", line_color="#aaa", line_width=1)
    return fig


# ---------------------------------------------------------------
# Section E — regime-conditional factor analysis
# ---------------------------------------------------------------

def build_regime_bar_figure(df: pd.DataFrame, metric: str = "ann_ret") -> go.Figure:
    """
    Grouped bar chart: x = 4 extreme regimes, one bar per strategy.

    df : index = strategies, columns = extreme regimes  (from RegimeEngine.get_regime_factor_metrics)
    """
    if df.empty:
        return go.Figure()

    is_pct      = metric in ("ann_ret", "ann_vol")
    y_tick_fmt  = "+.0%" if metric == "ann_ret" else (".0%" if metric == "ann_vol" else ".2f")
    hover_fmt   = "+.1%" if metric == "ann_ret" else (".1%" if metric == "ann_vol" else ".2f")
    y_title     = {"ann_ret": "Ann. Return", "ann_vol": "Ann. Vol",
                   "sharpe":  "Ann. Sharpe"}.get(metric, metric)

    # x-axis: regimes (4 extreme + "Normal (mid)" baseline); bars: strategies
    regime_cols = [r for r in EXTREME_REGIMES if r in df.columns]
    if "Normal (mid)" in df.columns:
        regime_cols = regime_cols + ["Normal (mid)"]

    lower_is_better = metric == "ann_vol"
    factor_strats = [s for s in df.index if s not in _BENCHMARKS]

    best_by_regime: dict[str, str] = {}
    for r in regime_cols:
        vals = {s: float(df.loc[s, r]) for s in factor_strats if s in df.index and r in df.columns and pd.notna(df.loc[s, r])}
        if vals:
            best_by_regime[r] = min(vals, key=vals.get) if lower_is_better else max(vals, key=vals.get)

    fig = go.Figure()
    for strat in df.index:
        vals = [float(df.loc[strat, r]) if r in df.columns else np.nan for r in regime_cols]
        line_colors = ["red" if best_by_regime.get(r) == strat else "rgba(0,0,0,0)" for r in regime_cols]
        line_widths = [2 if best_by_regime.get(r) == strat else 0 for r in regime_cols]
        fig.add_trace(go.Bar(
            name=_FACTOR_LABELS.get(strat, strat),
            x=regime_cols,
            y=vals,
            marker_color=_FACTOR_COLORS.get(strat, "#999"),
            marker_line_color=line_colors,
            marker_line_width=line_widths,
            hovertemplate=(
                f"<b>{_FACTOR_LABELS.get(strat, strat)}</b><br>"
                f"%{{x}}<br>%{{y:{hover_fmt}}}<extra></extra>"
            ),
        ))

    # Dashed vertical separator before "Normal (mid)" baseline
    n_extreme = len([r for r in EXTREME_REGIMES if r in df.columns])
    if "Normal (mid)" in df.columns and n_extreme > 0:
        fig.add_vline(
            x=n_extreme - 0.5,
            line_dash="dash", line_color="#bbb", line_width=1.5,
        )

    x_labels = [r if r != "Normal (mid)" else "Normal\n(mid)" for r in regime_cols]
    fig.update_layout(
        barmode="group",
        yaxis=dict(tickformat=y_tick_fmt, title=y_title),
        xaxis=dict(
            title="",
            showgrid=False,
            ticktext=x_labels,
            tickvals=list(range(len(regime_cols))),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=60, b=70),
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        hovermode="x unified", height=420,
    )
    if metric != "ann_vol":
        fig.add_hline(y=0, line_dash="dot", line_color="#aaa", line_width=1)
    return fig


def build_regime_factor_table(
    df: pd.DataFrame,
    metric: str = "ann_ret",
) -> "dash_table.DataTable | html.Div":
    """
    Table: rows = factors, columns = extreme regimes.
    Values formatted and color-coded (green/red for return/sharpe, yellow for vol).
    """
    if df.empty:
        return html.Div("No regime data available.", style={"color": "#999"})

    regimes = [r for r in EXTREME_REGIMES if r in df.columns]
    is_pct  = metric in ("ann_ret", "ann_vol")
    fmt     = "+.1%" if metric == "ann_ret" else (".1%" if metric == "ann_vol" else "+.2f")

    display = df[regimes].copy().reset_index()
    display.columns = ["Strategy"] + [r.replace("–", "–\n") for r in regimes]

    for col_orig, col_disp in zip(regimes, display.columns[1:]):
        display[col_disp] = display[col_disp].apply(
            lambda x: fmt.format(float(x)) if pd.notna(x) else "—"
        )

    columns = [{"name": c, "id": c} for c in display.columns]

    # Conditional formatting
    style_data_cond = [
        {"if": {"filter_query": '{Strategy} contains "benchmark"'},
         "backgroundColor": "#f5f5f5", "fontStyle": "italic"},
    ]

    return dash_table.DataTable(
        data=display.to_dict("records"),
        columns=columns,
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": "13px", "padding": "5px 14px",
                    "textAlign": "right", "fontFamily": "monospace",
                    "whiteSpace": "pre-line"},
        style_cell_conditional=[
            {"if": {"column_id": "Strategy"}, "textAlign": "left",
             "fontWeight": "bold", "minWidth": "120px"},
        ],
        style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0",
                      "borderBottom": "2px solid #ccc", "whiteSpace": "pre-line"},
        style_data_conditional=style_data_cond,
    )


# ---------------------------------------------------------------
# FactorDashboardTab
# ---------------------------------------------------------------

class FactorDashboardTab:
    """
    Encapsulates layout + callbacks for the Factor Dashboard tab.

    Usage
    -----
    engine = FactorEngine(data_manager)
    engine.build()
    tab = FactorDashboardTab(engine)
    app.layout = ...tab.build_layout()...
    tab.register_callbacks(app)
    """

    def __init__(self, engine: FactorEngine,
                 regime_engine: RegimeEngine | None = None) -> None:
        self.engine = engine
        self.regime_engine = regime_engine
        # Pre-build sorted list of available monthly dates
        self._avail_dates: list[pd.Timestamp] = sorted(
            engine.factor_scores_by_month.keys()
        )

    # ----------------------------------------------------------
    # Layout helpers
    # ----------------------------------------------------------

    def _slider_config(self, gran: str) -> tuple[list[pd.Timestamp], dict[int, str]]:
        """Return (dates, marks) for the given granularity."""
        from data_viz.dashboard.market_overview import _get_dates_for_granularity, _make_marks
        all_dates = self.engine.dm.dates
        dates = _get_dates_for_granularity(gran, all_dates)
        return dates, _make_marks(dates)

    def _date_from_slider(self, idx: int, gran: str) -> pd.Timestamp:
        from data_viz.dashboard.market_overview import _get_dates_for_granularity
        dates = _get_dates_for_granularity(gran, self.engine.dm.dates)
        idx = min(int(idx), len(dates) - 1)
        return dates[idx]

    def build_layout(self) -> html.Div:
        engine = self.engine
        fr = engine.monthly_lo_returns
        min_date = fr.index.min() if fr is not None and not fr.empty else pd.Timestamp("2000-01-01")
        max_date = fr.index.max() if fr is not None and not fr.empty else pd.Timestamp("2024-12-31")

        # Slider for C and D (monthly granularity by default)
        dates_m, marks_m = self._slider_config("monthly")
        n_m = len(dates_m)
        last_m = n_m - 1

        return html.Div(
            [
                dcc.Store(id="fct-perf-numeric-store"),
                html.H3("Factor Dashboard", style={"marginBottom": "6px"}),
                html.P(
                    "Monthly-rebalanced decile factor portfolios (D10 − D1 L/S and D10 LO). "
                    "Z-scores winsorised at 2–98 %. Transaction costs: 10 bps on actual turnover.",
                    style={"color": "#666", "fontSize": "13px", "marginBottom": "24px"},
                ),

                # ================================================
                # A — Cumulative & Rolling Returns
                # ================================================
                html.H4("A — Factor Returns", style={"marginBottom": "8px"}),

                # Controls row
                html.Div([
                    # Mode
                    html.Div([
                        html.Label("Mode", style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                        dcc.RadioItems(
                            id="fct-mode",
                            options=[
                                {"label": "Long-Only (D10)", "value": "lo"},
                                {"label": "Long-Short (D10−D1)", "value": "ls"},
                            ],
                            value="ls", inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "20px", "fontSize": "13px"},
                        ),
                    ]),
                    # Start date
                    html.Div([
                        html.Label("Start date", style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                        dcc.DatePickerSingle(
                            id="fct-start-date",
                            date=str(min_date.date()),
                            min_date_allowed=str(min_date.date()),
                            max_date_allowed=str(max_date.date()),
                            display_format="DD MMM YYYY",
                            style={"fontSize": "13px"},
                        ),
                    ]),
                    # IR benchmark (only relevant for LO)
                    html.Div([
                        html.Label("IR / benchmark (LO)", style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                        dcc.RadioItems(
                            id="fct-ir-bm",
                            options=[
                                {"label": "EW Market", "value": "ew"},
                                {"label": "CW Market", "value": "cw"},
                            ],
                            value="ew", inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "16px", "fontSize": "13px"},
                        ),
                    ]),
                    # Show benchmarks
                    html.Div([
                        html.Label("Chart", style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                        dcc.Checklist(
                            id="fct-show-bm",
                            options=[{"label": " show benchmarks", "value": "yes"}],
                            value=["yes"],
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"fontSize": "13px"},
                        ),
                    ]),
                ], style={"display": "flex", "gap": "32px", "alignItems": "flex-end", "flexWrap": "wrap", "marginBottom": "12px"}),

                # Cumulative chart
                dcc.Graph(id="fct-cumret-chart",
                          config={"displayModeBar": True, "scrollZoom": False},
                          style={"height": "400px", "marginBottom": "8px"}),

                # Performance table
                html.Div(id="fct-perf-table",
                         style={"marginBottom": "16px", "overflowX": "auto"}),

                # Rolling chart controls + chart (just below perf table)
                html.Div([
                    html.Div([
                        html.Label("Rolling metric", style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                        dcc.RadioItems(
                            id="fct-rolling-metric",
                            options=[
                                {"label": "Ann. Return", "value": "return"},
                                {"label": "Ann. Vol",    "value": "vol"},
                                {"label": "Ann. Sharpe", "value": "sharpe"},
                            ],
                            value="return", inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "16px", "fontSize": "13px"},
                        ),
                    ]),
                    html.Div([
                        html.Label("Window", style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                        dcc.RadioItems(
                            id="fct-rolling-window",
                            options=[
                                {"label": "6M",  "value": "6"},
                                {"label": "12M", "value": "12"},
                                {"label": "24M", "value": "24"},
                                {"label": "36M", "value": "36"},
                            ],
                            value="12", inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "16px", "fontSize": "13px"},
                        ),
                    ]),
                ], style={"display": "flex", "gap": "32px", "alignItems": "flex-end", "marginBottom": "8px"}),

                dcc.Graph(id="fct-rolling-chart",
                          config={"displayModeBar": False},
                          style={"height": "360px", "marginBottom": "28px"}),

                # Annual bar chart
                html.Div([
                    html.Label("Bar metric", style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                    dcc.RadioItems(
                        id="fct-bar-metric",
                        options=[
                            {"label": "Ann. Return", "value": "ann_ret"},
                            {"label": "Ann. Vol",    "value": "ann_vol"},
                            {"label": "Ann. Sharpe", "value": "sharpe"},
                        ],
                        value="ann_ret", inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "16px", "fontSize": "13px"},
                    ),
                ], style={"marginBottom": "8px"}),

                dcc.Graph(id="fct-annual-bar",
                          config={"displayModeBar": False},
                          style={"height": "380px", "marginBottom": "16px"}),

                # Section B — Factor Performance by Regime
                *(self._build_regime_section() if self.regime_engine is not None else []),

                html.Div(style={"marginBottom": "20px"}),

                # ================================================
                # B — Factor Heatmaps
                # ================================================
                html.H4("C — Factor Heatmaps", style={"marginBottom": "8px"}),

                html.Div([
                    html.Div([
                        html.Label("Mode", style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                        dcc.RadioItems(
                            id="fct-hm-mode",
                            options=[{"label": "LO", "value": "lo"}, {"label": "L/S", "value": "ls"}],
                            value="ls", inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "12px", "fontSize": "13px"},
                        ),
                    ]),
                    html.Div([
                        html.Label("Metric", style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                        dcc.RadioItems(
                            id="fct-hm-metric",
                            options=[
                                {"label": "Return (raw)",      "value": "return"},
                                {"label": "Return (rolling)",  "value": "roll_return"},
                                {"label": "Vol (raw)",         "value": "vol_raw"},
                                {"label": "Vol (rolling)",     "value": "vol"},
                                {"label": "Sharpe (raw)",      "value": "sharpe_raw"},
                                {"label": "Sharpe (rolling)",  "value": "sharpe"},
                            ],
                            value="return", inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "14px", "fontSize": "13px"},
                        ),
                    ]),
                    html.Div([
                        html.Label("Granularity", style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                        dcc.RadioItems(
                            id="fct-hm-gran",
                            options=[{"label": "Monthly", "value": "monthly"}, {"label": "Yearly", "value": "yearly"}],
                            value="monthly", inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "16px", "fontSize": "13px"},
                        ),
                    ]),
                    html.Div([
                        html.Label("Sort factors", style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                        dcc.Dropdown(
                            id="fct-hm-sort",
                            options=[
                                {"label": "Alphabetical",   "value": "alpha"},
                                {"label": "Avg value (↓)",  "value": "avg"},
                            ],
                            value="alpha", clearable=False,
                            style={"width": "160px", "fontSize": "13px"},
                        ),
                    ]),
                    html.Div([
                        html.Label("Rolling window (vol/Sharpe)",
                                   style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                        dcc.RadioItems(
                            id="fct-hm-window",
                            options=[{"label": "6M", "value": "6"}, {"label": "12M", "value": "12"},
                                     {"label": "24M", "value": "24"}],
                            value="12", inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "12px", "fontSize": "13px"},
                        ),
                    ]),
                ], style={"display": "flex", "gap": "28px", "alignItems": "flex-end",
                          "flexWrap": "wrap", "marginBottom": "12px"}),

                dcc.Graph(id="fct-factor-heatmap",
                          config={"displayModeBar": False},
                          style={"height": "280px", "marginBottom": "36px"}),

                # ================================================
                # C — Factor–Sector Scores
                # ================================================
                html.H4("D — Factor–Sector Scores", style={"marginBottom": "8px"}),
                html.P(
                    "Average factor z-score per GICS sector at the selected date.",
                    style={"color": "#666", "fontSize": "13px", "marginBottom": "8px"},
                ),

                html.Div([
                    html.Div([
                        html.Label("Granularity", style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                        dcc.RadioItems(
                            id="fct-exp-gran",
                            options=[
                                {"label": "Daily",   "value": "daily"},
                                {"label": "Weekly",  "value": "weekly"},
                                {"label": "Monthly", "value": "monthly"},
                                {"label": "Yearly",  "value": "yearly"},
                            ],
                            value="monthly", inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "14px", "fontSize": "13px"},
                        ),
                    ]),
                    html.Div([
                        html.Label("Sort sectors", style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                        dcc.Dropdown(
                            id="fct-exp-sort-sectors",
                            options=[
                                {"label": "Alphabetical", "value": "alpha"},
                                {"label": "Avg score (↓)", "value": "avg_score"},
                            ],
                            value="alpha", clearable=False,
                            style={"width": "160px", "fontSize": "13px"},
                        ),
                    ]),
                    html.Div([
                        html.Label("Sort factors", style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                        dcc.Dropdown(
                            id="fct-exp-sort-factors",
                            options=[
                                {"label": "Alphabetical", "value": "alpha"},
                                {"label": "Avg score (↓)", "value": "avg_score"},
                            ],
                            value="alpha", clearable=False,
                            style={"width": "160px", "fontSize": "13px"},
                        ),
                    ]),
                ], style={"display": "flex", "gap": "28px", "alignItems": "flex-end",
                          "flexWrap": "wrap", "marginBottom": "8px"}),

                html.Div([
                    html.Label("Snapshot date",
                               style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                    html.Div([
                        html.Div(
                            dcc.Slider(
                                id="fct-exp-slider",
                                min=0, max=n_m - 1, step=1, value=last_m,
                                marks=marks_m,
                                tooltip={"always_visible": False, "placement": "bottom"},
                            ),
                            style={"flex": "1"},
                        ),
                        html.Div(
                            id="fct-exp-date-label",
                            children=dates_m[-1].strftime("%b %Y") if dates_m else "",
                            style={"fontSize": "13px", "fontWeight": "bold",
                                   "minWidth": "80px", "textAlign": "right", "flexShrink": "0"},
                        ),
                    ], style={"display": "flex", "alignItems": "center", "gap": "12px"}),
                ], style={"marginBottom": "12px"}),

                dcc.Graph(id="fct-sector-scores",
                          config={"displayModeBar": False},
                          style={"height": "380px", "marginBottom": "36px"}),

                # ================================================
                # D — Factor Profile — Cross-sectional View
                # ================================================
                html.H4("E — Factor Profile — Cross-sectional View",
                        style={"marginBottom": "8px"}),
                html.P(
                    "Percentile rank (0–100) cross-sectional on the universe at the selected date. "
                    "Score 50 = universe median. Factors with missing data are set to 50th percentile.",
                    style={"color": "#666", "fontSize": "13px", "marginBottom": "12px"},
                ),

                # Controls row
                html.Div([
                    html.Div([
                        html.Label("Mode", style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                        dcc.RadioItems(
                            id="fct-radar-mode",
                            options=[
                                {"label": "Stock only",        "value": "single"},
                                {"label": "vs Sector median",  "value": "vs_sector"},
                                {"label": "vs Another stock",  "value": "vs_ticker"},
                            ],
                            value="single", inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "16px", "fontSize": "13px"},
                        ),
                    ]),
                    html.Div([
                        html.Label("Ticker 1", style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                        dcc.Dropdown(
                            id="fct-radar-ticker1",
                            options=[],
                            placeholder="Search ticker…",
                            searchable=True, clearable=False,
                            style={"width": "160px", "fontSize": "13px"},
                        ),
                    ]),
                    html.Div(
                        id="fct-radar-ticker2-container",
                        children=[
                            html.Label("Ticker 2", style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                            dcc.Dropdown(
                                id="fct-radar-ticker2",
                                options=[],
                                placeholder="Search ticker…",
                                searchable=True, clearable=True,
                                style={"width": "160px", "fontSize": "13px"},
                            ),
                        ],
                        style={"display": "none"},
                    ),
                ], style={"display": "flex", "gap": "28px", "alignItems": "flex-end",
                          "flexWrap": "wrap", "marginBottom": "12px"}),

                dcc.Interval(id="fct-radar-interval", interval=600, disabled=True),

                # Date slider + Play button
                html.Div([
                    html.Label("Snapshot date",
                               style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                    html.Div([
                        html.Button(
                            "▶", id="fct-radar-play-btn",
                            style={"width": "36px", "height": "36px", "borderRadius": "50%",
                                   "border": "none", "backgroundColor": "#e6f1fb",
                                   "color": "#185FA5", "fontSize": "14px",
                                   "cursor": "pointer", "flexShrink": "0"},
                        ),
                        html.Div(
                            dcc.Slider(
                                id="fct-radar-slider",
                                min=0, max=len(self._avail_dates) - 1, step=1,
                                value=len(self._avail_dates) - 1,
                                marks=_make_radar_marks(self._avail_dates),
                                tooltip={"always_visible": False, "placement": "bottom"},
                            ),
                            style={"flex": "1"},
                        ),
                        html.Div(
                            id="fct-radar-date-label",
                            children=self._avail_dates[-1].strftime("%b %Y") if self._avail_dates else "",
                            style={"fontSize": "13px", "fontWeight": "bold",
                                   "minWidth": "80px", "textAlign": "right", "flexShrink": "0"},
                        ),
                    ], style={"display": "flex", "alignItems": "center", "gap": "12px"}),
                ], style={"marginBottom": "12px"}),

                dcc.Graph(id="fct-radar-chart",
                          config={"displayModeBar": False},
                          style={"height": "607px", "marginBottom": "12px"}),

                html.Div(id="fct-radar-breakdown",
                         style={"marginBottom": "36px", "overflowX": "auto"}),

                # ================================================
                # E — Stock Picker
                # ================================================
                html.H4("F — Stock Picker", style={"marginBottom": "8px"}),
                html.P(
                    "All stocks ranked by factor score. "
                    "Format: z-score (raw value [unit]). "
                    "D10 = top decile (long), D1 = bottom decile (short).",
                    style={"color": "#666", "fontSize": "13px", "marginBottom": "8px"},
                ),

                html.Div([
                    html.Div([
                        html.Label("Factor", style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                        dcc.Dropdown(
                            id="fct-picker-factor",
                            options=[{"label": _FACTOR_LABELS[f], "value": f}
                                     for f in FactorEngine.FACTORS],
                            value="momentum", clearable=False,
                            style={"width": "160px", "fontSize": "13px"},
                        ),
                    ]),
                    html.Div([
                        html.Label("Granularity", style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                        dcc.RadioItems(
                            id="fct-picker-gran",
                            options=[
                                {"label": "Daily",   "value": "daily"},
                                {"label": "Weekly",  "value": "weekly"},
                                {"label": "Monthly", "value": "monthly"},
                                {"label": "Yearly",  "value": "yearly"},
                            ],
                            value="monthly", inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "14px", "fontSize": "13px"},
                        ),
                    ]),
                ], style={"display": "flex", "gap": "28px", "alignItems": "flex-end",
                          "flexWrap": "wrap", "marginBottom": "8px"}),

                html.Div([
                    html.Label("Snapshot date",
                               style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                    html.Div([
                        html.Div(
                            dcc.Slider(
                                id="fct-picker-slider",
                                min=0, max=n_m - 1, step=1, value=last_m,
                                marks=marks_m,
                                tooltip={"always_visible": False, "placement": "bottom"},
                            ),
                            style={"flex": "1"},
                        ),
                        html.Div(
                            id="fct-picker-date-label",
                            children=dates_m[-1].strftime("%b %Y") if dates_m else "",
                            style={"fontSize": "13px", "fontWeight": "bold",
                                   "minWidth": "80px", "textAlign": "right", "flexShrink": "0"},
                        ),
                    ], style={"display": "flex", "alignItems": "center", "gap": "12px"}),
                ], style={"marginBottom": "12px"}),

                html.Div(id="fct-stock-table"),
            ],
            style={"padding": "10px"},
        )

    # ----------------------------------------------------------
    # Section E layout helper
    # ----------------------------------------------------------

    def _build_regime_section(self) -> list:
        engine = self.engine
        fr = engine.monthly_lo_returns
        min_date = fr.index.min() if fr is not None and not fr.empty else pd.Timestamp("2000-01-01")
        max_date = fr.index.max() if fr is not None and not fr.empty else pd.Timestamp("2024-12-31")

        return [
            html.Hr(style={"borderColor": "#ddd", "marginBottom": "28px"}),
            html.H4("B — Factor Performance by Regime",
                    style={"marginBottom": "8px"}),
            html.P(
                "Factor and benchmark returns conditional on the 4 extreme market regimes "
                "(25th/75th percentile thresholds on CW 12M return and vol). "
                "Mid-regime months excluded.",
                style={"color": "#666", "fontSize": "13px", "marginBottom": "16px"},
            ),

            # Controls
            html.Div([
                html.Div([
                    html.Label("Mode", style={"fontWeight": "bold",
                               "display": "block", "marginBottom": "4px"}),
                    dcc.RadioItems(
                        id="fct-regime-mode",
                        options=[{"label": "Long-Only (D10)", "value": "lo"},
                                 {"label": "Long-Short (D10−D1)", "value": "ls"}],
                        value="ls", inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "20px", "fontSize": "13px"},
                    ),
                ]),
                html.Div([
                    html.Label("Metric", style={"fontWeight": "bold",
                               "display": "block", "marginBottom": "4px"}),
                    dcc.RadioItems(
                        id="fct-regime-metric",
                        options=[
                            {"label": "Ann. Return", "value": "ann_ret"},
                            {"label": "Ann. Vol",    "value": "ann_vol"},
                            {"label": "Ann. Sharpe", "value": "sharpe"},
                        ],
                        value="ann_ret", inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "16px", "fontSize": "13px"},
                    ),
                ]),
                html.Div([
                    html.Label("Start date", style={"fontWeight": "bold",
                               "display": "block", "marginBottom": "4px"}),
                    dcc.DatePickerSingle(
                        id="fct-regime-start-date",
                        date=str(min_date.date()),
                        min_date_allowed=str(min_date.date()),
                        max_date_allowed=str(max_date.date()),
                        display_format="DD MMM YYYY",
                        style={"fontSize": "13px"},
                    ),
                ]),
            ], style={"display": "flex", "gap": "32px", "alignItems": "flex-end",
                      "flexWrap": "wrap", "marginBottom": "16px"}),

            # Bar chart
            dcc.Graph(id="fct-regime-bar",
                      config={"displayModeBar": False},
                      style={"height": "400px", "marginBottom": "28px"}),
        ]

    # ----------------------------------------------------------
    # Callbacks
    # ----------------------------------------------------------

    def register_callbacks(self, app) -> None:
        engine = self.engine

        # ---- A: cumulative + rolling + perf table ----
        _regime_series = self.regime_engine.regime_series if self.regime_engine else None

        @app.callback(
            Output("fct-cumret-chart",       "figure"),
            Output("fct-rolling-chart",      "figure"),
            Output("fct-perf-table",         "children"),
            Output("fct-perf-numeric-store", "data"),
            Input("fct-mode",           "value"),
            Input("fct-start-date",     "date"),
            Input("fct-ir-bm",          "value"),
            Input("fct-show-bm",        "value"),
            Input("fct-rolling-window", "value"),
            Input("fct-rolling-metric", "value"),
        )
        def update_section_a(mode, start_date_str, ir_bm, show_bm, window_str, roll_metric):
            mode = mode or "ls"
            start_date = pd.Timestamp(start_date_str) if start_date_str else None
            window = int(window_str or 12)
            show_benchmarks = bool(show_bm)
            roll_metric = roll_metric or "return"

            cumret = engine.get_daily_cumulative_returns(mode, start_date)
            if not show_benchmarks:
                cumret = cumret[[c for c in cumret.columns if c in engine.FACTORS]]

            rolling = engine.get_rolling_metric(mode, window, metric=roll_metric, start_date=start_date)
            if not show_benchmarks:
                rolling = rolling[[c for c in rolling.columns if c in engine.FACTORS]]

            perf = engine.get_performance_table(
                mode, start_date, ir_benchmark=ir_bm or "ew"
            )

            perf_store = perf.reset_index().to_json(orient="records")

            return (
                build_cumret_figure(cumret, mode, regime_series=_regime_series),
                build_rolling_figure(rolling, mode, window, roll_metric, regime_series=_regime_series),
                build_perf_datatable(perf),
                perf_store,
            )

        # ---- A: highlight best row in perf table on sort ----
        _LABEL_TO_METRIC = {v: k for k, v in _PERF_METRIC_LABELS.items()}
        _LOWER_IS_BETTER = {"ann_vol"}

        @app.callback(
            Output("fct-perf-datatable", "style_data_conditional"),
            Input("fct-perf-datatable",  "sort_by"),
            State("fct-perf-numeric-store", "data"),
            prevent_initial_call=True,
        )
        def highlight_best_row(sort_by, store_json):
            base = [
                {"if": {"filter_query": '{Strategy} contains "benchmark"'},
                 "backgroundColor": "#f5f5f5", "fontStyle": "italic"},
            ]
            if not sort_by or not store_json:
                return base

            sorted_col_label = sort_by[0]["column_id"]
            metric_key = _LABEL_TO_METRIC.get(sorted_col_label)
            if metric_key is None:
                return base

            try:
                records = pd.read_json(store_json, orient="records")
            except Exception:
                return base

            if metric_key not in records.columns:
                return base

            col_vals = pd.to_numeric(records[metric_key], errors="coerce")
            if col_vals.isna().all():
                return base

            best_idx = col_vals.idxmin() if metric_key in _LOWER_IS_BETTER else col_vals.idxmax()

            strategy_col = records.columns[0]
            best_strategy = str(records.loc[best_idx, strategy_col])

            return base + [
                {"if": {"filter_query": f'{{{strategy_col}}} = "{best_strategy}"'},
                 "backgroundColor": "#d4f5d4", "fontWeight": "bold"},
            ]

        # ---- A: annual bar chart ----
        @app.callback(
            Output("fct-annual-bar", "figure"),
            Input("fct-mode",        "value"),
            Input("fct-start-date",  "date"),
            Input("fct-bar-metric",  "value"),
        )
        def update_annual_bar(mode, start_date_str, bar_metric):
            mode       = mode or "ls"
            start_date = pd.Timestamp(start_date_str) if start_date_str else None
            bar_metric = bar_metric or "ann_ret"
            df = engine.get_annual_metrics(mode, metric=bar_metric, start_date=start_date)
            return build_annual_bar_figure(df, bar_metric)

        # ---- B: factor heatmap ----
        @app.callback(
            Output("fct-factor-heatmap", "figure"),
            Input("fct-hm-mode",   "value"),
            Input("fct-hm-metric", "value"),
            Input("fct-hm-gran",   "value"),
            Input("fct-hm-sort",   "value"),
            Input("fct-hm-window", "value"),
        )
        def update_heatmap(mode, metric, gran, sort_by, window_str):
            df = engine.get_factor_heatmap_data(
                mode or "ls",
                granularity=gran or "monthly",
                metric=metric or "return",
                window=int(window_str or 12),
            )
            return build_factor_heatmap(df, sort_by or "alpha", metric or "return")

        # ---- C: sector scores slider sync ----
        @app.callback(
            Output("fct-exp-slider",     "max"),
            Output("fct-exp-slider",     "marks"),
            Output("fct-exp-slider",     "value"),
            Input("fct-exp-gran",        "value"),
            prevent_initial_call=True,
        )
        def sync_exp_slider(gran):
            dates, marks = self._slider_config(gran or "monthly")
            n = len(dates)
            return n - 1, marks, n - 1

        # ---- C: sector scores update ----
        @app.callback(
            Output("fct-sector-scores",   "figure"),
            Output("fct-exp-date-label",  "children"),
            Input("fct-exp-slider",       "value"),
            Input("fct-exp-gran",         "value"),
            Input("fct-exp-sort-sectors", "value"),
            Input("fct-exp-sort-factors", "value"),
        )
        def update_sector_scores(idx, gran, sort_s, sort_f):
            date = self._date_from_slider(idx, gran or "monthly")
            df = engine.get_sector_scores(
                date,
                sort_sectors=sort_s or "alpha",
                sort_factors=sort_f or "alpha",
            )
            return build_sector_scores_figure(df), date.strftime("%b %Y")

        # ---- D: stock picker slider sync ----
        @app.callback(
            Output("fct-picker-slider",     "max"),
            Output("fct-picker-slider",     "marks"),
            Output("fct-picker-slider",     "value"),
            Input("fct-picker-gran",        "value"),
            prevent_initial_call=True,
        )
        def sync_picker_slider(gran):
            dates, marks = self._slider_config(gran or "monthly")
            n = len(dates)
            return n - 1, marks, n - 1

        # ---- D: stock picker update ----
        @app.callback(
            Output("fct-stock-table",        "children"),
            Output("fct-picker-date-label",  "children"),
            Input("fct-picker-factor",       "value"),
            Input("fct-picker-slider",       "value"),
            Input("fct-picker-gran",         "value"),
        )
        def update_stock_picker(factor, idx, gran):
            if not factor:
                return html.Div("Select a factor.", style={"color": "#999"}), ""
            date = self._date_from_slider(idx, gran or "monthly")
            ranked = engine.get_ranked_stocks(factor, date)
            return build_stock_table(ranked, factor), date.strftime("%b %Y")

        # ── Section B — Factor Profile radar ─────────────────────────────────

        @app.callback(
            Output("fct-radar-breakdown", "children"),
            Input("fct-radar-slider",     "value"),
            Input("fct-radar-ticker1",    "value"),
        )
        def update_radar_breakdown(slider_idx, ticker1):
            if not ticker1:
                return html.P("Select a ticker to see factor breakdown.",
                              style={"color": "#999", "fontSize": "13px"})
            date = self._avail_dates[int(slider_idx)]
            df   = engine.factor_scores_by_month.get(date, pd.DataFrame())
            if df.empty:
                return html.P("No data.", style={"color": "#999"})
            rows = df[df["ticker"] == ticker1]
            if rows.empty:
                return html.P("Ticker not found at this date.", style={"color": "#999"})
            row = rows.iloc[0]

            factors = _RADAR_FACTORS
            # correction perf : précalcul des 2 ranks en une passe
            pct_ranks  = df[factors].rank(pct=True) * 100
            desc_ranks = df[factors].rank(ascending=False)

            _cell = {"padding": "5px 12px", "fontSize": "13px",
                     "borderBottom": "1px solid #f0f0f0", "textAlign": "right"}
            _lbl  = {**_cell, "textAlign": "left", "color": "#888", "fontWeight": "bold"}
            _hdr  = {**_cell, "backgroundColor": "#f5f5f5", "fontWeight": "bold"}

            header = html.Tr([
                html.Th("", style={**_lbl, "backgroundColor": "#f5f5f5"}),
                *[html.Th(_FACTOR_LABELS[f], style=_hdr) for f in factors],
            ])

            z_cells, pct_cells, rank_cells = [], [], []
            for f in factors:
                z_val = row[f]
                if pd.isna(z_val):
                    z_cells.append(html.Td("n/a", style=_cell))
                    pct_cells.append(html.Td("n/a", style={**_cell, "fontWeight": "bold"}))
                    rank_cells.append(html.Td("n/a", style={**_cell, "color": "#bbb"}))
                else:
                    pct_val  = int(round(pct_ranks.loc[row.name, f]))
                    rank_val = int(desc_ranks.loc[row.name, f])
                    n_val    = int(df[f].dropna().shape[0])
                    sign     = "+" if float(z_val) >= 0 else ""
                    z_cells.append(html.Td(f"{sign}{float(z_val):.2f}", style=_cell))
                    pct_cells.append(html.Td(str(pct_val),
                                             style={**_cell, "fontWeight": "bold", "color": "#111"}))
                    rank_cells.append(html.Td(f"{rank_val} / {n_val}",
                                              style={**_cell, "color": "#bbb"}))

            return html.Table(
                [
                    html.Thead(header),
                    html.Tbody([
                        html.Tr([html.Td("Z-score",    style=_lbl), *z_cells]),
                        html.Tr([html.Td("Percentile", style=_lbl), *pct_cells]),
                        html.Tr([html.Td("Rank",       style=_lbl), *rank_cells]),
                    ]),
                ],
                style={"width": "100%", "borderCollapse": "collapse",
                       "border": "1px solid #e8e8e8"},
            )

        @app.callback(
            Output("fct-radar-interval", "disabled"),
            Output("fct-radar-play-btn", "children"),
            Input("fct-radar-play-btn",  "n_clicks"),
            State("fct-radar-interval",  "disabled"),
            prevent_initial_call=True,
        )
        def toggle_radar_play(n_clicks, is_disabled):
            if is_disabled:
                return False, "⏸"
            return True, "▶"

        @app.callback(
            Output("fct-radar-slider",   "value"),
            Output("fct-radar-interval", "disabled",  allow_duplicate=True),
            Output("fct-radar-play-btn", "children",  allow_duplicate=True),
            Input("fct-radar-interval",  "n_intervals"),
            State("fct-radar-slider",    "value"),
            State("fct-radar-slider",    "max"),
            prevent_initial_call=True,
        )
        def advance_radar_slider(n_intervals, current, max_val):
            nxt = int(current) + 1
            if nxt >= int(max_val):
                return int(max_val), True, "▶"
            return nxt, no_update, no_update

        @app.callback(
            Output("fct-radar-ticker1",    "options"),
            Output("fct-radar-ticker1",    "value"),
            Output("fct-radar-ticker2",    "options"),
            Output("fct-radar-ticker2",    "value"),
            Output("fct-radar-date-label", "children"),
            Input("fct-radar-slider",      "value"),
            State("fct-radar-ticker1",     "value"),   # correction #2 : persistance
            State("fct-radar-ticker2",     "value"),
        )
        def update_radar_tickers(slider_idx, current_t1, current_t2):
            date = self._avail_dates[int(slider_idx)]
            df   = engine.factor_scores_by_month.get(date, pd.DataFrame())
            tickers = sorted(df["ticker"].dropna().unique().tolist()) if not df.empty else []
            opts    = [{"label": t, "value": t} for t in tickers]
            ticker_set = set(tickers)

            # correction #2 : garder la sélection courante si toujours disponible
            new_t1 = current_t1 if current_t1 in ticker_set else (tickers[0] if tickers else None)
            new_t2 = current_t2 if current_t2 in ticker_set else None

            return opts, new_t1, opts, new_t2, date.strftime("%b %Y")

        @app.callback(
            Output("fct-radar-ticker2-container", "style"),
            Input("fct-radar-mode", "value"),
        )
        def toggle_radar_ticker2(mode):
            if mode == "vs_ticker":
                return {"display": "block"}
            return {"display": "none"}

        @app.callback(
            Output("fct-radar-chart",  "figure"),
            Input("fct-radar-slider",  "value"),
            Input("fct-radar-mode",    "value"),
            Input("fct-radar-ticker1", "value"),
            Input("fct-radar-ticker2", "value"),
        )
        def update_radar_chart(slider_idx, mode, ticker1, ticker2):
            date = self._avail_dates[int(slider_idx)]
            df   = engine.factor_scores_by_month.get(date, pd.DataFrame())
            return build_radar_figure(df, ticker1, mode or "single", ticker2)

        # ── Section E callbacks (only if regime_engine is wired up) ──────────
        if self.regime_engine is None:
            return

        re = self.regime_engine

        @app.callback(
            Output("fct-regime-bar", "figure"),
            Input("fct-regime-mode",       "value"),
            Input("fct-regime-metric",     "value"),
            Input("fct-regime-start-date", "date"),
        )
        def update_regime_section(mode, metric, start_date_str):
            mode       = mode   or "ls"
            metric     = metric or "ann_ret"
            start_date = pd.Timestamp(start_date_str) if start_date_str else None
            df = re.get_regime_factor_metrics(engine, mode=mode, metric=metric,
                                              start_date=start_date)
            return build_regime_bar_figure(df, metric)