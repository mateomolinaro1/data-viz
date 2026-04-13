"""
Market Overview tab.

Two sections:

1. Sector treemap  — animated cross-sectional snapshot at the selected date.
   - Level 1 : GICS sectors, sized by total market cap.
   - Level 2 : Top-20 stocks per sector + an "Others (N)" node for the rest.
   Controlled by:
     • Date slider with Play / Pause animation button.
     • Granularity radio (Daily / Weekly / Monthly / Yearly) — controls the
       dates available in the slider.
     • Performance colouring: when a period is selected each node is coloured
       green (positive return) or red (negative return) over that period.
       Falls back to sector-colour mode when performance data are unavailable.

2. Graph structure time series — 2×4 subplot grid tracking how the
   correlation network evolves across regimes:
       Nodes              | Edges
       Density            | Avg degree
       Avg weighted degree| Total edge weight
       Components         | Largest component
   Crisis bands (GFC / COVID / Rate shock) overlaid; a dotted vertical line
   marks the currently selected date.

Performance notes
-----------------
* Monthly universe snapshots are pre-computed at startup into a dict so
  that animation in Monthly mode is O(1) per frame.
* The graph-structure chart is rebuilt only when the date changes (it is
  lightweight — ~240 rows).
* Daily and Weekly animation is computed on demand and will be slower.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html, no_update
from plotly.subplots import make_subplots

from data_viz.data.data import DataManager
from data_viz.utils.config import Config
from data_viz.dashboard.volatility_heatmap import (
    build_volatility_heatmap_layout,
    register_volatility_heatmap_callbacks,
)
from data_viz.data.regime_engine import (
    RegimeEngine, EXTREME_REGIMES, REGIME_COLORS, _ALL_REGIMES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

_TOP_N: int = 50          # top stocks shown per sector in the treemap

# Crisis annotations — light grey across all charts
_CRISIS_GREY_FILL = "rgba(160,160,160,0.13)"
_CRISIS_GREY_LINE = "rgba(110,110,110,0.40)"

_CRISIS_PERIODS: list[dict] = [
    {"label": "GFC",        "start": "2008-09-01", "end": "2009-03-31"},
    {"label": "COVID",      "start": "2020-02-01", "end": "2020-04-30"},
    {"label": "Rate shock", "start": "2022-01-01", "end": "2022-12-31"},
]

# Number of trading days per granularity (used for period return)
_PERIOD_DAYS: dict[str, int] = {
    "daily": 1,
    "weekly": 5,
    "monthly": 21,
    "yearly": 252,
}

# Symmetric clamp for the performance colorscale (per granularity)
_PERF_CLAMP: dict[str, float] = {
    "daily":   0.03,
    "weekly":  0.07,
    "monthly": 0.12,
    "yearly":  0.40,
}

# Green → neutral → Red diverging colorscale for performance
_PERF_COLORSCALE = [
    [0.0, "#d73027"],   # strong red   (negative)
    [0.5, "#ffffbf"],   # light yellow (neutral)
    [1.0, "#1a9850"],   # strong green (positive)
]

# Stable qualitative palette for sector-colour mode (up to 15 sectors)
_PALETTE: list[str] = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2",
]

# Colours for the 8 graph-structure metric lines
_LINE_COLORS: list[str] = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#CCB974",
]

# Metrics shown in the 2×4 subplot grid
_GRAPH_METRICS: list[tuple[str, str, str]] = [
    ("n_nodes",                "Nodes",                ",.0f"),
    ("n_edges",                "Edges",                ",.0f"),
    ("density",                "Density",              ".4f"),
    ("avg_degree",             "Avg degree",           ".2f"),
    ("avg_weighted_degree",    "Avg weighted degree",  ".2f"),
    ("total_edge_weight",      "Total edge weight",    ",.1f"),
    ("n_components",           "Components",           ",.0f"),
    ("largest_component_size", "Largest component",    ",.0f"),
]


# ---------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------

def _get_dates_for_granularity(
    granularity: str,
    all_dates: list[pd.Timestamp],
) -> list[pd.Timestamp]:
    """Return one date per period (last trading day) for the given granularity."""
    if granularity == "daily":
        return all_dates
    df = pd.DataFrame({"date": all_dates})
    di = pd.DatetimeIndex(df["date"])
    if granularity == "weekly":
        df["period"] = di.to_period("W")
    elif granularity == "monthly":
        df["period"] = di.to_period("M")
    elif granularity == "yearly":
        df["year"] = di.year
        return df.groupby("year")["date"].last().tolist()
    else:
        return all_dates
    return df.groupby("period")["date"].last().tolist()


def _make_marks(dates: list[pd.Timestamp]) -> dict[int, str]:
    """Slider marks: one label per year, at January (or nearest available)."""
    seen_years: set[int] = set()
    marks: dict[int, str] = {}
    for i, d in enumerate(dates):
        yr = d.year
        if yr not in seen_years:
            seen_years.add(yr)
            marks[i] = str(yr)
    return marks


def _stat_badge(label: str, value: str) -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"fontSize": "11px", "color": "#888", "marginBottom": "2px"}),
            html.Div(value, style={"fontSize": "15px", "fontWeight": "bold"}),
        ],
        style={
            "padding": "10px 18px",
            "border": "1px solid #e0e0e0",
            "borderRadius": "6px",
            "backgroundColor": "#f9f9f9",
            "minWidth": "110px",
            "textAlign": "center",
        },
    )


# ---------------------------------------------------------------
# Treemap builder
# ---------------------------------------------------------------

def build_treemap_figure(
    universe: pd.DataFrame,
    sector_color_map: dict[str, str],
    perf: pd.Series | None = None,
    perf_clamp: float = 0.12,
    top_n: int = _TOP_N,
) -> go.Figure:
    """
    Build a two-level Treemap: sectors → top-N stocks + "Others".

    Parameters
    ----------
    universe : pd.DataFrame
        Columns: permno, ticker, gics_sector, market_cap.
        Pre-sorted by market_cap descending.
    sector_color_map : dict
        Maps sector name → hex colour (used when perf is None).
    perf : pd.Series | None
        Index = permno, values = cumulative return as decimal.
        When provided, nodes are coloured by performance (RdYlGn scale).
    perf_clamp : float
        Symmetric clamp for the performance colorscale (e.g. 0.12 = ±12 %).
    top_n : int
        Max stocks shown per sector (rest collapsed into "Others (N)").
    """
    if universe.empty:
        return go.Figure()

    # If performance data available, join to universe
    use_perf = perf is not None and not perf.empty
    if use_perf:
        perf_df = perf.rename("ret").reset_index()
        perf_df.columns = ["permno", "ret"]
        universe = universe.merge(perf_df, on="permno", how="left")

    total_mcap = universe["market_cap"].sum()

    labels:     list[str]   = []
    parents:    list[str]   = []
    values:     list[float] = []
    colors:     list        = []   # hex strings OR floats depending on mode
    customdata: list[list]  = []

    for sector, grp in universe.groupby("gics_sector", sort=False):
        grp = grp.sort_values("market_cap", ascending=False)
        sector_mcap  = float(grp["market_cap"].sum())
        sector_n     = len(grp)
        sector_color = sector_color_map.get(str(sector), "#aaaaaa")

        # Sector-level return = equal-weighted average across all stocks
        sector_ret = float(grp["ret"].mean()) if use_perf and "ret" in grp.columns else np.nan

        # ---- Sector node (root) ----
        labels.append(str(sector))
        parents.append("")
        values.append(0)      # branchvalues="remainder" → filled by children
        colors.append(sector_ret if use_perf else sector_color)
        customdata.append([
            sector_n,
            f"{sector_mcap / total_mcap * 100:.1f}%",
            f"${sector_mcap / 1e9:.0f} B",
            f"{sector_ret:+.1%}" if (use_perf and not np.isnan(sector_ret)) else "—",
        ])

        top  = grp.head(top_n)
        rest = grp.iloc[top_n:]

        # ---- Top-N stock nodes ----
        for _, row in top.iterrows():
            ticker = str(row["ticker"])
            mcap   = float(row["market_cap"])
            ret    = float(row["ret"]) if (use_perf and "ret" in row and pd.notna(row["ret"])) else np.nan

            labels.append(ticker)
            parents.append(str(sector))
            values.append(mcap)
            colors.append(ret if use_perf else sector_color)
            customdata.append([
                1,
                f"{mcap / total_mcap * 100:.2f}%",
                f"${mcap / 1e9:.1f} B",
                f"{ret:+.1%}" if (use_perf and not np.isnan(ret)) else "—",
            ])

        # ---- "Others" node ----
        if not rest.empty:
            others_mcap = float(rest["market_cap"].sum())
            n_others    = len(rest)
            others_ret  = float(rest["ret"].mean()) if use_perf and "ret" in rest.columns else np.nan

            labels.append(f"Others ({n_others})")
            parents.append(str(sector))
            values.append(others_mcap)
            colors.append(others_ret if use_perf else "#cccccc")
            customdata.append([
                n_others,
                f"{others_mcap / total_mcap * 100:.2f}%",
                f"${others_mcap / 1e9:.0f} B",
                f"{others_ret:+.1%}" if (use_perf and not np.isnan(others_ret)) else "—",
            ])

    # ---- Marker configuration ----
    if use_perf:
        marker = dict(
            colors=colors,
            colorscale=_PERF_COLORSCALE,
            cmin=-perf_clamp,
            cmid=0.0,
            cmax=perf_clamp,
            showscale=True,
            colorbar=dict(
                title=dict(text="Return", side="right"),
                tickformat="+.0%",
                thickness=14,
                len=0.6,
            ),
        )
    else:
        marker = dict(colors=colors)

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="remainder",
            customdata=customdata,
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Stocks : %{customdata[0]}<br>"
                "Weight : %{customdata[1]}<br>"
                "Mkt Cap: %{customdata[2]}<br>"
                "Return : %{customdata[3]}<br>"
                "<extra></extra>"
            ),
            marker=marker,
            textinfo="label+percent root",
            textfont=dict(size=12),
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=4, b=0),
        paper_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------
# Graph structure subplot builder
# ---------------------------------------------------------------

def build_graph_summary_figure(
    ts: pd.DataFrame,
    selected_date: pd.Timestamp,
    regime_series: pd.Series | None = None,
) -> go.Figure:
    """
    2×4 subplot grid of graph-structure metrics over time.
    A dotted vline marks selected_date in every panel.
    Regime shaded bands and grey crisis bands are overlaid when regime_series is given.
    """
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[m[1] for m in _GRAPH_METRICS],
        shared_xaxes=True,
        vertical_spacing=0.18,
        horizontal_spacing=0.07,
    )

    for idx, (col_name, title, fmt) in enumerate(_GRAPH_METRICS):
        row = idx // 4 + 1
        col = idx % 4 + 1
        if col_name not in ts.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=ts.index.tolist(),
                y=ts[col_name].tolist(),
                mode="lines",
                name=title,
                line=dict(width=1.5, color=_LINE_COLORS[idx]),
                hovertemplate=(
                    f"<b>{title}</b><br>%{{x|%Y-%m}}<br>"
                    f"%{{y:{fmt}}}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row, col=col,
        )

    # Regime bands
    if regime_series is not None:
        add_regime_bands_to_figure(fig, regime_series)

    # Crisis bands (light grey)
    for crisis in _CRISIS_PERIODS:
        try:
            t0 = pd.Timestamp(crisis["start"])
            t1 = pd.Timestamp(crisis["end"])
        except Exception:
            continue
        if ts.empty or t0 > ts.index[-1] or t1 < ts.index[0]:
            continue
        fig.add_vrect(
            x0=t0, x1=t1,
            fillcolor=_CRISIS_GREY_FILL, line_color=_CRISIS_GREY_LINE,
            line_width=1, layer="below",
            row="all", col="all",
        )
        fig.add_annotation(
            x=t0 + (t1 - t0) / 2, y=1.02, yref="paper",
            text=crisis["label"], showarrow=False,
            font=dict(size=9, color="#777"), xanchor="center",
        )

    # Selected-date vline
    if selected_date is not None and not ts.empty:
        if ts.index[0] <= selected_date <= ts.index[-1]:
            for idx in range(len(_GRAPH_METRICS)):
                fig.add_vline(
                    x=selected_date, line_dash="dot",
                    line_color="#333", line_width=1.2,
                    row=idx // 4 + 1, col=idx % 4 + 1,
                )

    fig.update_xaxes(showgrid=False, tickformat="%Y")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee", gridwidth=1)
    fig.update_layout(
        margin=dict(l=50, r=20, t=50, b=30),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        height=440,
    )
    return fig


# ---------------------------------------------------------------
# Regime figure builders
# ---------------------------------------------------------------

_CRISIS_REGIME = _CRISIS_PERIODS   # alias used in _add_crisis_bands


def _hex_rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB + alpha float to an rgba() string accepted by Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def add_regime_bands_to_figure(
    fig: go.Figure,
    regime_series: pd.Series,
    row: int | str = "all",
    col: int | str = "all",
    alpha: float = 0.15,
    highlighted_regime: str | None = None,
) -> None:
    """
    Add lightly shaded vrects for each extreme-regime run (monthly resolution).

    Parameters
    ----------
    row / col        : pass specific int for a subplot, or "all" for all panels.
    highlighted_regime : if set, this regime gets alpha×2 and all others get alpha×0.3.
    """
    monthly = (
        regime_series.dropna()
        .resample("M")
        .agg(lambda x: x.mode().iloc[0] if len(x) else np.nan)
        .dropna()
    )
    if monthly.empty:
        return

    def _vrect(t_start, t_end, label):
        if label not in EXTREME_REGIMES:
            return
        if highlighted_regime is not None:
            a = alpha * 2 if label == highlighted_regime else alpha * 0.3
        else:
            a = alpha
        kw: dict = dict(
            x0=t_start, x1=t_end,
            fillcolor=_hex_rgba(REGIME_COLORS[label], a),
            line_width=0, layer="below",
        )
        if row != "all":
            kw["row"] = row
            kw["col"] = col
        else:
            kw["row"] = "all"
            kw["col"] = "all"
        fig.add_vrect(**kw)

    prev, t0 = monthly.iloc[0], monthly.index[0]
    for date, label in monthly.items():
        if label != prev:
            _vrect(t0, date, prev)
            t0, prev = date, label
    _vrect(t0, monthly.index[-1], prev)


# Keep the private name as an alias for backward compatibility
_add_regime_bands = add_regime_bands_to_figure


def _add_crisis_bands(
    fig: go.Figure,
    row: int | str = "all",
    col: int | str = "all",
    x_min: pd.Timestamp | None = None,
    x_max: pd.Timestamp | None = None,
) -> None:
    """Add grey crisis vrects, skipping any that fall entirely outside [x_min, x_max]."""
    for c in _CRISIS_REGIME:
        t0 = pd.Timestamp(c["start"])
        t1 = pd.Timestamp(c["end"])
        if x_min is not None and t1 < x_min:
            continue
        if x_max is not None and t0 > x_max:
            continue
        kw: dict = dict(
            x0=t0, x1=t1,
            fillcolor=_CRISIS_GREY_FILL, line_color=_CRISIS_GREY_LINE,
            line_width=1, layer="below",
        )
        if row != "all":
            kw["row"] = row
            kw["col"] = col
        else:
            kw["row"] = "all"
            kw["col"] = "all"
        fig.add_vrect(**kw)
        fig.add_annotation(
            x=t0 + (t1 - t0) / 2,
            y=1.01, yref="paper", text=c["label"], showarrow=False,
            font=dict(size=9, color="#777"), xanchor="center",
        )


def build_regime_signal_figure(
    regime_engine: RegimeEngine,
    highlighted_regime: str | None = None,
) -> go.Figure:
    """
    Two-panel figure:
      Top    — trailing 12M CW market return + sector 10/90 band
      Bottom — trailing 12M annualised vol    + sector 10/90 band
    Regime colored bands and grey crisis bands overlaid.
    Horizontal dashed lines at 25th/75th percentile thresholds.
    X-axis clipped to first valid data point (after 252-day warm-up).
    """
    ret   = regime_engine.trailing_ret
    vol   = regime_engine.trailing_vol
    bands = regime_engine.cs_bands

    if ret is None or ret.empty:
        return go.Figure()

    # Clip to first non-NaN; fall back to first actual data point if data_start is None
    x_start = regime_engine.data_start
    if x_start is None and not ret.empty:
        x_start = ret.first_valid_index() or ret.index[0]
    if x_start is not None:
        ret   = ret.loc[ret.index >= x_start]
        vol   = vol.loc[vol.index >= x_start] if vol is not None else vol
        bands = bands.loc[bands.index >= x_start] if bands is not None else bands

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.06,
        subplot_titles=[
            "Trailing 12M Return (CW Index)",
            "Trailing 12M Annualised Volatility (CW Index)",
        ],
    )

    dates = ret.index.tolist()

    # ── Row 1: return ──────────────────────────────────────────
    if bands is not None and not bands.empty:
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=bands["ret_p90"].tolist() + bands["ret_p10"].tolist()[::-1],
            fill="toself", fillcolor="rgba(120,120,120,0.12)",
            line=dict(width=0), name="Sector range (min–max)", showlegend=True,
            hoverinfo="skip",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=ret.tolist(),
        mode="lines", name="CW Market",
        line=dict(color="#1a2744", width=2),
        hovertemplate="%{x|%d %b %Y}<br>%{y:+.1%}<extra>CW 12M Ret</extra>",
    ), row=1, col=1)

    for q, dash, lbl in [
        (regime_engine.ret_q25, "dot",  "Q25"),
        (regime_engine.ret_q75, "dash", "Q75"),
    ]:
        fig.add_hline(y=q, line_dash=dash, line_color="#888", line_width=1,
                      annotation_text=f" {lbl} {q:+.1%}",
                      annotation_position="right", annotation_font_size=9,
                      row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#aaa", line_width=1, row=1, col=1)

    # ── Row 2: vol ─────────────────────────────────────────────
    if bands is not None and not bands.empty:
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=bands["vol_p90"].tolist() + bands["vol_p10"].tolist()[::-1],
            fill="toself", fillcolor="rgba(120,120,120,0.12)",
            line=dict(width=0), name="Sector range (min–max)", showlegend=False,
            hoverinfo="skip",
        ), row=2, col=1)

    if vol is not None:
        fig.add_trace(go.Scatter(
            x=dates, y=vol.tolist(),
            mode="lines", name="CW Vol",
            line=dict(color="#c0392b", width=2),
            hovertemplate="%{x|%d %b %Y}<br>%{y:.1%}<extra>CW 12M Vol</extra>",
        ), row=2, col=1)

    for q, dash, lbl in [
        (regime_engine.vol_q25, "dot",  "Q25"),
        (regime_engine.vol_q75, "dash", "Q75"),
    ]:
        fig.add_hline(y=q, line_dash=dash, line_color="#888", line_width=1,
                      annotation_text=f" {lbl} {q:.1%}",
                      annotation_position="right", annotation_font_size=9,
                      row=2, col=1)

    x_end = ret.index[-1] if not ret.empty else None

    # ── Shared: regime + crisis bands ─────────────────────────
    if regime_engine.regime_series is not None:
        add_regime_bands_to_figure(
            fig, regime_engine.regime_series,
            highlighted_regime=highlighted_regime,
        )
    _add_crisis_bands(fig, x_min=x_start, x_max=x_end)

    # Lock both x-axes to data range (prevents vrects/annotations from expanding the axis)
    axis_range = [x_start, x_end] if (x_start is not None and x_end is not None) else None
    ax_dict = dict(showgrid=False, range=axis_range, autorange=(axis_range is None))
    fig.update_layout(
        height=520,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.04, x=0),
        margin=dict(l=60, r=100, t=60, b=40),
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        yaxis=dict(tickformat="+.0%"),
        yaxis2=dict(tickformat=".0%"),
        xaxis2=ax_dict,
        xaxis=ax_dict,
    )
    return fig


def build_regime_timeline_figure(regime_engine: RegimeEngine, freq: str) -> go.Figure:
    """
    Horizontal colored bar timeline — one Bar trace per regime (for legend interactivity).
    Bars for inactive periods have y=0 so only the active regime bar is visible.
    X-axis is synced to the signal chart range (clipped to data_start).
    """
    resampled = regime_engine.get_regime_at_frequency(freq)
    if resampled.empty:
        return go.Figure()

    all_dates = resampled.index.tolist()
    all_values = resampled.values

    fig = go.Figure()
    for regime in _ALL_REGIMES:
        if regime not in all_values:
            continue
        y_vals = [1 if r == regime else 0 for r in all_values]
        fig.add_trace(go.Bar(
            x=all_dates,
            y=y_vals,
            name=regime,
            marker_color=REGIME_COLORS.get(regime, "#e0e0e0"),
            marker_line_width=0,
            hovertemplate="%{x|%b %Y}<br>" + regime + "<extra></extra>",
            legendgroup=regime,
        ))

    # Sync x-range to signal chart (clipped to data_start, falling back to first data point)
    x_start = regime_engine.data_start
    if x_start is None and not resampled.empty:
        x_start = resampled.index[0]
    x_end = resampled.index[-1] if not resampled.empty else None
    axis_range = [x_start, x_end] if (x_start is not None and x_end is not None) else None

    fig.update_layout(
        barmode="overlay", bargap=0.0,
        yaxis=dict(visible=False, range=[0, 1.05]),
        xaxis=dict(
            showgrid=False, title="",
            range=axis_range,
            autorange=(axis_range is None),
        ),
        legend=dict(orientation="h", y=1.15, x=0, font=dict(size=11)),
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
        height=110,
    )
    return fig


def build_regime_monitor_figure(
    regime_engine: RegimeEngine,
    freq: str = "monthly",
    highlighted_regime: str | None = None,
) -> go.Figure:
    """
    Combined 3-row figure sharing a single x-axis:
      Row 1 — Trailing 12M Return (CW Index) + regime/crisis bands
      Row 2 — Trailing 12M Annualised Volatility (CW Index)
      Row 3 — Regime Timeline bar chart (at the requested frequency)

    All three rows share the same x-axis, so zooming any row zooms all three.
    """
    ret   = regime_engine.trailing_ret
    vol   = regime_engine.trailing_vol
    bands = regime_engine.cs_bands

    resampled = regime_engine.get_regime_at_frequency(freq)

    if (ret is None or ret.empty) and resampled.empty:
        return go.Figure()

    # ── Determine clipped x range ─────────────────────────────────────────────
    x_start = regime_engine.data_start
    if x_start is None and ret is not None and not ret.empty:
        x_start = ret.first_valid_index() or ret.index[0]
    if x_start is None and not resampled.empty:
        x_start = resampled.index[0]

    if ret is not None and not ret.empty and x_start is not None:
        ret   = ret.loc[ret.index >= x_start]
        vol   = vol.loc[vol.index >= x_start] if vol is not None else vol
        bands = bands.loc[bands.index >= x_start] if bands is not None else bands

    x_end = (ret.index[-1] if ret is not None and not ret.empty
             else (resampled.index[-1] if not resampled.empty else None))

    axis_range = ([x_start, x_end]
                  if (x_start is not None and x_end is not None) else None)

    # ── Build 3-row subplot ───────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.40, 0.40, 0.20],
        vertical_spacing=0.04,
        subplot_titles=[
            "Trailing 12M Return (CW Index)",
            "Trailing 12M Annualised Volatility (CW Index)",
            "",
        ],
    )

    # ── Row 1: return ─────────────────────────────────────────────────────────
    if ret is not None and not ret.empty:
        dates = ret.index.tolist()

        if bands is not None and not bands.empty:
            fig.add_trace(go.Scatter(
                x=dates + dates[::-1],
                y=bands["ret_p90"].tolist() + bands["ret_p10"].tolist()[::-1],
                fill="toself", fillcolor="rgba(120,120,120,0.12)",
                line=dict(width=0), name="Sector range (10–90%)", showlegend=True,
                hoverinfo="skip", legendgroup="bands",
            ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=dates, y=ret.tolist(),
            mode="lines", name="CW Market",
            line=dict(color="#1a2744", width=2),
            hovertemplate="%{x|%d %b %Y}<br>%{y:+.1%}<extra>CW 12M Ret</extra>",
        ), row=1, col=1)

        for q, dash, lbl in [
            (regime_engine.ret_q25, "dot",  "Q25"),
            (regime_engine.ret_q75, "dash", "Q75"),
        ]:
            if q is not None:
                fig.add_hline(y=q, line_dash=dash, line_color="#888", line_width=1,
                              annotation_text=f" {lbl} {q:+.1%}",
                              annotation_position="right", annotation_font_size=9,
                              row=1, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="#aaa", line_width=1, row=1, col=1)

        # ── Row 2: vol ────────────────────────────────────────────────────────
        if bands is not None and not bands.empty:
            fig.add_trace(go.Scatter(
                x=dates + dates[::-1],
                y=bands["vol_p90"].tolist() + bands["vol_p10"].tolist()[::-1],
                fill="toself", fillcolor="rgba(120,120,120,0.12)",
                line=dict(width=0), name="Sector range (10–90%)", showlegend=False,
                hoverinfo="skip", legendgroup="bands",
            ), row=2, col=1)

        if vol is not None and not vol.empty:
            fig.add_trace(go.Scatter(
                x=dates, y=vol.reindex(ret.index).tolist(),
                mode="lines", name="CW Vol",
                line=dict(color="#c0392b", width=2),
                hovertemplate="%{x|%d %b %Y}<br>%{y:.1%}<extra>CW 12M Vol</extra>",
            ), row=2, col=1)

        for q, dash, lbl in [
            (regime_engine.vol_q25, "dot",  "Q25"),
            (regime_engine.vol_q75, "dash", "Q75"),
        ]:
            if q is not None:
                fig.add_hline(y=q, line_dash=dash, line_color="#888", line_width=1,
                              annotation_text=f" {lbl} {q:.1%}",
                              annotation_position="right", annotation_font_size=9,
                              row=2, col=1)

        # ── Shared regime + crisis bands (rows 1 & 2) ─────────────────────────
        if regime_engine.regime_series is not None:
            add_regime_bands_to_figure(
                fig, regime_engine.regime_series,
                highlighted_regime=highlighted_regime,
                row="all", col="all",
            )
        _add_crisis_bands(fig, x_min=x_start, x_max=x_end)

    # ── Row 3: regime timeline bars ───────────────────────────────────────────
    if not resampled.empty:
        all_dates  = resampled.index.tolist()
        all_values = resampled.values
        for regime in _ALL_REGIMES:
            if regime not in all_values:
                continue
            y_vals = [1 if r == regime else 0 for r in all_values]
            fig.add_trace(go.Bar(
                x=all_dates, y=y_vals,
                name=regime,
                marker_color=REGIME_COLORS.get(regime, "#e0e0e0"),
                marker_line_width=0,
                hovertemplate="%{x|%b %Y}<br>" + regime + "<extra></extra>",
                legendgroup=regime,
                legendgrouptitle_text=None,
            ), row=3, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    ax_shared = dict(showgrid=False, range=axis_range, autorange=(axis_range is None))
    fig.update_layout(
        height=700,
        barmode="overlay",
        bargap=0.0,
        hovermode="x unified",
        # Legend below the figure to avoid covering subplot titles
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.06,
            xanchor="left", x=0,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=100, t=80, b=90),
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        yaxis=dict(tickformat="+.0%"),
        yaxis2=dict(tickformat=".0%"),
        yaxis3=dict(visible=False, range=[0, 1.05]),
        xaxis=ax_shared,
        xaxis2=ax_shared,
        xaxis3=ax_shared,
    )
    return fig


def build_transition_matrix_figure(tm: pd.DataFrame) -> go.Figure:
    """4×4 heatmap of transition probabilities with percentage annotations."""
    if tm.empty:
        return go.Figure()

    z      = tm.values
    labels = [r.replace("–", "–\n") for r in tm.index.tolist()]
    text   = [[f"{v:.0%}" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        text=text, texttemplate="%{text}",
        colorscale="Blues",
        zmin=0, zmax=1,
        colorbar=dict(title="P", tickformat=".0%", thickness=14, len=0.8),
        hovertemplate="From: <b>%{y}</b><br>To: <b>%{x}</b><br>P = %{z:.1%}<extra></extra>",
        xgap=2, ygap=2,
    ))
    fig.update_layout(
        yaxis=dict(autorange="reversed", showgrid=False),
        xaxis=dict(showgrid=False, side="bottom"),
        margin=dict(l=160, r=60, t=20, b=120),
        plot_bgcolor="white", paper_bgcolor="white",
        height=380,
    )
    return fig


def build_persistence_figure(rolling_persistence: pd.DataFrame) -> go.Figure:
    """Time series of rolling regime persistence (diagonal of rolling TM)."""
    if rolling_persistence.empty:
        return go.Figure()

    fig = go.Figure()
    for regime in EXTREME_REGIMES:
        if regime not in rolling_persistence.columns:
            continue
        fig.add_trace(go.Scatter(
            x=rolling_persistence.index.tolist(),
            y=rolling_persistence[regime].tolist(),
            mode="lines", name=regime,
            line=dict(color=REGIME_COLORS.get(regime, "#999"), width=2),
            hovertemplate=f"<b>{regime}</b><br>%{{x|%b %Y}}<br>P(stay) = %{{y:.1%}}<extra></extra>",
        ))
    fig.add_hline(y=0.5, line_dash="dot", line_color="#aaa", line_width=1)
    fig.update_layout(
        yaxis=dict(tickformat=".0%", range=[0, 1], title="P(same regime next day)"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=1.04, x=0),
        margin=dict(l=60, r=20, t=40, b=40),
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        hovermode="x unified", height=260,
    )
    return fig


def build_tm_timeseries_figure(rolling_tm: pd.DataFrame) -> go.Figure:
    """
    4-panel (2×2) figure showing rolling transition probabilities.
    Each panel = one source regime; 4 lines = 4 target regimes.
    """
    if rolling_tm.empty:
        return go.Figure()

    _SHORT = {
        "High Mom–High Vol": "HM–HV",
        "High Mom–Low Vol":  "HM–LV",
        "Low Mom–High Vol":  "LM–HV",
        "Low Mom–Low Vol":   "LM–LV",
    }
    titles = [f"From: {_SHORT.get(r, r)}" for r in EXTREME_REGIMES]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=titles,
        shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=0.14, horizontal_spacing=0.10,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for i, from_r in enumerate(EXTREME_REGIMES):
        r, c = positions[i]
        for to_r in EXTREME_REGIMES:
            key = (from_r, to_r)
            if key not in rolling_tm.columns:
                continue
            fig.add_trace(go.Scatter(
                x=rolling_tm.index.tolist(),
                y=rolling_tm[key].tolist(),
                mode="lines",
                name=to_r,
                legendgroup=to_r,
                showlegend=(i == 0),
                line=dict(color=REGIME_COLORS.get(to_r, "#999"), width=1.8),
                hovertemplate=(
                    f"<b>P({_SHORT.get(from_r, from_r)} → {_SHORT.get(to_r, to_r)})</b>"
                    "<br>%{x|%b %Y}<br>P = %{y:.1%}<extra></extra>"
                ),
            ), row=r, col=c)

    fig.update_yaxes(tickformat=".0%", range=[0, 1.05],
                     showgrid=True, gridcolor="#eeeeee")
    fig.update_xaxes(showgrid=False)
    fig.update_layout(
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=11),
                    title=dict(text="Target regime → ", font=dict(size=11))),
        margin=dict(l=60, r=20, t=50, b=100),
        plot_bgcolor="#fafafa", paper_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------
# MarketOverviewTab
# ---------------------------------------------------------------

class MarketOverviewTab:
    """
    Encapsulates layout and callbacks for the Market Overview tab.

    Usage
    -----
    tab = MarketOverviewTab(data_manager, config)
    tab.precompute()
    app.layout = ...tab.build_layout()...
    tab.register_callbacks(app)
    """

    def __init__(self, data_manager: DataManager, config: Config,
                 vol_df: pd.DataFrame | None = None,
                 regime_engine: RegimeEngine | None = None) -> None:
        self.dm = data_manager
        self.config = config
        self.vol_df: pd.DataFrame = vol_df if vol_df is not None else pd.DataFrame()
        self.regime_engine: RegimeEngine | None = regime_engine
        self.monthly_dates: list[pd.Timestamp] = []
        self.sector_color_map: dict[str, str] = {}
        # Pre-computed snapshots for fast monthly animation
        self._monthly_snapshots: dict[pd.Timestamp, pd.DataFrame] = {}
        # Pre-built static figure (regime monitor — rebuilt once at startup)
        self._regime_monitor_fig: go.Figure | None = None

    # ----------------------------------------------------------
    # Pre-computation
    # ----------------------------------------------------------

    def precompute(self) -> None:
        self._build_monthly_dates()
        self._build_sector_color_map()
        self._precompute_monthly_snapshots()
        self.dm.build_graph_summary_timeseries(
            monthly_dates=self.monthly_dates,
            threshold=float(self.config.layout_threshold_ref),
            min_periods=self.config.layout_min_periods,
        )
        if self.regime_engine is not None:
            self._regime_monitor_fig = build_regime_monitor_figure(self.regime_engine)
        logger.info("MarketOverviewTab pre-computation complete.")

    def _build_monthly_dates(self) -> None:
        if self.dm.dates is None:
            raise ValueError("data_manager.dates is not built.")
        self.monthly_dates = _get_dates_for_granularity("monthly", self.dm.dates)
        logger.info("Monthly dates: %d", len(self.monthly_dates))

    def _build_sector_color_map(self) -> None:
        if self.dm.network_data is None:
            return
        sectors = sorted(self.dm.network_data["gics_sector"].dropna().unique().tolist())
        self.sector_color_map = {
            s: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(sectors)
        }

    def _precompute_monthly_snapshots(self) -> None:
        """Pre-build universe snapshots for every monthly date for fast animation."""
        logger.info("Pre-computing %d monthly snapshots…", len(self.monthly_dates))
        for date in self.monthly_dates:
            self._monthly_snapshots[date] = self.dm.get_universe_snapshot(date)
        logger.info("Monthly snapshots ready.")

    # ----------------------------------------------------------
    # Layout
    # ----------------------------------------------------------

    def build_layout(self) -> html.Div:
        n = len(self.monthly_dates)
        marks = _make_marks(self.monthly_dates)
        initial_label = self.monthly_dates[-1].strftime("%b %Y") if self.monthly_dates else ""

        return html.Div(
            [
                dcc.Interval(
                    id="mkt-animation-interval",
                    interval=600,
                    n_intervals=0,
                    disabled=True,
                ),

                html.H3("Market Overview", style={"marginBottom": "6px"}),
                html.P(
                    "Macro snapshot of the investment universe: sector composition "
                    "and how the correlation structure evolves across market regimes.",
                    style={"color": "#666", "fontSize": "13px", "marginBottom": "20px"},
                ),

                # ---- Granularity selector ----
                html.Div(
                    [
                        html.Label(
                            "Time granularity",
                            style={"fontWeight": "bold", "marginRight": "14px"},
                        ),
                        dcc.RadioItems(
                            id="mkt-granularity",
                            options=[
                                {"label": "Daily",   "value": "daily"},
                                {"label": "Weekly",  "value": "weekly"},
                                {"label": "Monthly", "value": "monthly"},
                                {"label": "Yearly",  "value": "yearly"},
                            ],
                            value="monthly",
                            inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "20px", "fontSize": "13px"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "marginBottom": "16px",
                    },
                ),

                # ---- Date slider row ----
                html.Div(
                    [
                        html.Label(
                            "Snapshot date",
                            style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    dcc.Slider(
                                        id="mkt-date-slider",
                                        min=0,
                                        max=n - 1,
                                        step=1,
                                        value=n - 1,
                                        marks=marks,
                                        tooltip={"always_visible": False, "placement": "bottom"},
                                    ),
                                    style={"flex": "1", "minWidth": "0"},
                                ),
                                html.Button(
                                    "Play",
                                    id="mkt-play-btn",
                                    n_clicks=0,
                                    style={
                                        "padding": "7px 16px",
                                        "fontSize": "13px",
                                        "fontWeight": "bold",
                                        "cursor": "pointer",
                                        "flexShrink": "0",
                                    },
                                ),
                                html.Div(
                                    id="mkt-date-display",
                                    children=initial_label,
                                    style={
                                        "fontSize": "14px",
                                        "fontWeight": "bold",
                                        "color": "#333",
                                        "minWidth": "80px",
                                        "textAlign": "right",
                                        "flexShrink": "0",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "12px",
                            },
                        ),
                    ],
                    style={"marginBottom": "24px"},
                ),

                # Stats badges row — populated by callback
                html.Div(
                    id="mkt-stats-row",
                    style={
                        "display": "flex",
                        "gap": "12px",
                        "marginBottom": "16px",
                        "flexWrap": "wrap",
                    },
                ),

                # Treemap — figure updated in-place (avoids remount on every tick)
                dcc.Graph(
                    id="mkt-treemap",
                    config={"displayModeBar": False},
                    style={"height": "420px", "marginBottom": "36px"},
                ),

                # Graph structure time series
                html.Div(
                    [
                        html.H4(
                            "Correlation structure across regimes",
                            style={"marginBottom": "4px"},
                        ),
                        html.P(
                            "In crises, density and edge count spike while components "
                            "collapse — diversification breaks down as the whole market "
                            "moves together.",
                            style={"color": "#666", "fontSize": "13px", "marginBottom": "12px"},
                        ),
                        dcc.Graph(
                            id="mkt-graph-summary-ts",
                            config={"displayModeBar": False},
                            style={"height": "440px"},
                        ),
                    ],
                    style={"marginBottom": "36px"},
                ),

                # ── Regime Monitor ──────────────────────────────────────
                html.Hr(style={"borderColor": "#ddd", "marginBottom": "28px"}),
                *(self._build_regime_section() if self.regime_engine is not None
                  else [html.P("Regime engine not available.", style={"color": "#999"})]),

                # Volatility heatmap
                html.Hr(style={"borderColor": "#ddd", "marginBottom": "28px"}),
                *(
                    [build_volatility_heatmap_layout(self.vol_df)]
                    if not self.vol_df.empty
                    else [html.P("Volatility data not available.",
                                 style={"color": "#999"})]
                ),
            ],
            style={"padding": "10px"},
        )

    # ----------------------------------------------------------
    # Regime section layout helper
    # ----------------------------------------------------------

    def _build_regime_section(self) -> list:
        """Returns a list of Dash components for the Regime Monitor section."""
        n_m    = len(self.monthly_dates)
        last_m = n_m - 1
        marks_m = _make_marks(self.monthly_dates)

        return [
            html.H4("Market Regime Monitor", style={"marginBottom": "4px"}),
            html.P(
                "Trailing 12M performance and volatility of the cap-weighted market index. "
                "Regimes are defined by high/low market return × high/low market vol "
                "(25th/75th percentile thresholds). Colored bands = extreme regimes; "
                "grey bands = crisis episodes; grey area = sector return/vol range (10–90%). "
                "Double-click a regime in the timeline legend to isolate it. "
                "All three panels share the same x-axis — zoom any row to zoom all.",
                style={"color": "#666", "fontSize": "13px", "marginBottom": "8px"},
            ),

            # Timeline frequency selector
            html.Div([
                html.Label("Timeline frequency", style={"fontWeight": "bold", "marginRight": "12px"}),
                dcc.RadioItems(
                    id="regime-timeline-gran",
                    options=[
                        {"label": "Daily",   "value": "daily"},
                        {"label": "Weekly",  "value": "weekly"},
                        {"label": "Monthly", "value": "monthly"},
                        {"label": "Yearly",  "value": "yearly"},
                    ],
                    value="monthly", inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "16px", "fontSize": "13px"},
                ),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),

            # Combined 3-panel figure (return / vol / regime timeline — shared x-axis)
            dcc.Graph(
                id="regime-monitor",
                figure=self._regime_monitor_fig or go.Figure(),
                config={"displayModeBar": True, "scrollZoom": False},
                style={"height": "680px", "marginBottom": "28px"},
            ),

            # Panel 3 — Transition matrix
            html.H5("Regime Transition Matrix", style={"marginBottom": "6px"}),
            html.P(
                "Empirical P(regime t+1 | regime t) estimated over the selected trailing window. "
                "Use the Play button or drag the slider to animate through time.",
                style={"color": "#666", "fontSize": "13px", "marginBottom": "10px"},
            ),
            dcc.Interval(
                id="regime-tm-interval",
                interval=900,
                n_intervals=0,
                disabled=True,
            ),
            html.Div([
                html.Div([
                    html.Label("Trailing window",
                               style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                    dcc.RadioItems(
                        id="regime-tm-window",
                        options=[
                            {"label": "12M",  "value": "12"},
                            {"label": "24M",  "value": "24"},
                            {"label": "36M",  "value": "36"},
                            {"label": "Full", "value": "0"},
                        ],
                        value="24", inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "14px", "fontSize": "13px"},
                    ),
                ]),
                html.Div([
                    html.Label("End date",
                               style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"}),
                    html.Div([
                        html.Button(
                            "▶ Play", id="regime-tm-play-btn", n_clicks=0,
                            style={"padding": "5px 14px", "fontSize": "13px",
                                   "fontWeight": "bold", "cursor": "pointer",
                                   "flexShrink": "0"},
                        ),
                        html.Div(
                            dcc.Slider(
                                id="regime-tm-slider",
                                min=0, max=last_m, step=1, value=last_m,
                                marks=marks_m,
                                tooltip={"always_visible": False, "placement": "bottom"},
                            ),
                            style={"flex": "1"},
                        ),
                        html.Div(
                            id="regime-tm-date-label",
                            children=self.monthly_dates[-1].strftime("%b %Y") if self.monthly_dates else "",
                            style={"fontSize": "13px", "fontWeight": "bold",
                                   "minWidth": "80px", "textAlign": "right", "flexShrink": "0"},
                        ),
                    ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "gap": "40px", "alignItems": "flex-end",
                      "flexWrap": "wrap", "marginBottom": "12px"}),

            html.Div([
                # Heatmap (left)
                html.Div(
                    dcc.Graph(id="regime-tm-matrix",
                              config={"displayModeBar": False},
                              style={"height": "380px"}),
                    style={"flex": "1", "minWidth": "340px"},
                ),
                # Stats table (right)
                html.Div(
                    html.Div(id="regime-tm-stats"),
                    style={"flex": "0 0 300px", "paddingLeft": "20px",
                           "paddingTop": "20px", "fontSize": "13px"},
                ),
            ], style={"display": "flex", "alignItems": "flex-start", "marginBottom": "20px"}),

            # Rolling transition probability time series (16 lines, 4 subplots)
            html.H5("Rolling Transition Probabilities", style={"marginBottom": "4px"}),
            html.P(
                "All 16 P(from → to) time series estimated with the selected trailing window. "
                "Each panel shows transitions out of one regime.",
                style={"color": "#666", "fontSize": "13px", "marginBottom": "8px"},
            ),
            dcc.Graph(
                id="regime-tm-timeseries",
                config={"displayModeBar": False},
                style={"height": "460px", "marginBottom": "28px"},
            ),
        ]

    # ----------------------------------------------------------
    # Callbacks
    # ----------------------------------------------------------

    def register_callbacks(self, app) -> None:
        dm = self.dm
        monthly_snapshots = self._monthly_snapshots
        monthly_dates = self.monthly_dates
        sector_color_map = self.sector_color_map
        re = self.regime_engine

        # Volatility heatmap callbacks (embedded section)
        if not self.vol_df.empty:
            register_volatility_heatmap_callbacks(app, self.vol_df)

        # ---- 1. Toggle play / pause ----
        @app.callback(
            Output("mkt-animation-interval", "disabled"),
            Output("mkt-play-btn", "children"),
            Input("mkt-play-btn", "n_clicks"),
            State("mkt-animation-interval", "disabled"),
            prevent_initial_call=True,
        )
        def toggle_animation(n_clicks, is_disabled):
            now_playing = bool(is_disabled)
            return not now_playing, "Pause" if now_playing else "Play"

        # ---- 2. Sync slider max & marks when granularity changes ----
        #        (slider value is NOT updated here — update_overview handles clamping)
        @app.callback(
            Output("mkt-date-slider", "max"),
            Output("mkt-date-slider", "marks"),
            Input("mkt-granularity", "value"),
            prevent_initial_call=True,
        )
        def sync_slider_to_granularity(granularity):
            dates = _get_dates_for_granularity(granularity, dm.dates)
            n = len(dates)
            return n - 1, _make_marks(dates)

        # ---- 3. Main update: slider drag, interval tick, or granularity change ----
        @app.callback(
            Output("mkt-stats-row", "children"),
            Output("mkt-treemap", "figure"),
            Output("mkt-graph-summary-ts", "figure"),
            Output("mkt-date-display", "children"),
            Output("mkt-date-slider", "value"),
            Input("mkt-date-slider", "value"),
            Input("mkt-animation-interval", "n_intervals"),
            Input("mkt-granularity", "value"),
            State("mkt-animation-interval", "disabled"),
        )
        def update_overview(date_idx, n_intervals, granularity, interval_disabled):
            triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
            gran = granularity or "monthly"
            dates = _get_dates_for_granularity(gran, dm.dates)
            n = len(dates)

            is_animating = (triggered == "mkt-animation-interval" and not interval_disabled)

            if is_animating:
                date_idx = (int(date_idx) + 1) % n
            else:
                date_idx = min(int(date_idx), n - 1)

            selected_date = dates[date_idx]
            date_label = selected_date.strftime("%b %Y")

            # ---- Universe snapshot (O(1) for monthly, on-demand otherwise) ----
            if gran == "monthly" and selected_date in monthly_snapshots:
                universe = monthly_snapshots[selected_date]
            else:
                universe = dm.get_universe_snapshot(selected_date)

            # ---- Period performance for colouring ----
            n_days = _PERIOD_DAYS.get(gran, 21)
            clamp  = _PERF_CLAMP.get(gran, 0.12)
            try:
                perf = dm.get_period_returns(selected_date, n_days)
            except Exception:
                perf = None

            # ---- Stats badges ----
            if universe.empty:
                stats_children = [html.Div("No data for this date.", style={"color": "#999"})]
                treemap_fig = go.Figure()
            else:
                total_mcap = universe["market_cap"].sum()
                n_stocks   = len(universe)
                n_sectors  = universe["gics_sector"].nunique()
                stats_children = [
                    _stat_badge("Date",             date_label),
                    _stat_badge("Stocks",           f"{n_stocks:,}"),
                    _stat_badge("Sectors",          str(n_sectors)),
                    _stat_badge("Universe Mkt Cap", f"${total_mcap / 1e12:.2f} T"),
                ]
                treemap_fig = build_treemap_figure(
                    universe=universe,
                    sector_color_map=sector_color_map,
                    perf=perf,
                    perf_clamp=clamp,
                )

            # ---- Graph structure time series ----
            # Skip expensive rebuild during animation — only update on manual seek
            # or granularity change.
            if is_animating:
                graph_fig = no_update
            else:
                ts = dm.graph_summary_timeseries
                if ts is not None and not ts.empty:
                    snap_idx = min(
                        ts.index.searchsorted(selected_date, side="left"),
                        len(ts.index) - 1,
                    )
                    graph_fig = build_graph_summary_figure(
                        ts, ts.index[snap_idx],
                        regime_series=re.regime_series if re is not None else None,
                    )
                else:
                    graph_fig = go.Figure()

            return stats_children, treemap_fig, graph_fig, date_label, date_idx

        # ── Regime callbacks (only registered if regime_engine is available) ──
        if re is None:
            return

        # ---- Combined regime monitor (freq change + legend double-click) ----
        @app.callback(
            Output("regime-monitor", "figure"),
            Input("regime-timeline-gran", "value"),
            Input("regime-monitor", "restyleData"),
            State("regime-monitor", "figure"),
            prevent_initial_call=True,
        )
        def update_regime_monitor(freq, restyle_data, current_fig):
            triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]
            highlighted = None

            # Legend toggle: read visibility state from the current figure
            if triggered_id == "regime-monitor" and restyle_data and current_fig:
                changes, indices = restyle_data
                if "visible" in changes:
                    traces = current_fig.get("data", [])
                    vis_state = {i: trace.get("visible", True)
                                 for i, trace in enumerate(traces)}
                    for idx, vis in zip(indices, changes["visible"]):
                        vis_state[idx] = vis
                    visible_regimes = [
                        traces[i].get("name", "")
                        for i, v in vis_state.items()
                        if v is True and i < len(traces)
                        and traces[i].get("name", "") in REGIME_COLORS
                    ]
                    highlighted = visible_regimes[0] if len(visible_regimes) == 1 else None

            return build_regime_monitor_figure(re, freq or "monthly", highlighted_regime=highlighted)

        # ---- TM play / pause ----
        @app.callback(
            Output("regime-tm-interval",  "disabled"),
            Output("regime-tm-play-btn",  "children"),
            Input("regime-tm-play-btn",   "n_clicks"),
            State("regime-tm-interval",   "disabled"),
            prevent_initial_call=True,
        )
        def toggle_tm_play(n_clicks, is_disabled):
            now_playing = bool(is_disabled)
            return not now_playing, "⏸ Pause" if now_playing else "▶ Play"

        # ---- Transition matrix + stats + TM time series (+ interval advance) ----
        # The interval advance is merged here so the next tick cannot fire until
        # the heavy TM computation finishes — no separate advance-slider callback.
        @app.callback(
            Output("regime-tm-matrix",     "figure"),
            Output("regime-tm-stats",       "children"),
            Output("regime-tm-timeseries",  "figure"),
            Output("regime-tm-date-label",  "children"),
            Output("regime-tm-slider",      "value", allow_duplicate=True),
            Input("regime-tm-interval",     "n_intervals"),
            Input("regime-tm-slider",       "value"),
            Input("regime-tm-window",       "value"),
            State("regime-tm-interval",     "disabled"),
            prevent_initial_call=True,
        )
        def update_transition_matrix(n_intervals, slider_val, window_str, is_disabled):
            triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
            max_idx   = len(monthly_dates) - 1

            if triggered == "regime-tm-interval" and not is_disabled:
                idx = (int(slider_val or 0) + 1) % (max_idx + 1)
            else:
                idx = min(int(slider_val or 0), max_idx)

            end_date  = monthly_dates[idx]
            window_mo = int(window_str or 24) or None   # "0" → full history

            tm      = re.get_transition_matrix(end_date=end_date,
                                               window_months=window_mo,
                                               extreme_only=True)
            stats   = re.get_regime_stats(end_date=end_date, window_months=window_mo)
            roll_ts = re.get_rolling_transition_timeseries(window_months=window_mo or 24)

            # Stats table (all regimes present in window)
            if stats.empty:
                stats_div = html.P("No data.", style={"color": "#999"})
            else:
                rows = []
                for regime, row in stats.iterrows():
                    color      = REGIME_COLORS.get(regime, "#aaa")
                    is_extreme = regime in EXTREME_REGIMES
                    rows.append(html.Tr([
                        html.Td(html.Span("■ ", style={"color": color, "fontSize": "15px"})),
                        html.Td(regime,
                                style={"fontWeight": "700" if is_extreme else "400",
                                       "paddingRight": "10px", "fontSize": "11px",
                                       "color": "#222" if is_extreme else "#666"}),
                        html.Td(f"{row['% of time']:.0%}",
                                style={"textAlign": "right", "paddingRight": "10px",
                                       "fontWeight": "600" if is_extreme else "400"}),
                        html.Td(f"{row['Avg duration (days)']:.0f}d",
                                style={"textAlign": "right", "color": "#666"}),
                    ]))
                stats_div = html.Div([
                    html.P("Regime statistics (selected window)",
                           style={"fontWeight": "700", "marginBottom": "8px",
                                  "fontSize": "12px", "color": "#555"}),
                    html.Table(
                        [html.Thead(html.Tr([
                            html.Th(""), html.Th("Regime"),
                            html.Th("% time"), html.Th("Avg dur."),
                        ], style={"fontSize": "10px", "color": "#888"})),
                         html.Tbody(rows)],
                        style={"borderCollapse": "collapse", "width": "100%"},
                    ),
                ])

            return (
                build_transition_matrix_figure(tm),
                stats_div,
                build_tm_timeseries_figure(roll_ts),
                end_date.strftime("%b %Y"),
                idx,
            )