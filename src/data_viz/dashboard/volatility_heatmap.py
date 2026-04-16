"""
Volatility regime timeline heatmap.

Displays monthly realized volatility (annualized) by GICS sector across time,
useful for spotting stress regimes (2008 GFC, 2020 COVID, 2022 rate shock).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html


# ---------------------------------------------------------------
# Hard-coded crisis / stress period annotations
# ---------------------------------------------------------------
_CRISIS_PERIODS: list[dict] = [
    {
        "label": "GFC",
        "start": "2008-09",
        "end": "2009-03",
        "fill": "rgba(220, 60, 60, 0.12)",
        "line": "rgba(220, 60, 60, 0.45)",
    },
    {
        "label": "COVID",
        "start": "2020-02",
        "end": "2020-04",
        "fill": "rgba(255, 140, 0, 0.12)",
        "line": "rgba(255, 140, 0, 0.45)",
    },
    {
        "label": "Rate shock",
        "start": "2022-01",
        "end": "2022-12",
        "fill": "rgba(100, 60, 180, 0.12)",
        "line": "rgba(100, 60, 180, 0.45)",
    },
]


# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------

def _month_marks(months: pd.DatetimeIndex) -> dict[int, str]:
    """Return {index: 'YYYY'} slider marks at each January."""
    return {i: str(m.year) for i, m in enumerate(months) if m.month == 1}


def _apply_normalization(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Apply one of three normalization modes to a vol DataFrame (months × sectors).

    raw           — as-is (annualized realized vol)
    zscore        — z-score within each sector's own history
    cross_section — divide each month row by its cross-sectional mean
    """
    if mode == "zscore":
        return (df - df.mean()) / df.std().replace(0.0, np.nan)
    if mode == "cross_section":
        row_mean = df.mean(axis=1).replace(0.0, np.nan)
        return df.div(row_mean, axis=0)
    return df  # "raw"


def _crisis_x_bounds(
    months: pd.DatetimeIndex,
    start_str: str,
    end_str: str,
) -> tuple[float, float] | None:
    """
    Convert a crisis period (month strings like "2008-09") to float x-axis
    bounds suitable for add_vrect on a numeric-x heatmap.
    Returns None when the period lies entirely outside the visible slice.
    """
    try:
        t0 = pd.Timestamp(start_str)
        t1 = pd.Timestamp(end_str) + pd.offsets.MonthEnd(0)
    except Exception:
        return None

    if months.empty or t0 > months[-1] or t1 < months[0]:
        return None

    idx = np.where((months >= t0) & (months <= t1))[0]
    if len(idx) == 0:
        return None

    # Extend half a cell on each side so the shaded band covers full columns
    return float(idx[0]) - 0.5, float(idx[-1]) + 0.5


# ---------------------------------------------------------------
# Layout
# ---------------------------------------------------------------

def build_volatility_heatmap_layout(vol_df: pd.DataFrame) -> html.Div:
    """
    Build the Dash layout for the volatility heatmap tab.

    Parameters
    ----------
    vol_df : pd.DataFrame
        Output of DataManager.get_sector_vol_heatmap_data().
        Index: month-end timestamps. Columns: GICS sector names.
    """
    months = vol_df.index
    n = len(months)
    all_sectors = sorted(vol_df.columns.tolist())
    marks = _month_marks(months)

    return html.Div(
        [
            html.H3("Volatility Regime Timeline", style={"marginBottom": "6px"}),
            html.P(
                "Monthly realized volatility (annualized std of daily returns) "
                "by GICS sector — equal-weighted portfolio within each sector. "
                "Shaded bands mark the 2008 GFC, 2020 COVID, and 2022 rate shock.",
                style={"color": "#666", "fontSize": "13px", "marginBottom": "20px"},
            ),

            # ---- Controls ----
            html.Div(
                [
                    # Date range slider
                    html.Div(
                        [
                            html.Label(
                                "Date range",
                                style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"},
                            ),
                            dcc.RangeSlider(
                                id="vol-date-range-slider",
                                min=0,
                                max=n - 1,
                                step=1,
                                value=[0, n - 1],
                                marks=marks,
                                allowCross=False,
                                tooltip={"always_visible": False, "placement": "bottom"},
                            ),
                        ],
                        style={"flex": "3", "minWidth": "280px"},
                    ),

                    # Sector multi-select
                    html.Div(
                        [
                            html.Label(
                                "Sectors",
                                style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"},
                            ),
                            dcc.Dropdown(
                                id="vol-sector-filter",
                                options=[{"label": s, "value": s} for s in all_sectors],
                                value=all_sectors,
                                multi=True,
                                placeholder="Select sectors…",
                                style={"fontSize": "13px"},
                            ),
                        ],
                        style={"flex": "2", "minWidth": "220px"},
                    ),

                    # Normalization
                    html.Div(
                        [
                            html.Label(
                                "Normalization",
                                style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"},
                            ),
                            dcc.Dropdown(
                                id="vol-normalization",
                                options=[
                                    {"label": "Raw annualized vol", "value": "raw"},
                                    {"label": "Z-score (within sector)", "value": "zscore"},
                                    {"label": "Cross-section relative", "value": "cross_section"},
                                ],
                                value="raw",
                                clearable=False,
                                style={"fontSize": "13px"},
                            ),
                        ],
                        style={"flex": "1", "minWidth": "180px"},
                    ),

                    # Sort
                    html.Div(
                        [
                            html.Label(
                                "Sort sectors by",
                                style={"fontWeight": "bold", "display": "block", "marginBottom": "4px"},
                            ),
                            dcc.Dropdown(
                                id="vol-sort-sectors",
                                options=[
                                    {"label": "Name (A→Z)", "value": "alpha"},
                                    {"label": "Avg vol (high→low)", "value": "avg_vol"},
                                ],
                                value="alpha",
                                clearable=False,
                                style={"fontSize": "13px"},
                            ),
                        ],
                        style={"flex": "1", "minWidth": "160px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "24px",
                    "alignItems": "flex-end",
                    "flexWrap": "wrap",
                    "marginBottom": "20px",
                },
            ),

            # ---- Heatmap ----
            dcc.Graph(
                id="vol-heatmap",
                config={"displayModeBar": True, "scrollZoom": False},
                style={"height": "520px"},
            ),
        ],
        style={"padding": "10px"},
    )


# ---------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------

def register_volatility_heatmap_callbacks(app, vol_df: pd.DataFrame) -> None:
    """
    Register all callbacks for the volatility heatmap.
    Must be called after the Dash app layout has been assigned.

    Parameters
    ----------
    app     : Dash application instance
    vol_df  : Output of DataManager.get_sector_vol_heatmap_data()
    """
    all_months: pd.DatetimeIndex = vol_df.index

    @app.callback(
        Output("vol-heatmap", "figure"),
        Input("vol-date-range-slider", "value"),
        Input("vol-sector-filter", "value"),
        Input("vol-normalization", "value"),
        Input("vol-sort-sectors", "value"),
    )
    def update_heatmap(
        date_range: list[int],
        selected_sectors: list[str],
        norm_mode: str,
        sort_mode: str,
    ) -> go.Figure:

        # ---- Slice on date range ----
        start_idx, end_idx = int(date_range[0]), int(date_range[1])
        months = all_months[start_idx: end_idx + 1]
        df = vol_df.loc[months]

        # ---- Filter sectors ----
        if not selected_sectors:
            selected_sectors = vol_df.columns.tolist()
        cols = [s for s in selected_sectors if s in df.columns]
        if not cols:
            return go.Figure()
        df = df[cols]

        # ---- Normalize ----
        df_display = _apply_normalization(df, norm_mode)

        # ---- Sort sector order ----
        if sort_mode == "avg_vol":
            order = df_display.mean(axis=0).sort_values(ascending=False).index.tolist()
        else:
            order = sorted(df_display.columns.tolist())
        df_display = df_display[order]

        # Raw vol always shown in tooltip regardless of current normalization
        df_raw = df[order]

        # ---- Build arrays ----
        # z shape: (n_sectors, n_months) — rows = y-axis, cols = x-axis
        z = df_display.T.to_numpy()
        z_raw = df_raw.T.to_numpy()
        n_sectors, n_months = z.shape
        x_indices = list(range(n_months))
        x_labels = [m.strftime("%Y-%m") for m in months]

        # customdata[sector_i, month_j] = [month_label, formatted_raw_vol]
        customdata = np.empty((n_sectors, n_months, 2), dtype=object)
        for si in range(n_sectors):
            for ti in range(n_months):
                raw_val = z_raw[si, ti]
                customdata[si, ti, 0] = x_labels[ti]
                customdata[si, ti, 1] = (
                    f"{raw_val:.1%}" if (raw_val is not None and not np.isnan(raw_val)) else "n/a"
                )

        # ---- Colorscale & colorbar config ----
        if norm_mode == "zscore":
            colorscale = "RdYlGn_r"
            colorbar_title = "Z-score"
            zmid = 0.0
            zmin = zmax = None
            tickfmt = ".2f"
        elif norm_mode == "cross_section":
            colorscale = "RdYlGn_r"
            colorbar_title = "× avg"
            zmid = 1.0
            zmin = zmax = None
            tickfmt = ".2f"
        else:  # raw
            colorscale = "YlOrRd"
            colorbar_title = "Ann. vol"
            zmid = None
            zmin = 0.0
            zmax = None
            tickfmt = ".0%"

        # ---- Figure ----
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=x_indices,
                y=order,
                customdata=customdata,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Month: %{customdata[0]}<br>"
                    "Raw vol: %{customdata[1]}<br>"
                    "<extra></extra>"
                ),
                colorscale=colorscale,
                zmid=zmid,
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(
                    title=dict(text=colorbar_title, side="right"),
                    thickness=16,
                    len=0.8,
                    tickformat=tickfmt,
                ),
                xgap=0.5,
                ygap=1,
            )
        )

        # ---- Crisis annotations ----
        for crisis in _CRISIS_PERIODS:
            bounds = _crisis_x_bounds(months, crisis["start"], crisis["end"])
            if bounds is None:
                continue
            x0, x1 = bounds
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=crisis["fill"],
                line_width=1,
                line_color=crisis["line"],
                annotation_text=crisis["label"],
                annotation_position="top left",
                annotation_font_size=10,
                annotation_font_color="#333",
                layer="below",
            )

        # ---- Axis ticks: one label per January ----
        tick_vals, tick_texts = [], []
        for i, m in enumerate(months):
            if m.month == 1:
                tick_vals.append(i)
                tick_texts.append(str(m.year))

        fig.update_layout(
            xaxis=dict(
                tickvals=tick_vals,
                ticktext=tick_texts,
                showgrid=False,
                zeroline=False,
                title="",
            ),
            yaxis=dict(
                showgrid=False,
                autorange="reversed",  # top = first in order
                title="",
            ),
            margin=dict(l=220, r=100, t=30, b=50),
            plot_bgcolor="#fafafa",
            paper_bgcolor="white",
            hoverlabel=dict(bgcolor="white", font_size=13),
        )

        return fig