"""
Heatmap component — uses ticker labels and mcap_rank for stock selection.
"""

import pandas as pd
import plotly.graph_objects as go


def build_correlation_heatmap(
    mkt_data: pd.DataFrame,
    date_start: str | None = None,
    date_end: str | None = None,
    top_n: int = 25,
    colorscale: str = "Plasma",
    ret_col: str = "ret",
    date_col: str = "date",
    label_col: str = "ticker",
    rank_col: str = "mcap_rank",
) -> go.Figure:
    df = mkt_data[[date_col, label_col, ret_col, rank_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if date_start:
        df = df[df[date_col] >= pd.to_datetime(date_start)]
    if date_end:
        df = df[df[date_col] <= pd.to_datetime(date_end)]

    if df.empty:
        return _empty("Aucune donnée pour cette période")

    # Top N par market cap (rang moyen le plus bas = plus grande capi)
    top_tickers = (
        df.groupby(label_col)[rank_col].mean()
        .nsmallest(top_n).index.tolist()
    )
    df = df[df[label_col].isin(top_tickers)]

    pivot = df.pivot_table(index=date_col, columns=label_col, values=ret_col)
    pivot = pivot.dropna(axis=1, thresh=int(len(pivot) * 0.5))
    corr = pivot.corr()

    labels = corr.columns.tolist()
    z = corr.values

    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        zmin=-1, zmax=1,
        colorscale=colorscale,
        xgap=2, ygap=2,
        colorbar=dict(
            title="ρ", thickness=12, len=0.8, outlinewidth=0,
            tickvals=[-1, -0.5, 0, 0.5, 1],
        ),
        hovertemplate="<b>%{y} × %{x}</b><br>ρ = %{z:.3f}<extra></extra>",
    ))

    period = _period(df[date_col])
    n = len(labels)
    size = max(500, min(900, n * 28 + 160))

    fig.update_layout(
        title=dict(text=f"Matrice de corrélation — {period}", x=0.5, xanchor="center",
                   font=dict(size=15, family="Inter, sans-serif", color="#e2e8f0")),
        width=size, height=size,
        paper_bgcolor="#1c1f2b",
        plot_bgcolor="#1c1f2b",
        font=dict(color="#94a3b8", size=10),
        xaxis=dict(tickangle=-45, side="bottom", tickfont=dict(size=9)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
        margin=dict(l=80, r=40, t=60, b=100),
    )
    return fig


def build_returns_heatmap(
    mkt_data: pd.DataFrame,
    date_start: str | None = None,
    date_end: str | None = None,
    top_n: int = 25,
    colorscale: str = "RdYlGn",
    ret_col: str = "ret",
    date_col: str = "date",
    label_col: str = "ticker",
    rank_col: str = "mcap_rank",
    freq: str = "ME",
) -> go.Figure:
    df = mkt_data[[date_col, label_col, ret_col, rank_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if date_start:
        df = df[df[date_col] >= pd.to_datetime(date_start)]
    if date_end:
        df = df[df[date_col] <= pd.to_datetime(date_end)]

    if df.empty:
        return _empty("Aucune donnée pour cette période")

    top_tickers = (
        df.groupby(label_col)[rank_col].mean()
        .nsmallest(top_n).index.tolist()
    )
    df = df[df[label_col].isin(top_tickers)]

    pivot = df.pivot_table(index=date_col, columns=label_col, values=ret_col)
    pivot = pivot.resample(freq).sum() * 100

    fig = go.Figure(go.Heatmap(
        z=pivot.T.values,
        x=[d.strftime("%b %Y") for d in pivot.index],
        y=pivot.columns.tolist(),
        zmid=0,
        colorscale=colorscale,
        xgap=1, ygap=1,
        colorbar=dict(title="Ret (%)", thickness=12, outlinewidth=0, ticksuffix="%"),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Rendement : %{z:.2f}%<extra></extra>",
    ))

    period = _period(df[date_col])
    fig.update_layout(
        title=dict(text=f"Rendements mensuels — {period}", x=0.5, xanchor="center",
                   font=dict(size=15, family="Inter, sans-serif", color="#e2e8f0")),
        paper_bgcolor="#1c1f2b",
        plot_bgcolor="#1c1f2b",
        font=dict(color="#94a3b8", size=10),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
        margin=dict(l=80, r=40, t=60, b=100),
        height=max(450, top_n * 20 + 160),
    )
    return fig


def _period(dates: pd.Series) -> str:
    return f"{dates.min().strftime('%b %Y')} → {dates.max().strftime('%b %Y')}"


def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=16, color="#64748b"))
    fig.update_layout(paper_bgcolor="#1c1f2b", plot_bgcolor="#1c1f2b",
                      xaxis=dict(visible=False), yaxis=dict(visible=False), height=400)
    return fig
