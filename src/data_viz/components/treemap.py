"""
Treemap component — style Finviz/TradingView
Taille = market cap, Couleur = rendement, Groupé par secteur
"""

import pandas as pd
import plotly.graph_objects as go


def build_treemap(
    mkt_data: pd.DataFrame,
    funda_data: pd.DataFrame,
    date_start: str | None = None,
    date_end: str | None = None,
    top_n: int = 100,
    ret_period: str = "YTD",
) -> go.Figure:

    mkt = mkt_data.copy()
    mkt["date"] = pd.to_datetime(mkt["date"])

    if date_start:
        mkt = mkt[mkt["date"] >= pd.to_datetime(date_start)]
    if date_end:
        mkt = mkt[mkt["date"] <= pd.to_datetime(date_end)]

    if mkt.empty:
        return _empty("Aucune donnée pour cette période")

    all_dates = sorted(mkt["date"].unique())
    last_date = all_dates[-1]

    if ret_period == "1M" and len(all_dates) >= 21:
        first_date = all_dates[-21]
    elif ret_period == "3M" and len(all_dates) >= 63:
        first_date = all_dates[-63]
    else:
        first_date = all_dates[0]

    start_prices = (
        mkt[mkt["date"] == first_date]
        .groupby("ticker")["prc"].mean().abs()
    )
    end_prices = (
        mkt[mkt["date"] == last_date]
        .groupby("ticker")["prc"].mean().abs()
    )
    ret_series = ((end_prices - start_prices) / start_prices * 100).dropna()

    latest_mcap = (
        mkt[mkt["date"] == last_date]
        .groupby("ticker")["market_cap"].mean().dropna()
    )

    sectors = (
        funda_data[["ticker", "gicdesc"]]
        .dropna(subset=["gicdesc"])
        .drop_duplicates("ticker")
        .set_index("ticker")["gicdesc"]
    )

    df = pd.DataFrame({
        "market_cap": latest_mcap,
        "ret": ret_series,
        "sector": sectors,
    }).dropna()

    df = df.nlargest(top_n, "market_cap").reset_index()
    df.columns = ["ticker", "market_cap", "ret", "sector"]
    df["cap_B"] = df["market_cap"] / 1e9

    max_abs = max(df["ret"].abs().quantile(0.9), 3)

    def ret_to_color(v):
        t = max(-1, min(1, v / max_abs))
        if t >= 0:
            r = int(20  + (50  - 20)  * t)
            g = int(140 + (220 - 140) * t)
            b = int(20  + (50  - 20)  * t)
        else:
            r = int(180 + (240 - 180) * (-t))
            g = int(20  + (50  - 20)  * (-t))
            b = int(20  + (50  - 20)  * (-t))
        return f"rgb({r},{g},{b})"

    df["color"] = df["ret"].apply(ret_to_color)

    ids, labels_list, parents, values, colors = [], [], [], [], []

    total_cap = df["cap_B"].sum()
    ids.append("root")
    labels_list.append("S&P 500")
    parents.append("")
    values.append(total_cap)
    colors.append("rgba(0,0,0,0)")

    for sector in df["sector"].unique():
        sector_cap = df[df["sector"] == sector]["cap_B"].sum()
        ids.append(sector)
        labels_list.append(f"<b>{sector}</b>")
        parents.append("root")
        values.append(sector_cap)
        colors.append("#1a1a2e")

    for _, row in df.iterrows():
        ids.append(row["ticker"])
        labels_list.append(f"<b>{row['ticker']}</b><br>{row['ret']:+.1f}%")
        parents.append(row["sector"])
        values.append(row["cap_B"])
        colors.append(row["color"])

    period_str = f"{first_date.strftime('%b %Y')} → {last_date.strftime('%b %Y')}"

    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=labels_list,
        parents=parents,
        values=values,
        marker=dict(
            colors=colors,
            line=dict(width=1.5, color="#0f1117"),
        ),
        textinfo="label",
        textfont=dict(family="Inter, sans-serif", size=11, color="white"),
        hovertemplate="<b>%{id}</b><br>Market cap : %{value:.1f}B$<extra></extra>",
        branchvalues="total",
        pathbar=dict(visible=False),
        maxdepth=3,
    ))

    fig.update_layout(
        title=dict(
            text=f"S&P 500 — Performance {period_str}",
            x=0.5, xanchor="center",
            font=dict(size=15, family="Inter, sans-serif", color="#e2e8f0"),
        ),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        margin=dict(l=10, r=10, t=50, b=10),
        height=650,
    )

    return fig


def _empty(msg):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=16, color="#64748b"))
    fig.update_layout(paper_bgcolor="#1c1f2b", plot_bgcolor="#1c1f2b",
                      xaxis=dict(visible=False), yaxis=dict(visible=False), height=400)
    return fig