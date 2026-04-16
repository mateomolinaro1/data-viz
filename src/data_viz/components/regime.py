"""
Regime Detection component - Prix / Volatilite / Drawdown
Detection bull/bear via moyenne mobile 20 semaines
Action choisie vs pairs du meme secteur
"""

import pandas as pd
import numpy as np
import json

EXCLUDE = {"FB", "GOOGL", "BRK"}

SECTOR_COLORS = {
    "Information Technology":  "#818cf8",
    "Communication Services":  "#f472b6",
    "Consumer Discretionary":  "#fbbf24",
    "Financials":              "#22d3ee",
    "Health Care":             "#34d399",
    "Consumer Staples":        "#a78bfa",
    "Energy":                  "#fb923c",
    "Industrials":             "#94a3b8",
    "Utilities":               "#7dd3fc",
    "Other":                   "#64748b",
}


def build_regime_html(
    mkt_data: pd.DataFrame,
    funda_data: pd.DataFrame,
    date_start: str | None = None,
    date_end: str | None = None,
    top_n: int = 60,
) -> str:
    mkt = mkt_data.copy()
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt = mkt[~mkt["ticker"].isin(EXCLUDE)]

    if date_start:
        mkt = mkt[mkt["date"] >= pd.to_datetime(date_start)]
    if date_end:
        mkt = mkt[mkt["date"] <= pd.to_datetime(date_end)]

    if mkt.empty:
        return "<html><body style='background:#0d0d12;color:#fff'>Aucune donnee</body></html>"

    sectors = (
        funda_data[["ticker", "gicdesc"]]
        .dropna(subset=["gicdesc"])
        .drop_duplicates("ticker")
        .set_index("ticker")["gicdesc"]
    )

    top_tickers = mkt.groupby("ticker")["mcap_rank"].mean().nsmallest(top_n).index.tolist()
    mkt_top = mkt[mkt["ticker"].isin(top_tickers)].copy()
    mkt_top["sector"] = mkt_top["ticker"].map(sectors)
    mkt_top = mkt_top.dropna(subset=["sector"])

    result = {}
    sector_map = {}

    for ticker in mkt_top["ticker"].unique():
        td = mkt_top[mkt_top["ticker"] == ticker].sort_values("date")
        if len(td) < 30:
            continue
        prices = td.set_index("date")["prc"].abs().resample("W-FRI").last().dropna()
        ret_w = td.set_index("date")["ret"].resample("W-FRI").apply(
            lambda x: (1 + x).prod() - 1
        ).dropna()
        base = prices.iloc[0]
        if base == 0:
            continue
        norm = (prices / base * 100).round(1)
        vol = (ret_w.rolling(8).std() * np.sqrt(52) * 100).round(1)
        cummax = norm.cummax()
        dd = ((norm - cummax) / cummax * 100).round(1)
        ma20 = norm.rolling(20).mean().round(1)
        sec = td["sector"].iloc[0]
        sector_map[ticker] = sec
        result[ticker] = {
            "s": sec,
            "p": [round(float(v), 1) for v in norm.values],
            "v": [round(float(v), 1) if not np.isnan(v) else None for v in vol.values],
            "d": [round(float(v), 1) for v in dd.values],
            "m": [round(float(v), 1) if not np.isnan(v) else None for v in ma20.values],
        }

    min_date = mkt_top["date"].min()
    max_date = mkt_top["date"].max()
    dates_weekly = [
        str(d.date()) for d in pd.date_range(min_date, max_date, freq="W-FRI")
    ]

    sector_tickers = {}
    for t, s in sector_map.items():
        sector_tickers.setdefault(s, []).append(t)

    data_json = json.dumps({
        "dates": dates_weekly,
        "tickers": result,
        "sectors": sector_tickers,
    })
    colors_json = json.dumps(SECTOR_COLORS)

    all_tickers = sorted(result.keys())
    ticker_opts = "\n".join(
        '<option value="' + t + '"' + (' selected' if t == "AAPL" else '') + '>' + t + '</option>'
        for t in all_tickers
    )
    n = len(dates_weekly)

    html = (
        "<!DOCTYPE html>\n<html>\n<head><meta charset=\"utf-8\">\n"
        "<script src=\"https://cdn.plot.ly/plotly-2.27.0.min.js\"></script>\n"
        "<style>\n"
        "*{margin:0;padding:0;box-sizing:border-box}\n"
        "body{background:#0d0d12;font-family:Inter,sans-serif;color:#e2e8f0;padding:10px}\n"
        "#ctrl{display:flex;gap:10px;align-items:center;margin-bottom:6px;flex-wrap:wrap;font-size:12px;color:#94a3b8}\n"
        "select{background:#1c1f2b;color:#e2e8f0;border:1px solid #2d3048;border-radius:5px;padding:4px 8px;font-size:12px;cursor:pointer}\n"
        "#srow{display:flex;gap:8px;align-items:center;margin-bottom:6px;font-size:11px;color:#94a3b8}\n"
        "#srow input{flex:1}\n"
        "#dlbl{color:#6366f1;font-weight:700;min-width:90px}\n"
        "button{background:#1c1f2b;color:#94a3b8;border:1px solid #2d3048;border-radius:4px;padding:3px 10px;cursor:pointer;font-size:11px}\n"
        "#pb{background:#6366f1;color:white;border-color:#6366f1}\n"
        "#leg{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:6px;font-size:10px;color:#94a3b8;align-items:center}\n"
        ".li{display:flex;align-items:center;gap:4px}\n"
        ".lc{width:20px;height:3px;border-radius:2px}\n"
        ".lb{width:12px;height:12px;border-radius:2px;opacity:0.5}\n"
        "#info{font-size:10px;color:#64748b;margin-bottom:4px;padding:4px 8px;background:#1c1f2b;border-radius:4px}\n"
        "</style>\n</head>\n<body>\n"
        "<div id=\"ctrl\">\n"
        "  <button id=\"pb\" onclick=\"togglePlay()\">Pause</button>\n"
        "  <span>Action :</span>\n"
        "  <select id=\"tsel\" onchange=\"changeTicker(this.value)\">\n"
        + ticker_opts +
        "\n  </select>\n"
        "  <span id=\"slbl\" style=\"color:#818cf8;font-weight:600\"></span>\n"
        "  <span style=\"color:#555;font-size:10px\">vs pairs secteur en gris</span>\n"
        "  <span style=\"margin-left:auto;font-size:10px;color:#6366f1\" id=\"regime-badge\"></span>\n"
        "</div>\n"
        "<div id=\"leg\">\n"
        "  <span style=\"color:#64748b;font-size:10px\">Legende :</span>\n"
        "  <span class=\"li\"><span class=\"lc\" style=\"background:#818cf8\"></span>Action selectionnee</span>\n"
        "  <span class=\"li\"><span class=\"lc\" style=\"background:rgba(148,163,184,0.4)\"></span>Pairs du secteur</span>\n"
        "  <span class=\"li\"><span class=\"lc\" style=\"background:#fbbf24;height:2px;border-top:2px dashed #fbbf24\"></span>Moy. mobile 20 sem.</span>\n"
        "  <span class=\"li\"><span class=\"lb\" style=\"background:#22c55e\"></span>Bull market (prix > MM20)</span>\n"
        "  <span class=\"li\"><span class=\"lb\" style=\"background:#ef4444\"></span>Bear market (prix < MM20)</span>\n"
        "  <span class=\"li\"><span class=\"lb\" style=\"background:#f97316\"></span>Haute volatilite (vol > 40%)</span>\n"
        "</div>\n"
        "<div id=\"info\">Zone rouge = Covid mars 2020 | Zone orange = Hausse taux Fed 2022 | Base 100 = premier prix de la periode</div>\n"
        "<div id=\"srow\">\n"
        "  <span>Debut</span>\n"
        "  <input type=\"range\" id=\"tslider\" min=\"0\" max=\""
        + str(n - 1)
        + "\" value=\"0\" step=\"1\" oninput=\"fi=+this.value;playing=false;document.getElementById('pb').textContent='Play';draw()\">\n"
        "  <span>Fin</span>\n"
        "  <span id=\"dlbl\"></span>\n"
        "</div>\n"
        "<div id=\"p1\" style=\"height:145px\"></div>\n"
        "<div id=\"p2\" style=\"height:105px\"></div>\n"
        "<div id=\"p3\" style=\"height:105px\"></div>\n"
        "<script>\n"
        "const RAW=" + data_json + ";\n"
        "const SCOLS=" + colors_json + ";\n"
        "const NN=RAW.dates.length;\n"
        "let fi=0,playing=true,ticker='AAPL';\n"
        "const CFG={displayModeBar:false,responsive:true};\n"
        "const BG={paper_bgcolor:'#0d0d12',plot_bgcolor:'#0d0d12',font:{color:'#94a3b8',size:10},margin:{l:52,r:8,t:4,b:22},showlegend:false,xaxis:{showgrid:false,tickfont:{size:9}},yaxis:{gridcolor:'rgba(255,255,255,0.05)',tickfont:{size:9}}};\n"
        "\n"
        "// Detect bull/bear regimes from price vs MA20\n"
        "function getRegimeShapes(key){\n"
        "  const t=RAW.tickers[ticker];if(!t||!t.m)return[];\n"
        "  const shapes=[];\n"
        "  const dates=RAW.dates;\n"
        "  let start=null,isBull=null;\n"
        "  for(let i=20;i<=fi;i++){\n"
        "    const p=t.p[i],m=t.m[i];\n"
        "    if(p==null||m==null)continue;\n"
        "    const bull=p>=m;\n"
        "    if(isBull===null){start=i;isBull=bull;continue;}\n"
        "    if(bull!==isBull){\n"
        "      shapes.push({type:'rect',x0:dates[start],x1:dates[i],y0:0,y1:1,yref:'paper',\n"
        "        fillcolor:isBull?'rgba(34,197,94,0.12)':'rgba(239,68,68,0.12)',line:{width:0}});\n"
        "      start=i;isBull=bull;\n"
        "    }\n"
        "  }\n"
        "  if(start!==null&&fi>start){\n"
        "    shapes.push({type:'rect',x0:dates[start],x1:dates[fi],y0:0,y1:1,yref:'paper',\n"
        "      fillcolor:isBull?'rgba(34,197,94,0.12)':'rgba(239,68,68,0.12)',line:{width:0}});\n"
        "  }\n"
        "  return shapes;\n"
        "}\n"
        "\n"
        "// High volatility zones on vol chart\n"
        "function getVolShapes(){\n"
        "  const t=RAW.tickers[ticker];if(!t||!t.v)return[];\n"
        "  const shapes=[];const dates=RAW.dates;\n"
        "  let start=null;\n"
        "  const thresh=40;\n"
        "  for(let i=0;i<=fi;i++){\n"
        "    const v=t.v[i];if(v==null)continue;\n"
        "    if(v>=thresh&&start===null)start=i;\n"
        "    else if(v<thresh&&start!==null){\n"
        "      shapes.push({type:'rect',x0:dates[start],x1:dates[i],y0:0,y1:1,yref:'paper',\n"
        "        fillcolor:'rgba(249,115,22,0.2)',line:{width:0}});\n"
        "      start=null;\n"
        "    }\n"
        "  }\n"
        "  if(start!==null)shapes.push({type:'rect',x0:dates[start],x1:dates[fi],y0:0,y1:1,yref:'paper',fillcolor:'rgba(249,115,22,0.2)',line:{width:0}});\n"
        "  return shapes;\n"
        "}\n"
        "\n"
        "function peerTraces(key){\n"
        "  const t=RAW.tickers[ticker];if(!t)return[];\n"
        "  const peers=(RAW.sectors[t.s]||[]).filter(x=>x!==ticker).slice(0,6);\n"
        "  const dates=RAW.dates.slice(0,fi+1);\n"
        "  const tr=[];\n"
        "  peers.forEach(p=>{const pt=RAW.tickers[p];if(!pt)return;\n"
        "    tr.push({x:dates,y:pt[key].slice(0,fi+1),mode:'lines',line:{color:'rgba(148,163,184,0.18)',width:1},hoverinfo:'skip'});\n"
        "  });\n"
        "  return tr;\n"
        "}\n"
        "\n"
        "function draw(){\n"
        "  const t=RAW.tickers[ticker];if(!t)return;\n"
        "  const col=SCOLS[t.s]||'#6366f1';\n"
        "  const dates=RAW.dates.slice(0,fi+1);\n"
        "  const rshapes=getRegimeShapes();\n"
        "  const vshapes=getVolShapes();\n"
        "\n"
        "  // Panel 1: Price + MA20 + regime zones\n"
        "  const tr1=[...peerTraces('p'),\n"
        "    {x:dates,y:t.p.slice(0,fi+1),mode:'lines',line:{color:col,width:2.5},name:ticker},\n"
        "    {x:dates,y:t.m.slice(0,fi+1),mode:'lines',line:{color:'#fbbf24',width:1.2,dash:'dash'},name:'MM20'},\n"
        "  ];\n"
        "  if(t.p[fi]!=null)tr1.push({x:[RAW.dates[fi]],y:[t.p[fi]],mode:'markers',marker:{color:col,size:7},hoverinfo:'skip'});\n"
        "  Plotly.react('p1',tr1,{...BG,yaxis:{...BG.yaxis,title:'Prix (base 100)'},shapes:rshapes},CFG);\n"
        "\n"
        "  // Panel 2: Volatility + high-vol zones\n"
        "  const tr2=[...peerTraces('v'),\n"
        "    {x:dates,y:t.v.slice(0,fi+1),mode:'lines',line:{color:col,width:2.5}},\n"
        "    {x:[RAW.dates[0],RAW.dates[fi]],y:[40,40],mode:'lines',line:{color:'#f97316',width:1,dash:'dot'},hoverinfo:'skip'},\n"
        "  ];\n"
        "  Plotly.react('p2',tr2,{...BG,yaxis:{...BG.yaxis,title:'Volatilite (%)',rangemode:'tozero'},shapes:vshapes},CFG);\n"
        "\n"
        "  // Panel 3: Drawdown\n"
        "  const tr3=[...peerTraces('d'),\n"
        "    {x:dates,y:t.d.slice(0,fi+1),mode:'lines',line:{color:col,width:2.5}},\n"
        "    {x:[RAW.dates[0],RAW.dates[fi]],y:[0,0],mode:'lines',line:{color:'rgba(255,255,255,0.15)',width:1,dash:'dot'},hoverinfo:'skip'},\n"
        "  ];\n"
        "  Plotly.react('p3',tr3,{...BG,yaxis:{...BG.yaxis,title:'Drawdown (%)'} ,shapes:rshapes},CFG);\n"
        "\n"
        "  // Regime badge\n"
        "  const p=t.p[fi],m=t.m[fi],v=t.v[fi];\n"
        "  let badge='';\n"
        "  if(p!=null&&m!=null){\n"
        "    if(p>=m)badge='<span style=\"color:#22c55e;font-weight:700\">BULL</span>';\n"
        "    else badge='<span style=\"color:#ef4444;font-weight:700\">BEAR</span>';\n"
        "  }\n"
        "  if(v!=null&&v>=40)badge+=' <span style=\"color:#f97316\">HAUTE VOL</span>';\n"
        "  document.getElementById('regime-badge').innerHTML='Regime : '+badge;\n"
        "  document.getElementById('dlbl').textContent=RAW.dates[fi]||'';\n"
        "  document.getElementById('tslider').value=fi;\n"
        "}\n"
        "\n"
        "function changeTicker(t){\n"
        "  ticker=t;\n"
        "  const sec=RAW.tickers[t]?RAW.tickers[t].s:'';\n"
        "  const col=SCOLS[sec]||'#6366f1';\n"
        "  document.getElementById('slbl').textContent=sec;\n"
        "  document.getElementById('slbl').style.color=col;\n"
        "  draw();\n"
        "}\n"
        "\n"
        "function togglePlay(){playing=!playing;document.getElementById('pb').textContent=playing?'Pause':'Play';}\n"
        "function loop(){if(playing){fi=fi<NN-1?fi+1:0;}draw();setTimeout(loop,150);}\n"
        "changeTicker('AAPL');\n"
        "loop();\n"
        "</script>\n</body>\n</html>"
    )

    return html