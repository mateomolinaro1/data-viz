"""
Network graph component — corrélations entre stocks
Rendu via HTML/Canvas avec simulation de forces, intégré dans Dash via Iframe
"""

import pandas as pd
import numpy as np
import json


SECTOR_COLORS = {
    "Information Technology": "#6366f1",
    "Financials":             "#22d3ee",
    "Health Care":            "#34d399",
    "Consumer Discretionary": "#f59e0b",
    "Consumer Staples":       "#a78bfa",
    "Energy":                 "#f97316",
    "Industrials":            "#94a3b8",
    "Communication Services": "#fb7185",
    "Materials":              "#84cc16",
    "Real Estate":            "#e879f9",
    "Utilities":              "#38bdf8",
}


def build_network_html(
    mkt_data: pd.DataFrame,
    funda_data: pd.DataFrame,
    date_start: str | None = None,
    date_end: str | None = None,
    top_n: int = 30,
    thresh: float = 0.5,
) -> str:
    """
    Calcule la matrice de corrélation et retourne un HTML complet
    avec le network graph interactif (Canvas + simulation de forces).
    """
    mkt = mkt_data.copy()
    mkt["date"] = pd.to_datetime(mkt["date"])

    if date_start:
        mkt = mkt[mkt["date"] >= pd.to_datetime(date_start)]
    if date_end:
        mkt = mkt[mkt["date"] <= pd.to_datetime(date_end)]

    # Top N par market cap
    top_tickers = (
        mkt.groupby("ticker")["mcap_rank"].mean()
        .nsmallest(top_n).index.tolist()
    )
    mkt = mkt[mkt["ticker"].isin(top_tickers)]

    # Matrice de corrélation
    pivot = mkt.pivot_table(index="date", columns="ticker", values="ret")
    pivot = pivot.dropna(axis=1, thresh=int(len(pivot) * 0.5))
    corr = pivot.corr()

    tickers = corr.columns.tolist()

    # Secteurs
    sectors = (
        funda_data[["ticker", "gicdesc"]]
        .dropna(subset=["gicdesc"])
        .drop_duplicates("ticker")
        .set_index("ticker")["gicdesc"]
    )

    nodes = []
    for t in tickers:
        sector = sectors.get(t, "Other")
        color = SECTOR_COLORS.get(sector, "#888888")
        nodes.append({"t": t, "s": sector, "c": color})

    # Matrice de corrélation → liste de liens
    n = len(tickers)
    corr_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            v = corr.iloc[i, j]
            row.append(round(float(v), 3) if not np.isnan(v) else 0.0)
        corr_matrix.append(row)

    nodes_json = json.dumps(nodes)
    corr_json = json.dumps(corr_matrix)
    thresh_val = thresh
    sector_colors_json = json.dumps(SECTOR_COLORS)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0d0d12; font-family:Inter,sans-serif; overflow:hidden; }}
  #wrap {{ padding:12px; }}
  #ctrl {{ display:flex; gap:10px; align-items:center; margin-bottom:8px; font-size:11px; color:#94a3b8; flex-wrap:wrap; }}
  #ctrl input {{ flex:1; min-width:80px; }}
  #tval {{ color:#6366f1; font-weight:600; min-width:30px; }}
  #cv {{ width:100%; display:block; cursor:grab; }}
  #info {{ min-height:24px; font-size:11px; color:#94a3b8; margin-top:6px;
           padding:5px 10px; background:#1c1f2b; border-radius:6px; }}
  #legend {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:6px; font-size:10px; color:#64748b; }}
  .li {{ display:flex; align-items:center; gap:4px; }}
  .ld {{ width:8px; height:8px; border-radius:50%; flex-shrink:0; }}
  button {{ background:#1c1f2b; color:#94a3b8; border:1px solid #2d3048;
            border-radius:4px; padding:3px 10px; cursor:pointer; font-size:11px; }}
  button:hover {{ background:#2d3048; }}
</style>
</head>
<body>
<div id="wrap">
  <div id="ctrl">
    <span>Seuil ρ</span>
    <input type="range" min="20" max="85" value="{int(thresh_val*100)}" step="5" id="tslider" oninput="setT(this.value)">
    <span id="tval">{thresh_val:.2f}</span>
    <button onclick="resetSim()">Reset</button>
    <span id="nlinks" style="color:#6366f1"></span>
  </div>
  <canvas id="cv"></canvas>
  <div id="info">Clique sur un nœud pour voir ses connexions</div>
  <div id="legend"></div>
</div>
<script>
const ST = {nodes_json};
const CORR = {corr_json};
const SCOLS = {sector_colors_json};
const N = ST.length;
let thresh = {thresh_val};
let nodes, dragging=-1, dragOX, dragOY, hov=-1, selected=-1;
const canvas = document.getElementById('cv');
const ctx = canvas.getContext('2d');
let W, H;

function initNodes() {{
  W = canvas.parentElement.clientWidth - 24;
  H = Math.max(300, window.innerHeight - 120);
  canvas.width = W * devicePixelRatio;
  canvas.height = H * devicePixelRatio;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  ctx.scale(devicePixelRatio, devicePixelRatio);
  nodes = ST.map((s, i) => {{
    const angle = i / N * Math.PI * 2;
    const r = Math.min(W, H) * 0.33;
    return {{...s, i,
      x: W/2 + Math.cos(angle)*r + (Math.random()-0.5)*50,
      y: H/2 + Math.sin(angle)*r + (Math.random()-0.5)*50,
      vx:0, vy:0
    }};
  }});
  updateLinkCount();
}}

function setT(v) {{
  thresh = v/100;
  document.getElementById('tval').textContent = (v/100).toFixed(2);
  updateLinkCount();
}}

function resetSim() {{ initNodes(); }}

function updateLinkCount() {{
  let c=0;
  for(let i=0;i<N;i++) for(let j=i+1;j<N;j++) if(CORR[i][j]>=thresh) c++;
  document.getElementById('nlinks').textContent = c + ' liens';
}}

function simulate() {{
  const rep=2500, att=0.003, damp=0.82, center=0.002;
  nodes.forEach(a => {{
    a.vx += (W/2 - a.x) * center;
    a.vy += (H/2 - a.y) * center;
    nodes.forEach(b => {{
      if(a===b) return;
      const dx=a.x-b.x, dy=a.y-b.y, d=Math.sqrt(dx*dx+dy*dy)||1;
      const f = rep/(d*d);
      a.vx += f*dx/d; a.vy += f*dy/d;
    }});
    nodes.forEach(b => {{
      if(a.i>=b.i) return;
      const c = CORR[a.i][b.i];
      if(c < thresh) return;
      const dx=b.x-a.x, dy=b.y-a.y, d=Math.sqrt(dx*dx+dy*dy)||1;
      const target = 60 + 90*(1-c);
      const f = att*(d-target);
      const fx=f*dx/d, fy=f*dy/d;
      a.vx+=fx; a.vy+=fy; b.vx-=fx; b.vy-=fy;
    }});
    if(dragging !== a.i) {{
      a.vx *= damp; a.vy *= damp;
      a.x += a.vx; a.y += a.vy;
      a.x = Math.max(18, Math.min(W-18, a.x));
      a.y = Math.max(18, Math.min(H-18, a.y));
    }}
  }});
}}

function draw() {{
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#0d0d12'; ctx.fillRect(0,0,W,H);
  const active = selected>=0 ? selected : hov;
  nodes.forEach(a => {{ nodes.forEach(b => {{
    if(a.i>=b.i) return;
    const c = CORR[a.i][b.i];
    if(c < thresh) return;
    const hl = active>=0 && (a.i===active||b.i===active);
    const alpha = hl ? 0.9 : (active>=0 ? 0.06 : (c-thresh)/(1-thresh)*0.55+0.08);
    ctx.strokeStyle = `rgba(99,102,241,${{alpha.toFixed(2)}})`;
    ctx.lineWidth = hl ? c*4+1 : c*1.5+0.3;
    ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
  }}); }});
  nodes.forEach(n => {{
    const isHov=n.i===hov, isSel=n.i===selected;
    const r = isSel?18:isHov?15:12;
    const dimmed = active>=0 && n.i!==active;
    ctx.fillStyle = n.c + (dimmed?'44':'');
    ctx.beginPath(); ctx.arc(n.x,n.y,r,0,Math.PI*2); ctx.fill();
    if(isSel) {{
      ctx.strokeStyle='white'; ctx.lineWidth=2.5;
      ctx.beginPath(); ctx.arc(n.x,n.y,r,0,Math.PI*2); ctx.stroke();
    }}
    ctx.fillStyle = dimmed ? 'rgba(255,255,255,0.3)' : 'white';
    ctx.font = `bold ${{Math.max(7,Math.min(10,r*0.75))}}px Inter,sans-serif`;
    ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText(n.t, n.x, n.y);
  }});
}}

function getHov(mx,my) {{
  for(let i=0;i<nodes.length;i++) {{
    const n=nodes[i], dx=mx-n.x, dy=my-n.y;
    if(Math.sqrt(dx*dx+dy*dy)<18) return i;
  }}
  return -1;
}}

canvas.addEventListener('mousedown', e => {{
  const r=canvas.getBoundingClientRect();
  const mx=e.clientX-r.left, my=e.clientY-r.top;
  const h=getHov(mx,my);
  if(h>=0) {{ dragging=h; dragOX=mx-nodes[h].x; dragOY=my-nodes[h].y; }}
}});

canvas.addEventListener('mousemove', e => {{
  const r=canvas.getBoundingClientRect();
  const mx=e.clientX-r.left, my=e.clientY-r.top;
  if(dragging>=0) {{ nodes[dragging].x=mx-dragOX; nodes[dragging].y=my-dragOY; }}
  hov = dragging>=0 ? -1 : getHov(mx,my);
  canvas.style.cursor = hov>=0||dragging>=0 ? 'grabbing' : 'default';
}});

canvas.addEventListener('mouseup', e => {{
  const r=canvas.getBoundingClientRect();
  const mx=e.clientX-r.left, my=e.clientY-r.top;
  const h=getHov(mx,my);
  if(dragging===h && h>=0) {{
    selected = selected===h ? -1 : h;
    if(selected>=0) {{
      const n=ST[selected];
      const links=nodes.filter((_,j)=>j!==selected&&CORR[selected][j]>=thresh);
      document.getElementById('info').innerHTML =
        `<b style="color:#e2e8f0">${{n.t}}</b> · ${{n.s}} · <span style="color:#6366f1">${{links.length}} connexions</span> : ${{links.map(l=>`<span style="color:${{l.c}}">${{l.t}}</span>`).join(', ')}}`;
    }} else {{
      document.getElementById('info').textContent = 'Clique sur un nœud pour voir ses connexions';
    }}
  }}
  dragging=-1;
}});

canvas.addEventListener('mouseleave', ()=>{{ hov=-1; dragging=-1; }});

function buildLegend() {{
  const leg=document.getElementById('legend');
  const seen=new Set(ST.map(s=>s.s));
  Object.entries(SCOLS).forEach(([s,c])=>{{
    if(!seen.has(s)) return;
    leg.innerHTML += `<span class="li"><span class="ld" style="background:${{c}}"></span>${{s}}</span>`;
  }});
}}

function loop() {{ simulate(); draw(); requestAnimationFrame(loop); }}
initNodes();
buildLegend();
loop();
</script>
</body>
</html>"""
    return html
