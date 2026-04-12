"""
Galaxy component — Carte des étoiles financières
Chaque action = une étoile dans une galaxie spirale
Taille = market cap, Luminosité = rendement, Couleur = secteur
"""

import pandas as pd
import numpy as np
import json


SECTOR_COLORS = {
    "Information Technology":  "#818cf8",
    "Communication Services":  "#f472b6",
    "Consumer Discretionary":  "#fbbf24",
    "Financials":              "#22d3ee",
    "Health Care":             "#34d399",
    "Consumer Staples":        "#a78bfa",
    "Energy":                  "#fb923c",
    "Industrials":             "#94a3b8",
    "Materials":               "#86efac",
    "Real Estate":             "#fda4af",
    "Utilities":               "#7dd3fc",
    "Other":                   "#64748b",
}

# Tickers à exclure (doublons ou données aberrantes)
EXCLUDE = {"FB", "GOOGL", "BRK"}


def build_galaxy_html(
    mkt_data: pd.DataFrame,
    funda_data: pd.DataFrame,
    date_start: str | None = None,
    date_end: str | None = None,
    top_n: int = 40,
) -> str:
    """
    Génère un HTML complet avec la galaxie financière interactive.
    """
    mkt = mkt_data.copy()
    mkt["date"] = pd.to_datetime(mkt["date"])

    if date_start:
        mkt = mkt[mkt["date"] >= pd.to_datetime(date_start)]
    if date_end:
        mkt = mkt[mkt["date"] <= pd.to_datetime(date_end)]

    if mkt.empty:
        return "<html><body style='background:#000;color:#fff;padding:20px'>Aucune donnée</body></html>"

    # Exclure les tickers problématiques
    mkt = mkt[~mkt["ticker"].isin(EXCLUDE)]

    # Top N par market cap
    top_tickers = (
        mkt.groupby("ticker")["mcap_rank"].mean()
        .nsmallest(top_n).index.tolist()
    )
    mkt = mkt[mkt["ticker"].isin(top_tickers)]

    # Secteurs
    sectors = (
        funda_data[["ticker", "gicdesc"]]
        .dropna(subset=["gicdesc"])
        .drop_duplicates("ticker")
        .set_index("ticker")["gicdesc"]
    )

    last_date = mkt["date"].max()
    first_date = mkt["date"].min()

    # Calcul par ticker
    stocks = []
    for ticker in top_tickers:
        td = mkt[mkt["ticker"] == ticker]
        if td.empty:
            continue
        p_start = td.sort_values("date")["prc"].abs().iloc[0]
        p_end_row = td[td["date"] == last_date]["prc"].abs()
        if p_end_row.empty or p_start == 0:
            continue
        p_end = p_end_row.mean()
        cum_ret = (p_end - p_start) / p_start * 100
        if abs(cum_ret) > 1000:
            continue  # exclure valeurs aberrantes
        vol = float(td["ret"].std() * np.sqrt(252) * 100)
        mc = td[td["date"] == last_date]["market_cap"].mean()
        if np.isnan(mc) or mc <= 0:
            continue
        stocks.append({
            "t": ticker,
            "s": str(sectors.get(ticker, "Other")),
            "ret": round(float(cum_ret), 1),
            "vol": round(vol, 1),
            "mc": round(float(mc) / 1e9, 1),
        })

    stocks_json = json.dumps(stocks)
    colors_json = json.dumps(SECTOR_COLORS)
    period = f"{first_date.strftime('%b %Y')} → {last_date.strftime('%b %Y')}"

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#000008;font-family:Inter,sans-serif;overflow:hidden}}
#wrap{{padding:10px}}
#ctrl{{display:flex;gap:8px;align-items:center;margin-bottom:6px;font-size:11px;color:#94a3b8;flex-wrap:wrap}}
#cv{{width:100%;display:block;cursor:crosshair}}
#tip{{font-size:11px;color:#94a3b8;padding:5px 10px;background:rgba(28,31,43,0.95);border-radius:6px;margin-top:5px;min-height:20px}}
#leg{{display:flex;gap:8px;flex-wrap:wrap;margin-top:5px;font-size:10px;color:#64748b}}
.li{{display:flex;align-items:center;gap:3px}}
.ld{{width:7px;height:7px;border-radius:50%;flex-shrink:0}}
button{{background:rgba(28,31,43,0.9);color:#94a3b8;border:1px solid #2d3048;border-radius:4px;padding:3px 10px;cursor:pointer;font-size:11px}}
button:hover{{background:#2d3048}}
#pb{{background:#6366f1;color:white;border-color:#6366f1}}
#period{{color:#6366f1;font-size:10px}}
</style>
</head>
<body>
<div id="wrap">
  <div id="ctrl">
    <button id="pb" onclick="togglePlay()">⏸ Pause</button>
    <button onclick="zoom=1;panX=0;panY=0">Reset</button>
    <span>Vitesse</span>
    <input type="range" min="1" max="8" value="3" step="1" style="width:80px" oninput="spd=+this.value">
    <span id="period">{period}</span>
  </div>
  <canvas id="cv"></canvas>
  <div id="tip">Survole une étoile · Scroll = zoom · Drag = naviguer</div>
  <div id="leg"></div>
</div>
<script>
const STOCKS={stocks_json};
const SCOLS={colors_json};
const maxMC=Math.max(...STOCKS.map(s=>s.mc));
let t=0,playing=true,zoom=1,panX=0,panY=0;
let drag=false,lastX,lastY,hov=-1,spd=3;
const canvas=document.getElementById('cv');
const ctx=canvas.getContext('2d');
let W,H,bgStars=[],positions=[];

function genStars(){{
  bgStars=[];
  for(let i=0;i<250;i++)
    bgStars.push({{x:Math.random()*W,y:Math.random()*H,r:Math.random()*0.8+0.1,phase:Math.random()*Math.PI*2}});
}}

function genPositions(){{
  positions=STOCKS.map((s,i)=>{{
    const arm=i%3;
    const n=STOCKS.length;
    const p=Math.floor(i/3)/(Math.ceil(n/3));
    const baseAngle=arm*Math.PI*2/3+p*Math.PI*4+(Math.random()-0.5)*0.3;
    const r=p*0.42+0.06;
    return{{baseAngle,r,speed:(1-r)*0.012+0.004}};
  }});
}}

function resize(){{
  W=canvas.parentElement.clientWidth-20;
  H=Math.max(280,window.innerHeight-110);
  canvas.width=W*devicePixelRatio;canvas.height=H*devicePixelRatio;
  canvas.style.width=W+'px';canvas.style.height=H+'px';
  ctx.scale(devicePixelRatio,devicePixelRatio);
  genStars();genPositions();
}}

function draw(){{
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#000008';ctx.fillRect(0,0,W,H);
  const cx=W/2+panX,cy=H/2+panY;
  const scale=Math.min(W,H)*0.43*zoom;

  // Nébuleuse centrale
  const nb=ctx.createRadialGradient(cx,cy,0,cx,cy,scale*0.8);
  nb.addColorStop(0,'rgba(99,102,241,0.07)');
  nb.addColorStop(0.4,'rgba(139,92,246,0.03)');
  nb.addColorStop(1,'rgba(0,0,0,0)');
  ctx.fillStyle=nb;ctx.fillRect(0,0,W,H);

  // Étoiles de fond
  bgStars.forEach(s=>{{
    const a=Math.sin(s.phase+t*0.015)*0.35+0.45;
    ctx.fillStyle=`rgba(255,255,255,${{(a*0.55).toFixed(2)}})`;
    ctx.beginPath();ctx.arc(s.x,s.y,s.r,0,Math.PI*2);ctx.fill();
  }});

  // Noyau galactique
  const gc=ctx.createRadialGradient(cx,cy,0,cx,cy,scale*0.12);
  gc.addColorStop(0,'rgba(255,220,120,0.35)');
  gc.addColorStop(0.4,'rgba(180,100,255,0.12)');
  gc.addColorStop(1,'rgba(0,0,0,0)');
  ctx.fillStyle=gc;ctx.fillRect(0,0,W,H);

  // Bras spiraux (légers)
  for(let arm=0;arm<3;arm++){{
    ctx.strokeStyle='rgba(120,120,220,0.04)';
    ctx.lineWidth=scale*0.10;ctx.lineCap='round';
    ctx.beginPath();
    for(let p=0;p<=1;p+=0.008){{
      const a=arm*Math.PI*2/3+p*Math.PI*4+t*0.0008;
      const r=p*scale;
      const x=cx+Math.cos(a)*r,y=cy+Math.sin(a)*r*0.52;
      p===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }}
    ctx.stroke();
  }}

  // Actions (étoiles)
  const drawn=[];
  STOCKS.forEach((s,i)=>{{
    const pos=positions[i];
    if(playing) pos.baseAngle+=pos.speed*spd*0.02;
    const a=pos.baseAngle;
    const r=pos.r*scale;
    const x=cx+Math.cos(a)*r;
    const y=cy+Math.sin(a)*r*0.52;
    const size=Math.sqrt(s.mc/maxMC)*18*zoom+3;
    const col=SCOLS[s.s]||'#888';
    const isH=i===hov;

    // Halo pour rendements positifs
    if(s.ret>30){{
      const intensity=Math.min(s.ret/300,0.8);
      const glow=ctx.createRadialGradient(x,y,0,x,y,size*3.5);
      glow.addColorStop(0,col+Math.round(intensity*80).toString(16).padStart(2,'0'));
      glow.addColorStop(1,'rgba(0,0,0,0)');
      ctx.fillStyle=glow;ctx.fillRect(x-size*3.5,y-size*3.5,size*7,size*7);
    }}

    // Corps de l'étoile
    ctx.fillStyle=isH?col+'ff':col+(s.ret>0?'dd':'88');
    ctx.beginPath();ctx.arc(x,y,isH?size+3:size,0,Math.PI*2);ctx.fill();

    // Halo hover
    if(isH){{
      ctx.strokeStyle=col;ctx.lineWidth=1.5;
      ctx.beginPath();ctx.arc(x,y,size+6,0,Math.PI*2);ctx.stroke();
    }}

    // Scintillement
    const tw=Math.sin(t*0.04+i*1.3)*0.25+0.65;
    ctx.fillStyle=`rgba(255,255,255,${{(tw*0.55).toFixed(2)}})`;
    ctx.beginPath();ctx.arc(x,y,size*0.32,0,Math.PI*2);ctx.fill();

    // Label
    if(size>9||isH){{
      ctx.fillStyle=isH?'white':'rgba(255,255,255,0.75)';
      ctx.font=`bold ${{Math.max(7,Math.min(11,size*0.62))}}px Inter,sans-serif`;
      ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.fillText(s.t,x,y);
    }}
    drawn.push({{i,x,y,size,s}});
  }});
  canvas._drawn=drawn;
}}

canvas.addEventListener('mousemove',e=>{{
  const rect=canvas.getBoundingClientRect();
  const mx=e.clientX-rect.left,my=e.clientY-rect.top;
  if(drag){{panX+=mx-lastX;panY+=my-lastY;lastX=mx;lastY=my;return;}}
  hov=-1;
  (canvas._drawn||[]).forEach(d=>{{
    const dx=mx-d.x,dy=my-d.y;
    if(Math.sqrt(dx*dx+dy*dy)<d.size+5){{
      hov=d.i;
      const s=d.s;
      const retColor=s.ret>=0?'#34d399':'#fb7185';
      const retStr=(s.ret>0?'+':'')+s.ret.toFixed(1)+'%';
      document.getElementById('tip').innerHTML=
        `<b style="color:${{SCOLS[s.s]||'#888'}}">${{s.t}}</b> · ${{s.s}} · Rendement : <b style="color:${{retColor}}">${{retStr}}</b> · Vol : ${{s.vol}}% · Market cap : ${{s.mc}}B$`;
    }}
  }});
  if(hov===-1) document.getElementById('tip').textContent='Survole une étoile · Scroll = zoom · Drag = naviguer';
}});

canvas.addEventListener('mousedown',e=>{{
  drag=true;canvas.style.cursor='grabbing';
  const r=canvas.getBoundingClientRect();lastX=e.clientX-r.left;lastY=e.clientY-r.top;
}});
canvas.addEventListener('mouseup',()=>{{drag=false;canvas.style.cursor='crosshair';}});
canvas.addEventListener('mouseleave',()=>{{drag=false;hov=-1;}});
canvas.addEventListener('wheel',e=>{{
  e.preventDefault();
  zoom=Math.max(0.4,Math.min(4,zoom*(e.deltaY<0?1.12:0.9)));
}},{{passive:false}});

function togglePlay(){{playing=!playing;document.getElementById('pb').textContent=playing?'⏸ Pause':'▶ Play';}}

// Légende
const leg=document.getElementById('leg');
const seen=new Set(STOCKS.map(s=>s.s));
Object.entries(SCOLS).forEach(([s,c])=>{{
  if(!seen.has(s))return;
  leg.innerHTML+=`<span class="li"><span class="ld" style="background:${{c}}"></span>${{s==='Information Technology'?'Tech':s==='Communication Services'?'Comm.':s==='Consumer Discretionary'?'Conso':s==='Consumer Staples'?'Staples':s}}</span>`;
}});

function loop(){{t++;draw();requestAnimationFrame(loop);}}
resize();loop();
window.addEventListener('resize',()=>{{ctx.setTransform(1,0,0,1,0,0);resize();}});
</script>
</body>
</html>"""
    return html
