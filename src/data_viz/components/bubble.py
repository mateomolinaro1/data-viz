"""
Bubble chart animé — style Hans Rosling
X = rendement cumulé, Y = volatilité, Taille = market cap, Couleur = secteur
Animation mois par mois sur données CRSP réelles
"""

import pandas as pd
import numpy as np
import json


SECTOR_COLORS = {
    "Information Technology":  "#6366f1",
    "Communication Services":  "#f472b6",
    "Consumer Discretionary":  "#f59e0b",
    "Financials":              "#22d3ee",
    "Health Care":             "#34d399",
    "Consumer Staples":        "#a78bfa",
    "Energy":                  "#fb923c",
    "Industrials":             "#94a3b8",
    "Materials":               "#86efac",
    "Other":                   "#64748b",
}

EXCLUDE = {"FB", "GOOGL", "BRK"}


def build_bubble_html(
    mkt_data: pd.DataFrame,
    funda_data: pd.DataFrame,
    date_start: str | None = None,
    date_end: str | None = None,
    top_n: int = 20,
) -> str:

    mkt = mkt_data.copy()
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt = mkt[~mkt["ticker"].isin(EXCLUDE)]

    if date_start:
        mkt = mkt[mkt["date"] >= pd.to_datetime(date_start)]
    if date_end:
        mkt = mkt[mkt["date"] <= pd.to_datetime(date_end)]

    if mkt.empty:
        return "<html><body style='background:#0d0d12;color:#fff;padding:20px'>Aucune donnée</body></html>"

    mkt["month"] = mkt["date"].dt.to_period("M")
    top_tickers = mkt.groupby("ticker")["mcap_rank"].mean().nsmallest(top_n).index.tolist()
    mkt_top = mkt[mkt["ticker"].isin(top_tickers)]

    sectors = (
        funda_data[["ticker", "gicdesc"]]
        .dropna(subset=["gicdesc"])
        .drop_duplicates("ticker")
        .set_index("ticker")["gicdesc"]
    )

    months = sorted(mkt_top["month"].unique())
    result = []
    for m in months:
        mdata = mkt_top[mkt_top["month"] == m]
        for ticker in top_tickers:
            tdata = mkt_top[mkt_top["ticker"] == ticker]
            tdata_m = mdata[mdata["ticker"] == ticker]
            if tdata_m.empty:
                continue
            start_price = tdata.sort_values("date")["prc"].abs().iloc[0]
            end_price = tdata_m["prc"].abs().mean()
            if start_price == 0:
                continue
            cum_ret = (end_price - start_price) / start_price * 100
            if abs(cum_ret) > 1000:
                continue
            recent = mkt_top[(mkt_top["ticker"] == ticker) & (mkt_top["month"] <= m)].tail(63)
            vol = float(recent["ret"].std() * np.sqrt(252) * 100)
            mc = tdata_m["market_cap"].mean()
            if np.isnan(mc) or mc <= 0:
                continue
            result.append({
                "month": str(m),
                "ticker": ticker,
                "sector": str(sectors.get(ticker, "Other")),
                "cum_ret": round(float(cum_ret), 1),
                "vol": round(vol, 1),
                "mc_B": round(float(mc) / 1e9, 1),
            })

    df = pd.DataFrame(result)
    months_list = [str(m) for m in months]

    tickers_info = {}
    for ticker in top_tickers:
        td = df[df["ticker"] == ticker]
        monthly = {}
        for _, row in td.iterrows():
            monthly[row["month"]] = [row["cum_ret"], row["vol"], row["mc_B"]]
        tickers_info[ticker] = {
            "s": str(sectors.get(ticker, "Other")),
            "d": monthly,
        }

    data_json = json.dumps({"months": months_list, "tickers": tickers_info})
    colors_json = json.dumps(SECTOR_COLORS)

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0d0d12;font-family:Inter,sans-serif;overflow:hidden}}
#wrap{{padding:10px}}
#ctrl{{display:flex;gap:8px;align-items:center;margin-bottom:4px;font-size:11px;color:#94a3b8;flex-wrap:wrap}}
#cv{{width:100%;display:block}}
#date-label{{color:#6366f1;font-size:20px;font-weight:700;text-align:right;margin:2px 4px}}
#tip{{font-size:11px;color:#94a3b8;padding:5px 10px;background:#1c1f2b;border-radius:6px;margin-top:5px;min-height:20px}}
#leg{{display:flex;gap:10px;flex-wrap:wrap;margin-top:5px;font-size:10px;color:#94a3b8}}
.li{{display:flex;align-items:center;gap:3px}}
.ld{{width:8px;height:8px;border-radius:50%;flex-shrink:0}}
button{{background:#1c1f2b;color:#94a3b8;border:1px solid #2d3048;border-radius:4px;padding:3px 10px;cursor:pointer;font-size:11px}}
button:hover{{background:#2d3048}}
#pb{{background:#6366f1;color:white;border-color:#6366f1}}
</style>
</head>
<body>
<div id="wrap">
  <div id="ctrl">
    <button id="pb" onclick="togglePlay()">&#9208; Pause</button>
    <button onclick="fi=0">&#8617; 2020</button>
    <span>Vitesse</span>
    <input type="range" min="1" max="10" value="2" step="1" style="width:70px" oninput="spd=+this.value">
    <input type="range" id="tslider" min="0" max="59" value="0" step="1" style="flex:2;min-width:80px" oninput="fi=+this.value;playing=false;document.getElementById('pb').textContent='Play'">
  </div>
  <p id="date-label">Jan 2020</p>
  <canvas id="cv" height="320"></canvas>
  <div id="tip">Survole une bulle pour voir les details</div>
  <div id="leg"></div>
</div>
<script>
const RAW={data_json};
const SCOLS={colors_json};
const MONTHS=RAW.months;
const TICKERS=Object.keys(RAW.tickers);
let fi=0,playing=true,spd=4,hov=-1;
const canvas=document.getElementById('cv');
const ctx=canvas.getContext('2d');

function getPoint(t,m){{
  const d=RAW.tickers[t].d[m];
  return d?{{ret:d[0],vol:d[1],mc:d[2]}}:null;
}}

function draw(){{
  const W=canvas.parentElement.clientWidth-20,H=320;
  canvas.width=W*devicePixelRatio;canvas.height=H*devicePixelRatio;
  canvas.style.width=W+'px';
  ctx.scale(devicePixelRatio,devicePixelRatio);
  const m=MONTHS[Math.round(fi)%MONTHS.length];
  const pts=TICKERS.map(t=>{{
    const p=getPoint(t,m);
    return p?{{t,p,s:RAW.tickers[t].s}}:null;
  }}).filter(Boolean);
  if(!pts.length)return;

  const rets=pts.map(x=>x.p.ret);
  const vols=pts.map(x=>x.p.vol);
  const minR=Math.min(...rets)-8,maxR=Math.max(...rets)+8;
  const minV=Math.min(...vols)-3,maxV=Math.max(...vols)+5;
  const maxMC=Math.max(...pts.map(x=>x.p.mc));
  const pad={{l:55,r:15,t:15,b:38}};
  const gW=W-pad.l-pad.r,gH=H-pad.t-pad.b;
  const toX=r=>pad.l+gW*(r-minR)/(maxR-minR||1);
  const toY=v=>pad.t+gH*(1-(v-minV)/(maxV-minV||1));

  ctx.fillStyle='#0d0d12';ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='rgba(255,255,255,0.04)';ctx.lineWidth=0.5;
  for(let i=0;i<=4;i++){{
    const y=pad.t+gH*i/4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(pad.l+gW,y);ctx.stroke();
    const x=pad.l+gW*i/4;ctx.beginPath();ctx.moveTo(x,pad.t);ctx.lineTo(x,pad.t+gH);ctx.stroke();
    ctx.fillStyle='#64748b';ctx.font='9px Inter,sans-serif';
    ctx.textAlign='right';ctx.fillText(Math.round(maxV-(maxV-minV)*i/4)+'%',pad.l-4,pad.t+gH*i/4+3);
    ctx.textAlign='center';ctx.fillText(Math.round(minR+(maxR-minR)*i/4)+'%',pad.l+gW*i/4,H-4);
  }}
  if(minR<0&&maxR>0){{
    const xz=toX(0);
    ctx.strokeStyle='rgba(255,255,255,0.12)';ctx.lineWidth=1;ctx.setLineDash([4,4]);
    ctx.beginPath();ctx.moveTo(xz,pad.t);ctx.lineTo(xz,pad.t+gH);ctx.stroke();
    ctx.setLineDash([]);
  }}
  ctx.fillStyle='#64748b';ctx.font='9px Inter,sans-serif';
  ctx.textAlign='center';ctx.fillText('Rendement cumule (%)',pad.l+gW/2,H-1);
  ctx.save();ctx.translate(10,pad.t+gH/2);ctx.rotate(-Math.PI/2);ctx.fillText('Volatilite (%)',0,0);ctx.restore();

  pts.forEach((x,i)=>{{
    const px=toX(x.p.ret),py=toY(x.p.vol);
    const r=Math.sqrt(x.p.mc/maxMC)*30+5;
    const col=SCOLS[x.s]||'#888';
    const isH=i===hov;
    ctx.fillStyle=col+(isH?'ff':'99');
    ctx.strokeStyle=col;ctx.lineWidth=isH?2:1;
    ctx.beginPath();ctx.arc(px,py,isH?r+3:r,0,Math.PI*2);ctx.fill();ctx.stroke();
    if(r>10||isH){{
      ctx.fillStyle='white';
      ctx.font='bold '+Math.max(7,Math.min(11,r*0.55))+'px Inter,sans-serif';
      ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.fillText(x.t,px,py);
    }}
  }});

  document.getElementById('date-label').textContent=m;
  document.getElementById('tslider').value=Math.round(fi);
  canvas._pts=pts.map((x,i)=>{{
    const r=Math.sqrt(x.p.mc/maxMC)*30+5;
    return{{...x,px:toX(x.p.ret),py:toY(x.p.vol),r,i}};
  }});
}}

canvas.addEventListener('mousemove',e=>{{
  const rect=canvas.getBoundingClientRect();
  const mx=e.clientX-rect.left,my=e.clientY-rect.top;
  hov=-1;
  (canvas._pts||[]).forEach((x,i)=>{{
    const dx=mx-x.px,dy=my-x.py;
    if(Math.sqrt(dx*dx+dy*dy)<x.r+4){{
      hov=i;
      const c=x.p.ret>=0?'#34d399':'#fb7185';
      const r=(x.p.ret>0?'+':'')+x.p.ret.toFixed(1)+'%';
      document.getElementById('tip').innerHTML=
        '<b style="color:'+( SCOLS[x.s]||'#888')+'">'+x.t+'</b> - '+x.s+' - Rendement : <b style="color:'+c+'">'+r+'</b> - Vol : '+x.p.vol.toFixed(1)+'% - Market cap : '+x.p.mc.toFixed(0)+'B$';
    }}
  }});
  if(hov===-1)document.getElementById('tip').textContent='Survole une bulle pour voir les details';
}});
canvas.addEventListener('mouseleave',()=>{{hov=-1;}});
function togglePlay(){{playing=!playing;document.getElementById('pb').textContent=playing?'Pause':'Play';}}

const leg=document.getElementById('leg');
const seen=new Set(Object.values(RAW.tickers).map(t=>t.s));
Object.entries(SCOLS).forEach(([s,c])=>{{
  if(!seen.has(s))return;
  const sh=s==='Information Technology'?'Tech':s==='Communication Services'?'Comm.':s==='Consumer Discretionary'?'Conso':s==='Consumer Staples'?'Staples':s;
  leg.innerHTML+='<span class="li"><span class="ld" style="background:'+c+'"></span>'+sh+'</span>';
}});

function loop(){{
  if(playing)fi=(fi+spd*0.008)%MONTHS.length;
  draw();
  requestAnimationFrame(loop);
}}
loop();
</script>
</body>
</html>"""
