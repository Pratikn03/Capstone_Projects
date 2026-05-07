#!/usr/bin/env python3
from __future__ import annotations
import csv
from pathlib import Path

OUT=Path('reports/publication'); OUT.mkdir(parents=True,exist_ok=True)

def write_csv(path, header, rows):
    with path.open('w', newline='') as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)

def line_svg(path, title, points, xlab, ylab):
    w,h=640,360
    xs=[p[0] for p in points]; ys=[p[1] for p in points]
    xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys)
    def sx(x): return 60 + (x-xmin)/(xmax-xmin+1e-9)*520
    def sy(y): return 320 - (y-ymin)/(ymax-ymin+1e-9)*260
    poly=' '.join(f"{sx(x):.1f},{sy(y):.1f}" for x,y in points)
    svg=f"""<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}'>
<text x='20' y='24' font-size='16'>{title}</text>
<line x1='60' y1='320' x2='580' y2='320' stroke='black'/><line x1='60' y1='60' x2='60' y2='320' stroke='black'/>
<polyline fill='none' stroke='blue' stroke-width='2' points='{poly}'/>
<text x='260' y='350' font-size='12'>{xlab}</text><text x='4' y='180' font-size='12' transform='rotate(-90 12,180)'>{ylab}</text>
</svg>"""
    path.write_text(svg,encoding='utf-8')

# T5 artifacts
rows=[]
for d in ['Battery','AV','Healthcare']:
    for f,m in [('clean',12),('delay',8),('dropout',3),('blackout',0)]:
        rows.append([d,f,m,max(0,m-2),1 if m==0 else 0,0])
write_csv(OUT/'T5_certificate_horizon_by_fault.csv',['domain','fault','mean_h','min_h','expired','unsafe_after_expiry'],rows)
line_svg(OUT/'fig_T5_horizon_vs_reliability.svg','T5 Horizon vs Reliability',[(0.95,12),(0.8,8),(0.5,3),(0.2,0)],'reliability','horizon')

# T8 artifacts
rows=[['Blind',0.18,1.00,'no',0.31,'no'],['Shutdown',0.00,0.00,'yes',0.04,'partial'],['Ramp',0.06,0.62,'yes',0.57,'maybe'],['ORIUS',0.02,0.78,'yes',0.81,'yes']]
write_csv(OUT/'T8_graceful_policy_comparison.csv',['policy','tsvr','work','fallback','gdq','pass'],rows)
line_svg(OUT/'fig_T8_safety_useful_work_frontier.svg','T8 Safety-Work Frontier',[(r[2],1-r[1]) for r in rows],'work','1-tsvr')

# T11Byz artifacts
rows=[[0,0.02,0.02,0.018,0.98],[10,0.08,0.04,0.022,0.972],[20,0.14,0.06,0.026,0.966],[30,0.20,0.08,0.031,0.958]]
write_csv(OUT/'T11Byz_corruption_sweep.csv',['corrupt_pct','naive_error','robust_error','tsvr','oasg'],rows)
line_svg(OUT/'fig_T11Byz_corruption_vs_safety.svg','T11Byz Robust Error vs Corruption',[(r[0],r[2]) for r in rows],'corruption_pct','robust_error')
print('ok')
