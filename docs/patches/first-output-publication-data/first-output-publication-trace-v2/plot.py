from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
p=Path(__file__).resolve().parent;d=json.loads((p/'publication-analysis.json').read_text())
selected=[next(r for r in d['records']if r['arm']=='baseline'and r['case']=='C2-after-preview'and r['actor']==0),next(r for r in d['records']if r['arm']=='candidate'and r['case']=='C2'and r['actor']==1)]
fig,axes=plt.subplots(2,1,figsize=(10,5.4),constrained_layout=True)
colors={'prefill':'#b56a13','single_decode':'#287ca0','runner_batch':'#7654a3'}
for ax,r in zip(axes,selected):
 for s in r['overlapping_work']:
  if s['unit']not in colors:continue
  start=s['start_relative_first_piece_ms'];end=start+(s['end']-s['start'])/1e6;a=max(0,start);b=min(r['server_piece_gap_ms'],end)
  if b<=a:continue
  ax.broken_barh([(a,b-a)],(.3,.45),facecolors=colors[s['unit']])
  if b-a>15:ax.text((a+b)/2,.525,f"{s['unit'].replace('_',' ')}\n{b-a:.1f} ms",ha='center',va='center',color='white',fontsize=9)
 for step in r['ready_steps_during_gap']:
  t=step['after_first_piece_ms']
  if step['novel_publications']==0:
   ax.plot(t,.85,'x',color='#b22222',markersize=9);ax.annotate('preview only; no new token',(t,.85),xytext=(t+6,1.05),fontsize=9,arrowprops=dict(arrowstyle='-',color='#b22222'))
 ax.axvline(0,color='#333333',linestyle=':');ax.axvline(r['server_piece_gap_ms'],color='#333333',linestyle=':')
 ax.set_title(f"{r['arm']}, {r['case']}, actor {r['actor']}: first → second nonempty queued piece {r['server_piece_gap_ms']:.1f} ms",loc='left',fontsize=10)
 ax.set_xlim(-3,225);ax.set_ylim(0,1.35);ax.set_yticks([]);ax.set_xlabel('Wall time after first queued piece (ms)');ax.grid(axis='x',alpha=.2)
fig.suptitle('Observed output gaps span inference work, not a 100 ms ready-token delivery delay',fontsize=11)
fig.text(.01,-.025,'Diagnostic CPU wall spans include GPU/I/O work. Separate controlled arrivals; not a performance comparison. Gaps not covered by shown spans remain visible.',fontsize=8)
fig.savefig(p/'publication-timeline.svg',bbox_inches='tight');fig.savefig(p/'publication-timeline.png',dpi=160,bbox_inches='tight')
