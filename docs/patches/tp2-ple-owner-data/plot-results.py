from pathlib import Path
import json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parent;s=json.loads((r/'statistical-summary.json').read_text());fig,axes=plt.subplots(2,2,figsize=(11,7.5))
titles=['First C1 server decode','Short C2 complete client wave','Repeated C1 server decode','Coding C2 complete client wave']
for ax,m,title in zip(axes.flat,s['metrics'],titles):
 for a,b in zip(m['baseline_values'],m['candidate_values']):ax.plot([0,1],[a,b],'-o',color='#53779a',alpha=.45,linewidth=.9,markersize=3)
 ax.plot([0,1],[m['baseline_geometric_mean'],m['candidate_geometric_mean']],'-o',color='#172d3b',linewidth=2.5,markersize=6,label='Geometric mean')
 low,high=m['marginal_bootstrap95_throughput_increase_pct'];tag='primary'if m['primary']else'exploratory'
 ax.set_title(f'{title} ({tag})\n{m["paired_geometric_throughput_increase_pct"]:+.2f}% · 95% CI [{low:+.2f}, {high:+.2f}]%',fontsize=10)
 ax.set_xticks([0,1],['Replicated PLE','Owner reads']);ax.set_ylabel('tokens / second');ax.grid(axis='y',alpha=.2);ax.margins(x=.18,y=.15)
fig.suptitle('Two Strix Halo nodes: cooperative PLE owner reads',fontsize=14)
fig.text(.5,.025,'10 fixed balanced pairs · UD-Q4_K_XL + shared Q8_0 MTP · context capacity 65,536 · 2 sessions\nAll 240 outputs/cache identities match; C2 work changed in 26 request comparisons. All 27 timing flags retained.\nPrimary Holm-adjusted p: C1 0.0159; short C2 0.0317. Marginal intervals; no repeated-C1 gain established.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.10,1,.95]);fig.savefig(r/'before-after.png',dpi=180);fig.savefig(r/'before-after.pdf');plt.close(fig)
