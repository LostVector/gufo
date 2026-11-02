from pathlib import Path
import json,zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
with zipfile.ZipFile(p/'acquisition.zip')as z:d=json.loads(z.read('early-end-release-comparison-v2/statistical-summary.json'))
fig,ax=plt.subplots(1,2,figsize=(10,4.4))
for i,(b,c)in enumerate(zip(d['primary']['values_ms']['baseline'],d['primary']['values_ms']['candidate'])):
 ax[0].plot([0,1],[b,c],color='#999999',alpha=.55,lw=1)
ax[0].scatter([0]*10,d['primary']['values_ms']['baseline'],color='#ba4b37',label='Baseline',zorder=3)
ax[0].scatter([1]*10,d['primary']['values_ms']['candidate'],color='#137d91',label='Early End',zorder=3)
ax[0].set(xticks=[0,1],xticklabels=['Baseline e250','Early End dd9449'],yscale='log',ylabel='Last content → HTTP finish (ms, log scale)',title='Primary: short C2 actor 1')
ax[0].set_xlim(-.35,1.35);ax[0].grid(axis='y',alpha=.2)
ax[0].text(.5,.55,'42.0 → 0.46 ms\n98.9% paired reduction',ha='center',transform=ax[0].transAxes,bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
for i,(key,s)in enumerate(d['secondary'].items()):
 est=s['paired_rate_change_pct'];lo,hi=s['marginal_bootstrap95_rate_change_pct'];ax[1].errorbar(est,i,xerr=[[est-lo],[hi-est]],fmt='o',color='#536575',capsize=4)
ax[1].axvline(0,color='#999999',lw=1);ax[1].set(yticks=[0,1,2],yticklabels=['C2 whole wave','C1 first decode','C1 repeat decode'],xlabel='Paired rate change (%) with marginal 95% CI',title='Secondary rates: inconclusive');ax[1].invert_yaxis();ax[1].grid(axis='x',alpha=.2)
fig.suptitle('Native TP2 · 20 fresh untraced starts · 10 fixed balanced-order pairs',fontsize=12)
fig.text(.5,.015,'UD-Q4_K_XL + Q8_0 MTP · 65,536 context · 2 slots · chunk 2,048 · 1 GiB cache\nAll 65 phase flags and 37 work differences retained; no throughput gain or equivalence claim.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.09,1,.94))
for ext in('png','pdf'):fig.savefig(p/f'completion-tail.{ext}',dpi=170)
