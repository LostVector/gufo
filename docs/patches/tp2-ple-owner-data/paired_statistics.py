"""Fixed ten-pair summaries. Balanced-order p is a sensitivity calculation."""
import itertools,math,random,statistics

def summarize(logs,observed,seed):
    assert len(logs)==len(observed)==10 and sorted(observed)==[-1]*5+[1]*5
    assert all(math.isfinite(x)for x in logs)
    mean=statistics.mean(logs)
    null=[statistics.mean(d*s for d,s in zip(logs,signs))for signs in itertools.product((-1,1),repeat=10)]
    balanced=[]
    for chosen in itertools.combinations(range(10),5):
        selected=set(chosen)
        balanced.append(statistics.mean(d*o*(1 if i in selected else-1)for i,(d,o)in enumerate(zip(logs,observed))))
    rng=random.Random(seed);boot=sorted(statistics.mean(rng.choices(logs,k=10))for _ in range(20000));ci=[boot[500],boot[19499]]
    return dict(pairs=10,paired_log_ratios=logs,mean_log_ratio=mean,
        paired_geometric_throughput_increase_pct=100*math.expm1(mean),
        duration_reduction_pct=100*(1-math.exp(-mean)),
        marginal_bootstrap95_throughput_increase_pct=[100*math.expm1(v)for v in ci],
        signflip_p=sum(abs(x)>=abs(mean)-1e-12 for x in null)/1024,
        signflip_assumption='Pairwise sign symmetry;1024patterns, not constrained assignment distribution',
        balanced_assignment_sensitivity_p=sum(abs(x)>=abs(mean)-1e-12 for x in balanced)/252,
        balanced_assignments=252,positive_pairs=sum(x>0 for x in logs),negative_pairs=sum(x<0 for x in logs))

def holm_primary(results):
    assert len(results)==2
    ordered=sorted(results,key=lambda x:x['balanced_assignment_sensitivity_p']);adjusted=0.0
    for i,result in enumerate(ordered):
        adjusted=max(adjusted,(2-i)*result['balanced_assignment_sensitivity_p'])
        result['holm_adjusted_p']=min(1.0,adjusted)
        result['confirmatory_gain']=result['holm_adjusted_p']<=.05 and result['marginal_bootstrap95_throughput_increase_pct'][0]>0
    return results

if __name__=='__main__':
    order=[-1]*5+[1]*5
    zero=summarize([0.0]*10,order,1);assert zero['balanced_assignment_sensitivity_p']==1 and zero['marginal_bootstrap95_throughput_increase_pct']==[0.0,0.0]
    positive=summarize([.02]*10,order,1);assert positive['balanced_assignment_sensitivity_p']==2/252
    x,y=holm_primary([zero.copy(),positive.copy()]);assert not x['confirmatory_gain']and y['confirmatory_gain']
    assert abs(y['holm_adjusted_p']-4/252)<1e-12
    print('PASS fixed-pair zero/positive/assignment-minimum and Holm sanity checks; no measurements analyzed')
