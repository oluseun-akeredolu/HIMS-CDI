"""Verify the numbers that appear in the manuscript tables and examples."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from statistics import mean, stdev

p=Path(__file__).resolve().parent
r=p/'results'
def read(f): return pd.read_csv(r/(f+'.csv'))
results={}
def equal(label,observed,expected):
    assert observed==expected, (label,observed,expected)
    results[label]=observed
d=read('decision_records')
t=read('execution_log')
a=read('postgate_resources')
m=read('correlation_matrix')
equal('Table 8 gate marginals',
      [list(Counter(d[f'G{i}_observed'])[v] for v in ['PASS','FAIL','UNKNOWN']) for i in range(1,5)],
      [[213,16,11],[73,59,108],[127,113,0],[133,70,37]])
lb=t.latency_ms>=200
mb=t.memory_mb>=100
equal('Section 5.1 resource breaches',[int(lb.sum()),int(mb.sum()),int((lb|mb).sum()),int((lb&mb).sum())],[1274,27,1296,5])
per=(lb|mb).groupby(t.environment_id).sum()
equal('Sections 5.1 and 5.3 environment resource patterns',
      [int((per==0).sum()),int(((per>0)&(per<30)).sum()),int((per==30).sum()),
       int(((d.latency_p95_ms<200)&(d.latency_exceedance_count>0)).sum()),
       int(((d.G3_observed=="PASS")&(d.latency_exceedance_count>0)).sum())],
      [107,133,0,24,20])
equal('Postgate percentile or memory ranges',
      [[round(a[c].min(),2),round(a[c].max(),2)] for c in ['latency_median_ms','latency_p95_ms','latency_p99_ms','memory_peak_mb']],
      [[89.64,165.06],[111.72,199.49],[121.67,221.70],[45.81,95.44]])
points=list(a[['latency_median_ms','memory_peak_mb']].itertuples(index=False,name=None))
count=sum((x[0]<=y[0] and x[1]<=y[1] or y[0]<=x[0] and y[1]<=x[1]) and x!=y for x,y in combinations(points,2))
equal('Postgate comparable dominance pairs',count,39)
equal('Postgate worked example values',[[round(v,2) for v in row] for row in a.set_index('environment_id')
      .loc[['ENV004','ENV021'],['latency_median_ms','memory_peak_mb']].itertuples(index=False,name=None)],[[101.84,82.19],[96.51,95.44]])
equal('Table 10 threshold grid',[len(read('threshold_sensitivity')),int(read('threshold_sensitivity').eligible_n.min()),
      int(read('threshold_sensitivity').eligible_n.max())],[20,2,32])
s=read('evidence_injection_sensitivity')
equal('Table 10 T2 disposition trend',s.eligible_n.tolist(),[15,14,10,5])
equal('Table 10 T2 final gate totals',s.iloc[-1][['g2_pass_n','g2_fail_n','g2_unknown_n','g4_pass_n','g4_fail_n','g4_unknown_n']].astype(int).tolist(),[24,59,157,99,70,71])
equal('Table 10 no UNKNOWN to ELIGIBLE conversions',s.unknown_to_eligible_conversions.tolist(),[0,0,0,0])
ms=read('cv_metrics_by_repeat')
measurements=['balanced_accuracy','sensitivity','specificity','macro_f1','auroc','average_precision']
model_list=ms.model.unique()
mnames=[name for name in model_list if name!='oracle']
summary=[[ [round(mean(row[metric]),3),round(stdev(row[metric]),3)] for metric in measurements]
         for name in mnames for row in [ms[ms.model==name]]]
equal('Table 9 displayed means and sample standard deviations',
      summary,[[[.903,.027],[.813,.053],[.992,.003],[.917,.023],[.992,.002],[.851,.032]],
               [[.500,.0],[.0,.0],[1.,.0],[.484,.0],[.572,.037],[.077,.007]],
               [[.547,.0],[.533,.0],[.560,.0],[.418,.0],[.563,.0],[.088,.0]]])
matrix=m.iloc[:,1:].to_numpy() if m.shape==(20,21) else m.to_numpy()
equal('Section 4.3 factor matrix minimum eigenvalue',round(float(np.linalg.eigvalsh(matrix)[0]),8),.75)
print(json.dumps(results,indent=2))
print('Checked numerical claims:',len(results))
