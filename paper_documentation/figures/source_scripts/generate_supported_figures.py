"""Generate evidence figures from frozen Stage 6/7/8 tables and Stage 2 report statistics.
All paths are relative to repository root; no model inference occurs."""
from pathlib import Path
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'paper_documentation'; P=DOC/'figures/png'; V=DOC/'figures/pdf'; D=DOC/'figures/figure_data'
P.mkdir(parents=True,exist_ok=True);V.mkdir(parents=True,exist_ok=True);D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'savefig.dpi':300})
# Figure 8/9: paired Stage 6 test comparisons, mean ± sample SD across seeds.
df=pd.read_csv(DOC/'results/stage6_per_seed_test.csv')
metrics=[('position_rmse_m','Position RMSE (m)'),('velocity_rmse_mps','Velocity RMSE (m/s)'),('acceleration_rmse_mps2','Acceleration RMSE (m/s²)'),('six_state_mse_scaled_units','Six-state MSE (scaled units²)')]
fig,axs=plt.subplots(1,4,figsize=(10,2.7))
colors={'M0':'#0072B2','M1':'#D55E00'}
for ax,(metric,label) in zip(axs,metrics):
 for j,L in enumerate([200,400]):
  for k,v in enumerate(['M0','M1']):
   z=df[(df.sequence_length==L)&(df.variant==v)][metric]
   ax.errorbar(j+(k-.5)*.18,z.mean(),yerr=z.std(ddof=1),fmt='o',color=colors[v],capsize=3,label=v if j==0 else None)
 ax.set_xticks([0,1],['T=200','T=400']);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
axs[0].legend(frameon=False,ncol=2,fontsize=8)
fig.suptitle('Matched-length final test performance (mean ± sample SD, five seeds)');fig.tight_layout()
for ext,path in [('png',P/'stage6_matched_length_results.png'),('pdf',V/'stage6_matched_length_results.pdf')]:fig.savefig(path,bbox_inches='tight')
plt.close(fig)
# paired percent effects from Stage 6 output; points are training seed pairs; no test-population CI.
pa=pd.read_csv(DOC/'results/stage6_paired.csv')
fig,axs=plt.subplots(1,4,figsize=(10,2.8))
for ax,(m,label) in zip(axs,metrics):
 key=m
 sub=pa[pa.metric==key] if 'metric' in pa else pa.iloc[0:0]
 # raw per-seed values recomputed from paired M0/M1 rows, preserves pairing from seed field.
 for yi,L in enumerate([200,400]):
  a=df[(df.sequence_length==L)&(df.variant=='M0')].sort_values('seed')[key].to_numpy(); b=df[(df.sequence_length==L)&(df.variant=='M1')].sort_values('seed')[key].to_numpy()
  pct=100*(b-a)/a
  ax.scatter(pct,np.full(5,yi)+np.linspace(-.08,.08,5),s=20,color='#009E73')
  mean=pct.mean();sd=pct.std(ddof=1);ax.errorbar(mean,yi,xerr=sd,fmt='D',color='#222222',capsize=3)
 ax.axvline(0,color='gray',lw=.8);ax.set_yticks([0,1],['T=200','T=400']);ax.set_xlabel('M1 vs M0 change (%)');ax.set_title(label.split(' (')[0],fontsize=8);ax.grid(axis='x',alpha=.2)
fig.suptitle('Paired seed changes; diamond ±1 SD across five seeds (fixed test sets)');fig.tight_layout()
for ext,path in [('png',P/'stage7_paired_seed_effects.png'),('pdf',V/'stage7_paired_seed_effects.pdf')]:fig.savefig(path,bbox_inches='tight')
plt.close(fig)
# Temporal Stage 6 means by window. Average seed-level rows, with mean ± SD.
t=pd.read_csv(DOC/'results/stage6_temporal_windows.csv')
fig,axs=plt.subplots(2,2,figsize=(8,5));
for ax,(metric,label) in zip(axs.flat,metrics):
 for L in [200,400]:
  for v in ['M0','M1']:
   z=t[(t.sequence_length==L)&(t.variant==v)]
   if z.empty:continue
   starts=sorted(z.start_step_1based.unique()); means=[];sds=[]
   for s in starts:
    q=z[z.start_step_1based==s][metric];means.append(q.mean());sds.append(q.std(ddof=1))
   ax.plot(starts,means,label=f'{v}, T={L}',color=colors[v],linestyle='-' if L==200 else '--')
   ax.fill_between(starts,np.array(means)-np.array(sds),np.array(means)+np.array(sds),color=colors[v],alpha=.12)
 ax.set_ylabel(label);ax.set_xlabel('Window start step');ax.grid(alpha=.2)
axs[0,0].legend(fontsize=7,frameon=False,ncol=2)
fig.suptitle('Stage 6 temporal-window test metrics (mean ± sample SD across seeds)');fig.tight_layout()
for ext,path in [('png',P/'stage6_temporal_windows.png'),('pdf',V/'stage6_temporal_windows.pdf')]:fig.savefig(path,bbox_inches='tight')
plt.close(fig)
# Exact dataset quantiles plotted as medians and p95/p99, explicitly not density distributions.
st=pd.read_csv(DOC/'datasets/dataset_statistics.csv')
fig,axs=plt.subplots(1,3,figsize=(9,2.8))
for ax,(col,title,unit) in zip(axs,[('radar_range_m_p50_p95_p99_max','Radar range','m'),('speed_mps_p50_p95_p99','Speed','m/s'),('acceleration_mps2_p50_p95_p99','Acceleration magnitude','m/s²')]):
 for L,c in [(200,'#0072B2'),(400,'#D55E00')]:
  z=st[(st.sequence_length==L)&(st.split=='test')].iloc[0]
  vals=[float(x.strip()) for x in z[col].split('/')]
  ax.plot([50,95,99],vals[:3],marker='o',label=f'T={L}',color=c)
 ax.set_yscale('log');ax.set_xticks([50,95,99]);ax.set_xlabel('Percentile');ax.set_ylabel(unit);ax.set_title(title);ax.grid(alpha=.2,which='both')
axs[0].legend(frameon=False)
fig.suptitle('Stage 2 test-set distribution quantiles (log scale; pooled observations)');fig.tight_layout()
for ext,path in [('png',P/'dataset_test_quantiles.png'),('pdf',V/'dataset_test_quantiles.pdf')]:fig.savefig(path,bbox_inches='tight')
plt.close(fig)
# Copy exact input tables to figure_data with provenance-friendly names.
for src,dst in [('results/stage6_per_seed_test.csv','stage6_per_seed_test.csv'),('results/stage6_temporal_windows.csv','stage6_temporal_windows.csv'),('results/stage6_paired.csv','stage6_paired_differences.csv'),('datasets/dataset_statistics.csv','stage2_distribution_statistics.csv'),('tables/stage8_all_baselines.csv','stage8_baselines.csv')]:
 (D/dst).write_bytes((DOC/src).read_bytes())
# Architecture/pipeline/schematic diagrams using matplotlib vector primitives.
def boxes_figure(name,title,boxes,figsize=(11,2.5)):
 fig,ax=plt.subplots(figsize=figsize);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
 xs=np.linspace(.03,.83,len(boxes));w=.14 if len(boxes)>=6 else .16
 for i,(label,sub,color) in enumerate(boxes):
  x=xs[i];ax.add_patch(plt.Rectangle((x,.36),w,.38,facecolor=color,edgecolor='#333333',lw=1.1,transform=ax.transAxes))
  ax.text(x+w/2,.57,label,ha='center',va='center',weight='bold',fontsize=8,transform=ax.transAxes,wrap=True)
  ax.text(x+w/2,.42,sub,ha='center',va='center',fontsize=7,transform=ax.transAxes,wrap=True)
  if i<len(boxes)-1:ax.annotate('',xy=(xs[i+1]-.01,.55),xytext=(x+w+.005,.55),xycoords='axes fraction',arrowprops={'arrowstyle':'->','lw':1.2})
 ax.set_title(title,fontsize=11,pad=8);fig.tight_layout()
 for ext,base in [('png',P/(name+'.png')),('pdf',V/(name+'.pdf')),('svg',DOC/'figures'/(name+'.svg'))]:fig.savefig(base,bbox_inches='tight')
 plt.close(fig)
boxes_figure('research_pipeline','Research and evaluation pipeline',[
 ('Synthetic radar','fixed profile','#DCEAF7'),('Features','range/bearing + init','#DCEAF7'),('M0 / M1','M1 train-only norm','#E8E1F3'),('Causal Residual Mamba','forward only','#E8E1F3'),('Train / validate','Stage 5A / 5B','#E5F2E8'),('Frozen test','Stages 4 / 6 / 8','#FFF0D5'),('Analysis','Stage 7 / 9','#F8E0E0')],(12,2.6))
boxes_figure('architecture','Causal Residual Mamba architecture',[
 ('Radar observations','[N,2,T]','#DCEAF7'),('Feature tokens','21 channels','#DCEAF7'),('M1 only','18 normalized channels','#FCE3C4'),('Projection','21→128 + LN + SiLU','#E8E1F3'),('3 residual blocks','Mamba + FFN','#E8E1F3'),('Prediction head','128→256→128→6','#E5F2E8'),('State estimate','coarse + correction','#FFF0D5')],(12,2.8))
# Schematic coordinate plot, explicitly conceptual rather than sampled observations.
fig,ax=plt.subplots(figsize=(5.2,3.7));ax.axhline(0,color='#777',lw=.7);ax.axvline(0,color='#777',lw=.7);ax.scatter([0],[0],marker='^',s=90,color='#D55E00',label='Radar origin')
traj=np.array([[4.2,2.2],[5.0,3.0],[6.1,3.4],[7.2,3.0]])
ax.plot(traj[:,0],traj[:,1],'-o',color='#0072B2',label='Target path (schematic)');
for q in traj:ax.plot([0,q[0]],[0,q[1]],'--',color='#999',lw=.7)
ax.annotate('r',xy=(3,1.5),fontsize=10);ax.annotate('θ',xy=(1.4,.25),fontsize=10);ax.set_xlabel('x position (illustrative)');ax.set_ylabel('y position (illustrative)');ax.set_aspect('equal');ax.legend(frameon=False,fontsize=8);ax.set_title('Range–bearing geometry (schematic only)');ax.text(.5,-.18,'Illustration of implemented h(x)=[range,bearing]; not measured radar data.',transform=ax.transAxes,ha='center',fontsize=8);fig.tight_layout()
for ext,path in [('png',P/'radar_geometry_schematic.png'),('pdf',V/'radar_geometry_schematic.pdf'),('svg',DOC/'figures/radar_geometry_schematic.svg')]:fig.savefig(path,bbox_inches='tight')
plt.close(fig)
# Training curves: representative matched seed 11 per T and variant, raw logs, validation six-state MSE.
import json
fig,axs=plt.subplots(1,2,figsize=(8,3))
for ax,L in zip(axs,[200,400]):
 for variant in ['M0','M1']:
  path=ROOT/f'artifacts/results/synthetic/paper_v4_long_sequence/stage5B_matched_length_confirmation_v1/T{L}_{variant}_seed11/epoch_log.jsonl'
  if not path.exists():continue
  rec=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]; ep=[r['epoch'] for r in rec]; tr=[r['train_mse'] for r in rec]; va=[r['validation_metrics']['six_state_mse_scaled_units'] for r in rec]
  ax.plot(ep,tr,ls='--',alpha=.75,label=f'{variant} train');ax.plot(ep,va,label=f'{variant} validation')
 ax.set_yscale('log');ax.set_title(f'T={L}, seed 11');ax.set_xlabel('Epoch');ax.set_ylabel('MSE');ax.grid(alpha=.2)
axs[0].legend(frameon=False,fontsize=7,ncol=2);fig.suptitle('Representative Stage 5B learning curves (seed 11; dashed=train, solid=validation)');fig.tight_layout()
for ext,path in [('png',P/'stage5b_learning_curves_seed11.png'),('pdf',V/'stage5b_learning_curves_seed11.pdf')]:fig.savefig(path,bbox_inches='tight')
plt.close(fig)
# Baseline comparison: acceleration RMSE only; groups visibly separated by method information.
b=pd.read_csv(DOC/'tables/stage8_all_baselines.csv');fig,axs=plt.subplots(1,2,figsize=(10,3.8))
for ax,L in zip(axs,[200,400]):
 q=b[b.sequence_length==L]
 order=['M0','M1','KalmanNet','EKF','UKF']; labels=[]; vals=[]; colors_b=[]
 for model in order:
  z=q[q.model==model]
  if z.empty:continue
  variants=z.variant.unique()
  for variant in variants:
   sub=z[z.variant==variant]; labels.append(model+'\n'+variant.replace('_',' '))
   vals.append(float(sub.acceleration_rmse_mps2.mean()));colors_b.append('#D55E00' if model in ['EKF','UKF'] else '#0072B2' if model in ['M0','M1'] else '#009E73')
 x=np.arange(len(vals));ax.bar(x,vals,color=colors_b);ax.set_xticks(x,labels,rotation=65,ha='right',fontsize=6);ax.set_yscale('log');ax.set_title(f'T={L} test');ax.set_ylabel('Acceleration RMSE (m/s²; log scale)');ax.grid(axis='y',alpha=.2)
fig.suptitle('Stage 8 baseline context: blue matched-length Mamba, green T=100 KalmanNet transfer, orange model-informed filters');fig.tight_layout()
for ext,path in [('png',P/'stage8_baseline_acceleration.png'),('pdf',V/'stage8_baseline_acceleration.pdf')]:fig.savefig(path,bbox_inches='tight')
plt.close(fig)
# Preserve Stage 4 frozen-checkpoint temporal plots as supplied; PDF wraps same raster pixels.
import shutil
stage4=ROOT/'artifacts/results/synthetic/paper_v4_long_sequence/stage4_frozen_100_to_long_test_v2/plots'
for src in stage4.glob('*.png'):
 dst=P/('stage4_'+src.name)
 if src.resolve()!=dst.resolve():shutil.copy2(src,dst)
 img=plt.imread(dst)
 fig,ax=plt.subplots(figsize=(7,3.6));ax.imshow(img);ax.axis('off');fig.tight_layout(pad=0)
 fig.savefig(V/('stage4_'+src.stem+'.pdf'),bbox_inches='tight',pad_inches=0);plt.close(fig)
