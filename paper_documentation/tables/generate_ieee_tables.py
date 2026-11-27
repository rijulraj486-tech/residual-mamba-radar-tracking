from pathlib import Path
import csv
R=Path('.');D=R/'paper_documentation'
def read(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rs):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def tex(name,rs,cols,caption,label,maxrows=30):
 def esc(s): return str(s).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%').replace('#',r'\#').replace('nan','')
 lines=[r'\begin{table*}[t]',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\scriptsize',r'\begin{tabular}{'+'l'*len(cols)+'}',r'\hline',' & '.join(c.replace('_',' ').title() for c in cols)+r' \\',r'\hline']
 for row in rs[:maxrows]:lines.append(' & '.join(esc(row.get(c,'')) for c in cols)+r' \\')
 if len(rs)>maxrows:lines.append(r'\multicolumn{'+str(len(cols))+r'}{l}{First '+str(maxrows)+r' rows shown; full CSV is authoritative.} \\')
 lines += [r'\hline',r'\end{tabular}',r'\end{table*}']
 (D/'tables'/name).write_text('\n'.join(lines)+'\n')
def sourcecsv(rel,source):
 rows=read(Path(source));writecsv(D/rel,rows);return rows
rows=read(D/'datasets/dataset_split_summary.csv');writecsv(D/'tables/dataset_splits.csv',rows)
tex('dataset_splits.tex',rows,['sequence_length','split','n_trajectories','split_seed'],'Dataset split sizes and seeds','tab:splits')
orig=read(Path('artifacts/results/synthetic/five_seed_n1024_fixedpool_v1/summary/per_seed_metrics.csv'));writecsv(D/'tables/original_T100_results.csv',orig)
tex('original_T100_results.tex',orig,['seed','method','position_rmse_m','velocity_rmse_mps','acceleration_rmse_mps2','six_state_mse_scaled_units'],'Original T=100 five-seed test results','tab:t100')
s4=sourcecsv('tables/stage4_results.csv',D/'results/stage4_per_seed.csv')
tex('stage4_extrapolation.tex',s4,['variant','seed','sequence_length','position_rmse_m','velocity_rmse_mps','acceleration_rmse_mps2','six_state_mse_scaled_units'],'Stage 4 frozen T=100 checkpoint extrapolation','tab:stage4',24)
s5a=sourcecsv('tables/stage5a_screening.csv',D/'results/stage5A_screening.csv')
tex('stage5a_screening.tex',s5a,['variant','sequence_length','selected_epoch','validation_position_rmse_m','validation_velocity_rmse_mps','validation_acceleration_rmse_mps2','validation_six_state_mse_scaled_units'],'Stage 5A one-seed validation screen','tab:stage5a')
s5b=read(D/'tables/stage5b_validation.csv')
tex('stage5b_validation.tex',s5b,['sequence_length','variant','seed','selected_epoch','validation_position_rmse_m','validation_velocity_rmse_mps','validation_acceleration_rmse_mps2','validation_six_state_mse_scaled_units'],'Stage 5B matched-length validation outcomes','tab:stage5b',30)
s6=read(D/'tables/stage6_full_results.csv')
tex('stage6_test_results.tex',s6,['sequence_length','variant','seed','position_rmse_m','velocity_rmse_mps','acceleration_rmse_mps2','six_state_mse_scaled_units'],'Stage 6 final held-out test results by seed','tab:stage6',30)
s7=read(D/'tables/stage7_inference.csv')
tex('stage7_statistics.tex',s7,['sequence_length','metric','paired_t_p_two_sided','wilcoxon_exact_sign_flip_p_two_sided','holm_adjusted_p_t_family','holm_adjusted_p_wilcoxon_family'],'Stage 7 exploratory paired tests (n=5 seed pairs)','tab:stage7')
s8=read(D/'tables/stage8_all_baselines.csv')
for L in [200,400]:
 z=[r for r in s8 if int(float(r['sequence_length']))==L]
 writecsv(D/f'tables/stage8_T{L}_results.csv',z)
 tex(f'stage8_T{L}_results.tex',z,['model','method_group','variant','seed','acceleration_rmse_mps2','position_rmse_m','velocity_rmse_mps','six_state_mse_scaled_units','runtime_device'],f'Stage 8 baseline comparison at T={L}; method groups indicate information access',f'tab:stage8T{L}',24)
rr=read(D/'tables/runtime_complexity.csv');tex('runtime_complexity.tex',rr,['sequence_length','method','parameter_count','runtime_mean_seconds','runtime_sd_seconds','device'],'Runtime and model complexity; device-specific timings','tab:runtime',30)
hp=read(D/'training/complete_hyperparameter_table.csv');tex('training_hyperparameters.tex',hp,['experiment','T','variant','optimizer','learning_rate','weight_decay','batch_size','max_epochs','early_stopping_patience','min_delta'],'Training hyperparameters by experiment family','tab:training',20)
