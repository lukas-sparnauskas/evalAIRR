import os
import sys
import time
import subprocess
import numpy as np
import random as rnd
import seaborn as sns
import colorcet as cc
import matplotlib.pyplot as plt

from yaml_files import threshold_test_preencoded_evalairr_spec

n_runs = 10
folder_index = 20

ks_results = []

run_timestamp = int(time.time()) - 1
timestamps = []
start_time = time.time()

print(f'[LOG] GETTING ENCODED DATA FILES')
data_list = sorted(os.listdir('/home/mint/masters/data/immunemldata/'))
print('[LOG] RUNNING WITH PREENCODED FILES:', data_list[folder_index:folder_index+10])
for n in range(0, n_runs):
    print(f'[LOG] RUNNING ITERATION {n+1}/{n_runs}')
    data_folder = data_list[folder_index]
    folder_index += 1
    output_timestamp = int(time.time())
    timestamps.append(str(output_timestamp))
    
    ### RUN EVALAIRR
    
    print(f'[LOG] RUNNING EVALAIRR')
    subprocess.run(f'sudo mkdir /home/mint/masters/data/evalairrdata/th_run_{run_timestamp}', shell=True)
    subprocess.run(f'sudo mkdir /home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_{output_timestamp}/', shell=True)
    subprocess.run(f'echo \"{threshold_test_preencoded_evalairr_spec(data_folder, run_timestamp, output_timestamp)}\" > /home/mint/masters/data/evalairrdata/yaml_files/th_main_yaml_{output_timestamp}.yaml', shell=True)
    subprocess.run(f'sudo evalairr -i /home/mint/masters/data/evalairrdata/yaml_files/th_main_yaml_{output_timestamp}.yaml', shell=True)

################
### COLLECT DATA

final_ks_feat = dict()
final_ks_obs = dict()
final_dist = dict()
final_dist_obs = dict()
final_stat_R = dict()
final_stat_obs_R = dict()
final_stat_S = dict()
final_stat_obs_S = dict()
final_jenshan = dict()
final_jenshan_obs = dict()

for t in timestamps:
    with open(f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_{t}/ks_feat.csv', 'r') as file:
        final_ks_feat[t] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
    with open(f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_{t}/ks_obs.csv', 'r') as file:
        final_ks_obs[t] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
    with open(f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_{t}/dist.csv', 'r') as file:
        final_dist[t] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
    with open(f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_{t}/dist_obs.csv', 'r') as file:
        final_dist_obs[t] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
    with open(f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_{t}/real_stat.csv', 'r') as file:
        final_stat_R[t] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
    with open(f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_{t}/real_obs_stat.csv', 'r') as file:
        final_stat_obs_R[t] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
    with open(f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_{t}/sim_stat.csv', 'r') as file:
        final_stat_S[t] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
    with open(f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_{t}/sim_obs_stat.csv', 'r') as file:
        final_stat_obs_S[t] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
    with open(f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_{t}/jenshan.csv', 'r') as file:
        final_jenshan[t] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
    with open(f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_{t}/jenshan_obs.csv', 'r') as file:
        final_jenshan_obs[t] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])

### CALCULATIONS
final_stat = dict()
final_stat_obs = dict()
t_results = {
    'ks_feat': [],
    'ks_feat_pval': [],
    'ks_obs': [],
    'ks_obs_pval': [],
    'dist': [],
    'dist_obs': [],
    'mean': [],
    'mean_obs': [],
    'median': [],
    'median_obs': [],
    'var': [],
    'var_obs': [],
    'std': [],
    'std_obs': [],
    'jenshan': [],
    'jenshan_obs': []
}
for t in timestamps:
    final_stat[t] = np.absolute(final_stat_R[t] - final_stat_S[t])
    final_stat_obs[t] = np.absolute(final_stat_obs_R[t] - final_stat_obs_S[t])
    
    t_results['ks_feat'].append(final_ks_feat[t][0])
    t_results['ks_feat_pval'].append(final_ks_feat[t][1])
    t_results['ks_obs'].append(final_ks_obs[t][0])
    t_results['ks_obs_pval'].append(final_ks_obs[t][1])
    t_results['dist'].append(final_dist[t])
    t_results['dist_obs'].append(final_dist_obs[t])
    t_results['mean'].append(final_stat[t][0])
    t_results['mean_obs'].append(final_stat_obs[t][0])
    t_results['median'].append(final_stat[t][1])
    t_results['median_obs'].append(final_stat_obs[t][1])
    t_results['std'].append(final_stat[t][2])
    t_results['std_obs'].append(final_stat_obs[t][2])
    t_results['var'].append(final_stat[t][3])
    t_results['var_obs'].append(final_stat_obs[t][3])
    t_results['jenshan'].append(final_jenshan[t])
    t_results['jenshan_obs'].append(final_jenshan_obs[t])

### FINAL RESULT FIGURE EXPORT
for key in ['ks_feat', 'ks_feat_pval', 'ks_obs', 'ks_obs_pval', 'dist', 'dist_obs', 'mean', 'mean_obs', 
            'median', 'median_obs', 'var', 'var_obs', 'std', 'std_obs', 'jenshan', 'jenshan_obs']:    
    print(f'[RESULT] Indicator:{key} - 80th percentile: {np.percentile(np.hstack(t_results[key]), 80)}')
    print(f'[RESULT] Indicator:{key} - 90th percentile: {np.percentile(np.hstack(t_results[key]), 90)}')
    print(f'[RESULT] Indicator:{key} - 95th percentile: {np.percentile(np.hstack(t_results[key]), 95)}')
    print(f'[RESULT] Indicator:{key} - 99th percentile: {np.percentile(np.hstack(t_results[key]), 99)}')

colours = sns.color_palette(cc.glasbey, n_runs).as_hex()
def draw_kdeplot(data, title, xlabel, output, stat=None):
    f, ax = plt.subplots(1, 1)
    f.set_size_inches(9, 7)
    f.suptitle(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    for idx, t in enumerate(timestamps):
        sns.kdeplot(data[t].squeeze() if stat is None else data[t][stat].squeeze(), ax=ax, color=colours[idx], label=f'Iteration {idx + 1}')
    ax.legend()
    f.savefig(output)
    del f
    plt.close()

# Feature KS Statistic
title = 'Distribution of feature KS statistic for each iteration'
xlabel = 'Feature Kolmogorov-Smirnov statistic'
output = f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_ks_feat.png'
draw_kdeplot(final_ks_feat, title, xlabel, output, 0)

# Feature KS P-values
title = 'Distribution of feature KS P-values for each iteration'
xlabel = 'Feature Kolmogorov-Smirnov P-values'
output = f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_ks_feat_pval.png'
draw_kdeplot(final_ks_feat, title, xlabel, output, 1)

# Observation KS Statistic
title = 'Distribution of observation KS statistic for each iteration'
xlabel = 'Observation Kolmogorov-Smirnov statistic'
output = f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_ks_obs.png'
draw_kdeplot(final_ks_obs, title, xlabel, output, 0)

# Observation KS P-values
title = 'Distribution of observation KS P-values for each iteration'
xlabel = 'Observation Kolmogorov-Smirnov P-values'
output = f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_ks_obs_pval.png'
draw_kdeplot(final_ks_obs, title, xlabel, output, 1)

# Euclidean distance
title = 'Distribution of Euclidean distance between \nfeatures for each iteration'
xlabel = 'Euclidean distance between features'
output = f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_dist.png'
draw_kdeplot(final_dist, title, xlabel, output)

# Observation Euclidean distance
title = 'Distribution of Euclidean distance between\nobservations for each iteration'
xlabel = 'Euclidean distance between observations'
output = f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_dist_obs.png'
draw_kdeplot(final_dist_obs, title, xlabel, output)

# Jensen-Shannon divergence
title = 'Distribution of the Jensen-Shannon divergence\nbetween features for each iteration'
xlabel = 'Jensen-Shannon divergence between features'
output = f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_jenshan.png'
draw_kdeplot(final_jenshan, title, xlabel, output)

# Observation Jensen-Shannon divergence
title = 'Distribution of the Jensen-Shannon divergence\nbetween observations for each iteration'
xlabel = 'Jensen-Shannon divergence between observations'
output = f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_jenshan_obs.png'
draw_kdeplot(final_jenshan_obs, title, xlabel, output)

# Statistics
title = 'Distribution of the difference between the\nmean feature values for each iteration'
xlabel = 'Difference between the mean feature values'
output = f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_stat_mean.png'
draw_kdeplot(final_stat, title, xlabel, output, 0)

title = 'Distribution of the difference between the\nmedian feature values for each iteration'
xlabel = 'Difference between the median feature values'
output = f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_stat_median.png'
draw_kdeplot(final_stat, title, xlabel, output, 1)

title = 'Distribution of the difference between the\nfeature standard deviation for each iteration'
xlabel = 'Difference between the feature standard deviation'
output = f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_stat_std.png'
draw_kdeplot(final_stat, title, xlabel, output, 2)

title = 'Distribution of the difference between the\nfeature variance for each iteration'
xlabel = 'Difference between the feature variance'
output = f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_stat_var.png'
draw_kdeplot(final_stat, title, xlabel, output, 3)

# Observation statistics
title = 'Distribution of the difference between the\nmean observation values for each iteration'
xlabel = 'Difference between the mean observation values'
output = f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_stat_obs_mean.png'
draw_kdeplot(final_stat_obs, title, xlabel, output, 0)

title = 'Distribution of the difference between the\nmedian observation values for each iteration'
xlabel = 'Difference between median observation values'
output = f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_stat_obs_median.png'
draw_kdeplot(final_stat_obs, title, xlabel, output, 1)

title = 'Distribution of the difference between the\nobservation standard deviation for each iteration'
xlabel = 'Difference between the observation standard deviation'
output = f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_stat_obs_std.png'
draw_kdeplot(final_stat_obs, title, xlabel, output, 2)

title = 'Distribution of the difference between the\nobservation variance for each iteration'
xlabel = 'Difference between the observation variance'
output = f'/home/mint/masters/data/evalairrdata/th_run_{run_timestamp}/results_stat_obs_var.png'
draw_kdeplot(final_stat_obs, title, xlabel, output, 3)

print(f'[LOG] EXECUTION TIME {(time.time() - start_time) / 60} m')
