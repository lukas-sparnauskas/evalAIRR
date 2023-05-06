import os
import sys
import time
import subprocess
import numpy as np
import random as rnd
import seaborn as sns
import colorcet as cc
import matplotlib.pyplot as plt

from yaml_files import evalairr_preencoded_spec

n_runs = 10
folder_index = 10

thresholds = {
    # PERCENTILES: 80th, 90th, 95th, 99th
    'ks_feat_pval': [0.006, 0.328, 0.713, 0.965],
    'ks_obs': [0.168, 0.231, 0.277, 0.364],
    'ks_obs_pval': [1.43e-05, 0.017, 0.119, 0.592],
    'dist': [20.371, 20.527, 20.664, 20.953],
    'dist_obs': [132.647, 137.537, 142.553, 151.6],
    'mean': [3.02e-16, 5.06e-16, 7.39e-16, 1.35e-15],
    'mean_obs': [0.149, 0.208, 0.260, 0.364],
    'median': [0.092, 0.135, 0.18, 0.291],
    'median_obs': [0.084, 0.119, 0.152, 0.25],
    'var': [3.66e-15, 5.88e-15, 8.1e-15, 1.27e-14],
    'var_obs': [0.462, 0.601, 0.711, 0.917]
}

ks_results = []

run_timestamp = int(time.time()) - 1
timestamps = []
start_time = time.time()

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
    subprocess.run(f'sudo mkdir /home/mint/masters/data/evalairrdata/run_{run_timestamp}', shell=True)
    subprocess.run(f'sudo mkdir /home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_{output_timestamp}/', shell=True)
    subprocess.run(f'echo \"{evalairr_preencoded_spec(data_folder, run_timestamp, output_timestamp)}\" > /home/mint/masters/data/evalairrdata/yaml_files/main_yaml_{output_timestamp}.yaml', shell=True)
    subprocess.run(f'sudo evalairr -i /home/mint/masters/data/evalairrdata/yaml_files/main_yaml_{output_timestamp}.yaml', shell=True)

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
    with open(f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_{t}/ks_feat.csv', 'r') as file:
        final_ks_feat[t] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
    with open(f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_{t}/ks_obs.csv', 'r') as file:
        final_ks_obs[t] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
    with open(f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_{t}/dist.csv', 'r') as file:
        final_dist[t] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
    with open(f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_{t}/dist_obs.csv', 'r') as file:
        final_dist_obs[t] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
    with open(f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_{t}/real_stat.csv', 'r') as file:
        final_stat_R[t] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
    with open(f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_{t}/real_obs_stat.csv', 'r') as file:
        final_stat_obs_R[t] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
    with open(f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_{t}/sim_stat.csv', 'r') as file:
        final_stat_S[t] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
    with open(f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_{t}/sim_obs_stat.csv', 'r') as file:
        final_stat_obs_S[t] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
    with open(f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_{t}/jenshan.csv', 'r') as file:
        final_jenshan[t] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
    with open(f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_{t}/jenshan_obs.csv', 'r') as file:
        final_jenshan_obs[t] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])

### CALCULATIONS
final_stat = dict()
final_stat_obs = dict()
t_results = {
    'ks_feat': [[], [], [], []],
    'ks_feat_pval': [[], [], [], []],
    'ks_obs': [[], [], [], []],
    'ks_obs_pval': [[], [], [], []],
    'dist': [[], [], [], []],
    'dist_obs': [[], [], [], []],
    'mean': [[], [], [], []],
    'mean_obs': [[], [], [], []],
    'median': [[], [], [], []],
    'median_obs': [[], [], [], []],
    'var': [[], [], [], []],
    'var_obs': [[], [], [], []],
    'jenshan': [[], [], [], []],
    'jenshan_obs': [[], [], [], []]
}

def check_thresholds(p_idx):
    t_results['ks_feat_pval'][p_idx].append(float(np.count_nonzero(np.where(final_ks_feat[t][1] >= thresholds['ks_feat_pval'][p_idx], final_ks_feat[t][1], 0))) / len(final_ks_feat[t][1]))
    t_results['ks_obs'][p_idx].append(float(np.count_nonzero(np.where(final_ks_obs[t][0] <= thresholds['ks_obs'][p_idx], final_ks_obs[t][0], 0))) / len(final_ks_obs[t][0]))
    t_results['ks_obs_pval'][p_idx].append(float(np.count_nonzero(np.where(final_ks_obs[t][1] >= thresholds['ks_obs_pval'][p_idx], final_ks_obs[t][1], 0))) / len(final_ks_obs[t][1]))
    t_results['dist'][p_idx].append(float(np.count_nonzero(np.where(final_dist[t][0] <= thresholds['dist'][p_idx], final_dist[t][0], 0))) / len(final_dist[t][0]))
    t_results['dist_obs'][p_idx].append(float(np.count_nonzero(np.where(final_dist_obs[t][0] <= thresholds['dist_obs'][p_idx], final_dist_obs[t][0], 0))) / len(final_dist_obs[t][0]))
    t_results['mean'][p_idx].append(float(np.count_nonzero(np.where(final_stat[t][0] <= thresholds['mean'][p_idx], final_stat[t][0], 0))) / len(final_stat[t][0]))
    t_results['mean_obs'][p_idx].append(float(np.count_nonzero(np.where(final_stat_obs[t][0] <= thresholds['mean_obs'][p_idx], final_stat_obs[t][0], 0))) / len(final_stat_obs[t][0]))
    t_results['median'][p_idx].append(float(np.count_nonzero(np.where(final_stat[t][1] <= thresholds['median'][p_idx], final_stat[t][1], 0))) / len(final_stat[t][1]))
    t_results['median_obs'][p_idx].append(float(np.count_nonzero(np.where(final_stat_obs[t][1] <= thresholds['median_obs'][p_idx], final_stat_obs[t][1], 0))) / len(final_stat_obs[t][1]))
    t_results['var'][p_idx].append(float(np.count_nonzero(np.where(final_stat[t][3] <= thresholds['var'][p_idx], final_stat[t][3], 0))) / len(final_stat[t][3]))
    t_results['var_obs'][p_idx].append(float(np.count_nonzero(np.where(final_stat_obs[t][3] <= thresholds['var_obs'][p_idx], final_stat_obs[t][3], 0))) / len(final_stat_obs[t][3]))

for t in timestamps:
    final_stat[t] = np.absolute(final_stat_R[t] - final_stat_S[t])
    final_stat_obs[t] = np.absolute(final_stat_obs_R[t] - final_stat_obs_S[t])
    
    # CHECK THRESHOLDS
    
    ### 80th percentile
    check_thresholds(0)
    ### 90th percentile
    check_thresholds(1)
    ### 95th percentile
    check_thresholds(2)
    ### 99th percentile
    check_thresholds(3)

### FINAL RESULT FIGURE EXPORT
for key in ['ks_feat_pval', 'ks_obs', 'ks_obs_pval', 'dist', 'dist_obs', 'mean', 'mean_obs', 
            'median', 'median_obs', 'var', 'var_obs']:
    ab_be = 'above' if key in ['ks_feat_pval', 'ks_obs_pval'] else 'below'
    print(f'[RESULT] Indicator:{key} - % {ab_be} 80th percentile threshold: {np.mean(t_results[key][0]) * 100}')
    print(f'[RESULT] Indicator:{key} - % {ab_be} 90th percentile threshold: {np.mean(t_results[key][1]) * 100}')
    print(f'[RESULT] Indicator:{key} - % {ab_be} 95th percentile threshold: {np.mean(t_results[key][2]) * 100}')
    print(f'[RESULT] Indicator:{key} - % {ab_be} 99th percentile threshold: {np.mean(t_results[key][3]) * 100}')

colours = sns.color_palette(cc.glasbey, n_runs).as_hex()
def draw_kdeplot(data, title, xlabel, output, threshold, stat=None, exclude_t_lines=[], xbound=None):
    f, ax = plt.subplots(1, 1)
    f.set_size_inches(9, 7)
    f.suptitle(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    for idx, t in enumerate(timestamps):
        sns.kdeplot(data[t].squeeze() if stat is None else data[t][stat].squeeze(), ax=ax, color=colours[idx], label=f'Iteration {idx + 1}')
    if threshold != None:
        t_lines = []
        t1 = None if 0 in exclude_t_lines else ax.vlines(x=threshold[0], ymin=0, ymax=ax.get_ylim()[1], colors='#30007d', ls='--', lw=1)
        t2 = None if 1 in exclude_t_lines else ax.vlines(x=threshold[1], ymin=0, ymax=ax.get_ylim()[1], colors='#541ab0', ls='-.', lw=1)
        t3 = None if 2 in exclude_t_lines else ax.vlines(x=threshold[2], ymin=0, ymax=ax.get_ylim()[1], colors='#7236d1', ls='--', lw=2)
        t4 = None if 3 in exclude_t_lines else ax.vlines(x=threshold[3], ymin=0, ymax=ax.get_ylim()[1], colors='#975cf7', ls=':', lw=1)
        t_lines.append([None if 0 in exclude_t_lines else t1,
                        None if 1 in exclude_t_lines else t2,
                        None if 2 in exclude_t_lines else t3,
                        None if 3 in exclude_t_lines else t4])
        tlegend = plt.legend(t_lines[0], [None if 0 in exclude_t_lines else '80th %ile', 
                                          None if 1 in exclude_t_lines else '90th %ile', 
                                          None if 2 in exclude_t_lines else '95th %ile', 
                                          None if 3 in exclude_t_lines else '99th %ile'], 
                             loc='lower left', bbox_to_anchor=(1, 0))
        plt.gca().add_artist(tlegend)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    if xbound != None:
        ax.set_xbound(xbound[0], xbound[1])
    f.savefig(output)
    del f
    plt.close()

# Feature KS Statistic
title = 'Distribution of feature KS statistic for each iteration'
xlabel = 'Feature Kolmogorov-Smirnov statistic'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_ks_feat.png'
draw_kdeplot(final_ks_feat, title, xlabel, output, None, 0)

# Feature KS P-values
title = 'Distribution of feature KS P-values for each iteration'
xlabel = 'Feature Kolmogorov-Smirnov P-values'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_ks_feat_pval.png'
draw_kdeplot(final_ks_feat, title, xlabel, output, thresholds['ks_feat_pval'], 1)

# Observation KS Statistic
title = 'Distribution of observation KS statistic for each iteration'
xlabel = 'Observation Kolmogorov-Smirnov statistic'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_ks_obs.png'
draw_kdeplot(final_ks_obs, title, xlabel, output, thresholds['ks_obs'], 0)

# Observation KS P-values
title = 'Distribution of observation KS P-values for each iteration'
xlabel = 'Observation Kolmogorov-Smirnov P-values'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_ks_obs_pval.png'
draw_kdeplot(final_ks_obs, title, xlabel, output, thresholds['ks_obs_pval'], 1)

# Euclidean distance
title = 'Distribution of Euclidean distance between real and simulated\nfeatures for each iteration'
xlabel = 'Euclidean distance between real and simulated features'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_dist.png'
draw_kdeplot(final_dist, title, xlabel, output, thresholds['dist'], xbound=[16, 23])

# Observation Euclidean distance
title = 'Distribution of Euclidean distance between real and simulated\nobservations for each iteration'
xlabel = 'Euclidean distance between real and simulated observations'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_dist_obs.png'
draw_kdeplot(final_dist_obs, title, xlabel, output, thresholds['dist_obs'])

# Jensen-Shannon divergence
title = 'Distribution of the Jensen-Shannon divergence\nbetween features for each iteration'
xlabel = 'Jensen-Shannon divergence between features'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_jenshan.png'
draw_kdeplot(final_jenshan, title, xlabel, output, None)

# Observation Jensen-Shannon divergence
title = 'Distribution of the Jensen-Shannon divergence\nbetween observations for each iteration'
xlabel = 'Jensen-Shannon divergence between observations'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_jenshan_obs.png'
draw_kdeplot(final_jenshan_obs, title, xlabel, output, None)

# Statistics
title = 'Distribution of the difference between the real and simulated\nmean feature values for each iteration'
xlabel = 'Difference between the real and simulated mean feature values'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_stat_mean.png'
draw_kdeplot(final_stat, title, xlabel, output, thresholds['mean'], 0, [0, 1], [-0.03e-13, 0.1e-13])

title = 'Distribution of the difference between the real and simulated\nmedian feature values for each iteration'
xlabel = 'Difference between the real and simulated median feature values'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_stat_median.png'
draw_kdeplot(final_stat, title, xlabel, output, thresholds['median'], 1)

title = 'Distribution of the difference between the real and simulated\nfeature variance for each iteration'
xlabel = 'Difference between the real and simulated feature variance'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_stat_var.png'
draw_kdeplot(final_stat, title, xlabel, output, thresholds['var'], 3, xbound=[-0.2e-14, 2e-14])

# Observation statistics
title = 'Distribution of the difference between the real and simulated\nmean observation values for each iteration'
xlabel = 'Difference between the real and simulated mean observation values'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_stat_obs_mean.png'
draw_kdeplot(final_stat_obs, title, xlabel, output, thresholds['mean_obs'], 0)

title = 'Distribution of the difference between the real and simulated\nmedian observation values for each iteration'
xlabel = 'Difference between the real and simulated median observation values'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_stat_obs_median.png'
draw_kdeplot(final_stat_obs, title, xlabel, output, thresholds['median_obs'], 1)

title = 'Distribution of the difference between the real and simulated\nobservation variance for each iteration'
xlabel = 'Difference between the real and simulated observation variance'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_stat_obs_var.png'
draw_kdeplot(final_stat_obs, title, xlabel, output, thresholds['var_obs'], 3)

print(f'[LOG] EXECUTION TIME {(time.time() - start_time) / 60} m')