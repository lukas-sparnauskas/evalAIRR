import os
import sys
import time
import subprocess
import numpy as np
import random as rnd
import seaborn as sns
import colorcet as cc
import matplotlib.pyplot as plt

from yaml_files import immuneml_spec, evalairr_spec

max_reps = 20
max_seqs = 1000
n_runs = 3
thresholds = {
    'ks': 0.4,
    'ks_pval': 0.05,
    'dist': 22,
    'dist_obs': 140,
    'avg': 1.5e-16,
    'avg_obs': 0.15,
    'median': 0.1,
    'median_obs': 0.05,
    'var': 4e-15,
    'var_obs': 0.4,
    'std': 1.6e-15,
    'std_obs': 0.25,
    'jenshan': 0.2,
    'jenshan_obs': 0.2
}

ks_results = []

run_timestamp = int(time.time()) - 1
timestamps = []
start_time = time.time()

for n in range(0, n_runs):
    print(f'[LOG] RUNNING ITERATION {n+1}/{n_runs}')
    output_timestamp = int(time.time())
    random_n = rnd.randint(1, 9)
    min_n_seq = sys.maxsize

    timestamps.append(str(output_timestamp))
    
    ### CREATE METADATA FILES
    
    print(f'[LOG] CREATING METADATA FILES')
    real_data_list = os.listdir('/home/mint/masters/data/real_data/repertoires/')
    sim_data_list = os.listdir(f'/home/mint/masters/data/sim_data/{random_n}/simulated_repertoires/')
    rnd.shuffle(real_data_list)
    rnd.shuffle(sim_data_list)
    n_reps = min(max_reps, len(real_data_list), len(sim_data_list))
    
    real_data_list = np.array(real_data_list, dtype=str)[:n_reps]
    with open(f'/home/mint/masters/data/real_data/metadata_{output_timestamp}.csv', 'w') as file:
        file.write('filename,subject_id\n')
        for r in real_data_list:
            subject_id = r.replace('.tsv', '')
            file.write(f'{r},{subject_id}\n')
            
            data = open(f'/home/mint/masters/data/real_data/repertoires/{r}', 'r')
            data_rows = data.readlines()
            if len(data_rows) - 1 < min_n_seq:
                min_n_seq = len(data_rows) - 1
            del data_rows
            data.close()
            
    sim_data_list = np.array(sim_data_list, dtype=str)[:n_reps]
    with open(f'/home/mint/masters/data/sim_data/metadata_{run_timestamp}_{output_timestamp}.csv', 'w') as file:
        file.write('filename,subject_id\n')
        for r in sim_data_list:
            subject_id = r.replace('.tsv', '')
            file.write(f'{r},{subject_id}\n')
            
            data = open(f'/home/mint/masters/data/sim_data/{random_n}/simulated_repertoires/{r}', 'r')
            data_rows = data.readlines()
            if len(data_rows) < min_n_seq:
                min_n_seq = len(data_rows)
            del data_rows
            data.close()
            
    ### CREATE TEMP DATA FILES
    
    print(f'[LOG] CREATING TEMP DATA FILES')
    subprocess.run(f'sudo rm -r /home/mint/masters/data/real_data/temp', shell=True)
    subprocess.run(f'sudo rm -r /home/mint/masters/data/sim_data/temp', shell=True)
    subprocess.run(f'sudo mkdir /home/mint/masters/data/real_data/temp', shell=True)
    subprocess.run(f'sudo mkdir /home/mint/masters/data/sim_data/temp', shell=True)
    n_seq = min(max_seqs, min_n_seq)
    for r in real_data_list:
        source = open(f'/home/mint/masters/data/real_data/repertoires/{r}', 'r')
        rows = [next(source) for _ in range(n_seq + 1)]
        with open(f'/home/mint/masters/data/real_data/temp/{r}', 'w') as target:
            target.writelines(rows)
        source.close()
        
    for r in sim_data_list:
        source = open(f'/home/mint/masters/data/sim_data/{random_n}/simulated_repertoires/{r}', 'r')
        rows = [next(source) for _ in range(n_seq)]
        with open(f'/home/mint/masters/data/sim_data/temp/{r}', 'w') as target:
            target.writelines(rows)
        source.close()
    
    ### RUN IMMUNEML ENCODING
    
    print(f'[LOG] RUNNING IMMUNEML ENCODING')
    subprocess.run(f'sudo echo \"{immuneml_spec(run_timestamp, output_timestamp)}\" > /home/mint/masters/data/immunemldata/yaml_files/immuneml_spec_{output_timestamp}.yaml', shell=True)
    subprocess.run(f'sudo docker run -it -v /home/mint/masters/data:/data milenapavlovic/immuneml:sha-5de9c51 immune-ml /data/immunemldata/yaml_files/immuneml_spec_{output_timestamp}.yaml /data/immunemldata/output_{output_timestamp}/', shell=True)
    
    ### RUN EVALAIRR
    
    print(f'[LOG] RUNNING EVALAIRR')
    subprocess.run(f'sudo mkdir /home/mint/masters/data/evalairrdata/run_{run_timestamp}', shell=True)
    subprocess.run(f'sudo mkdir /home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_{output_timestamp}/', shell=True)
    subprocess.run(f'echo \"{evalairr_spec(run_timestamp, output_timestamp)}\" > /home/mint/masters/data/evalairrdata/yaml_files/main_yaml_{output_timestamp}.yaml', shell=True)
    subprocess.run(f'sudo evalairr -i /home/mint/masters/data/evalairrdata/yaml_files/main_yaml_{output_timestamp}.yaml', shell=True)

################
### COLLECT DATA

final_ks = dict()
final_dist = dict()
final_dist_obs = dict()
final_stat_R = dict()
final_stat_obs_R = dict()
final_stat_S = dict()
final_stat_obs_S = dict()
final_jenshan = dict()
final_jenshan_obs = dict()

for t in timestamps:
    with open(f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_{t}/ks.csv', 'r') as file:
        final_ks[t] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
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
    subprocess.run(f'sudo rm /home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_{t}/corr.csv', shell=True)

### CALCULATIONS
final_stat = dict()
final_stat_obs = dict()
t_results = {
    'ks': [],
    'ks_pval': [],
    'dist': [],
    'dist_obs': [],
    'avg': [],
    'avg_obs': [],
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
    
    t_results['ks'].append(float(np.count_nonzero(np.where(final_ks[t][0] <= thresholds['ks'], final_ks[t][0], 0))) / len(final_ks[t][0]))
    t_results['ks_pval'].append(float(np.count_nonzero(np.where(final_ks[t][1] >= thresholds['ks_pval'], final_ks[t][1], 0))) / len(final_ks[t][1]))
    t_results['dist'].append(float(np.count_nonzero(np.where(final_dist[t][0] <= thresholds['dist'], final_dist[t][0], 0))) / len(final_dist[t][0]))
    t_results['dist_obs'].append(float(np.count_nonzero(np.where(final_dist_obs[t][0] <= thresholds['dist_obs'], final_dist_obs[t][0], 0))) / len(final_dist_obs[t][0]))
    t_results['avg'].append(float(np.count_nonzero(np.where(final_stat[t][0] <= thresholds['avg'], final_stat[t][0], 0))) / len(final_stat[t][0]))
    t_results['avg_obs'].append(float(np.count_nonzero(np.where(final_stat_obs[t][0] <= thresholds['avg_obs'], final_stat_obs[t][0], 0))) / len(final_stat_obs[t][0]))
    t_results['median'].append(float(np.count_nonzero(np.where(final_stat[t][1] <= thresholds['median'], final_stat[t][1], 0))) / len(final_stat[t][1]))
    t_results['median_obs'].append(float(np.count_nonzero(np.where(final_stat_obs[t][1] <= thresholds['median_obs'], final_stat_obs[t][1], 0))) / len(final_stat_obs[t][1]))
    t_results['std'].append(float(np.count_nonzero(np.where(final_stat[t][2] <= thresholds['std'], final_stat[t][2], 0))) / len(final_stat[t][2]))
    t_results['std_obs'].append(float(np.count_nonzero(np.where(final_stat_obs[t][2] <= thresholds['std_obs'], final_stat_obs[t][2], 0))) / len(final_stat_obs[t][2]))
    t_results['var'].append(float(np.count_nonzero(np.where(final_stat[t][3] <= thresholds['var'], final_stat[t][3], 0))) / len(final_stat[t][3]))
    t_results['var_obs'].append(float(np.count_nonzero(np.where(final_stat_obs[t][3] <= thresholds['var_obs'], final_stat_obs[t][3], 0))) / len(final_stat_obs[t][3]))
    t_results['jenshan'].append(float(np.count_nonzero(np.where(final_jenshan[t][0] <= thresholds['jenshan'], final_jenshan[t][0], 0))) / len(final_jenshan[t][0]))
    t_results['jenshan_obs'].append(float(np.count_nonzero(np.where(final_jenshan_obs[t][0] <= thresholds['jenshan_obs'], final_jenshan_obs[t][0], 0))) / len(final_jenshan_obs[t][0]))

### FINAL RESULT FIGURE EXPORT
for key in ['ks_pval', 'dist', 'dist_obs', 'avg', 'avg_obs', 'median', 'median_obs', 'var', 'var_obs', 'std', 'std_obs', 'jenshan', 'jenshan_obs']:
    ab_be = 'above' if key in 'ks_pval' else 'below'
    print(f'[RESULT] Indicator:{key} - % {ab_be} threshold: {np.average(t_results[key]) * 100}')

colours = sns.color_palette(cc.glasbey, n_runs).as_hex()
def draw_kdeplot(data, title, xlabel, output, threshold, stat=None):
    f, ax = plt.subplots(1, 1)
    f.set_size_inches(9, 7)
    f.suptitle(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    for idx, t in enumerate(timestamps):
        sns.kdeplot(data[t].squeeze() if stat is None else data[t][stat].squeeze(), ax=ax, color=colours[idx], label=f'Iteration {idx + 1}')
    ax.vlines(x=threshold, ymin=0, ymax=ax.get_ylim()[1], colors='purple', ls='--', lw=2)
    ax.legend()
    f.savefig(output)
    del f
    plt.close()

# KS Statistic
title = 'Distribution of KS statistic for each iteration'
xlabel = 'Kolmogorov-Smirnov statistic'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_ks.png'
draw_kdeplot(final_ks, title, xlabel, output, thresholds['ks'], 0)

# KS P-values
title = 'Distribution of KS P-values for each iteration'
xlabel = 'Kolmogorov-Smirnov P-values'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_ks_pval.png'
draw_kdeplot(final_ks, title, xlabel, output, thresholds['ks_pval'], 1)

# Euclidean distance
title = 'Distribution of Euclidean distance between real and simulated\nfeatures for each iteration'
xlabel = 'Euclidean distance between real and simulated features'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_dist.png'
draw_kdeplot(final_dist, title, xlabel, output, thresholds['dist'])

# Observation Euclidean distance
title = 'Distribution of Euclidean distance between real and simulated\nobservations for each iteration'
xlabel = 'Euclidean distance between real and simulated observations'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_dist_obs.png'
draw_kdeplot(final_dist_obs, title, xlabel, output, thresholds['dist_obs'])

# Jensen-Shannon divergence
title = 'Distribution of the Jensen-Shannon divergence\nbetween features for each iteration'
xlabel = 'Jensen-Shannon divergence between features'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_jenshan.png'
draw_kdeplot(final_jenshan, title, xlabel, output, thresholds['jenshan'])

# Observation Jensen-Shannon divergence
title = 'Distribution of the Jensen-Shannon divergence\nbetween observations for each iteration'
xlabel = 'Jensen-Shannon divergence between observations'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_jenshan_obs.png'
draw_kdeplot(final_jenshan_obs, title, xlabel, output, thresholds['jenshan_obs'])

# Statistics
title = 'Distribution of the difference between the real and simulated\naverage feature values for each iteration'
xlabel = 'Difference between the real and simulated average feature values'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_stat_average.png'
draw_kdeplot(final_stat, title, xlabel, output, thresholds['avg'], 0)

title = 'Distribution of the difference between the real and simulated\nmedian feature values for each iteration'
xlabel = 'Difference between the real and simulated median feature values'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_stat_median.png'
draw_kdeplot(final_stat, title, xlabel, output, thresholds['median'], 1)

title = 'Distribution of the difference between the real and simulated\nfeature standard deviation for each iteration'
xlabel = 'Difference between the real and simulated feature standard deviation'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_stat_std.png'
draw_kdeplot(final_stat, title, xlabel, output, thresholds['std'], 2)

title = 'Distribution of the difference between the real and simulated\nfeature variance for each iteration'
xlabel = 'Difference between the real and simulated feature variance'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_stat_var.png'
draw_kdeplot(final_stat, title, xlabel, output, thresholds['var'], 3)

# Observation statistics
title = 'Distribution of the difference between the real and simulated\naverage observation values for each iteration'
xlabel = 'Difference between the real and simulated average observation values'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_stat_obs_average.png'
draw_kdeplot(final_stat_obs, title, xlabel, output, thresholds['avg_obs'], 0)

title = 'Distribution of the difference between the real and simulated\nmedian observation values for each iteration'
xlabel = 'Difference between the real and simulated median observation values'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_stat_obs_median.png'
draw_kdeplot(final_stat_obs, title, xlabel, output, thresholds['median_obs'], 1)

title = 'Distribution of the difference between the real and simulated\nobservation standard deviation for each iteration'
xlabel = 'Difference between the real and simulated observation standard deviation'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_stat_obs_std.png'
draw_kdeplot(final_stat_obs, title, xlabel, output, thresholds['std_obs'], 2)

title = 'Distribution of the difference between the real and simulated\nobservation variance for each iteration'
xlabel = 'Difference between the real and simulated observation variance'
output = f'/home/mint/masters/data/evalairrdata/run_{run_timestamp}/results_stat_obs_var.png'
draw_kdeplot(final_stat_obs, title, xlabel, output, thresholds['var_obs'], 3)

print(f'[LOG] EXECUTION TIME {(time.time() - start_time) / 60} m')
