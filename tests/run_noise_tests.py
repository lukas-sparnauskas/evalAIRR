import time
import math
import subprocess
import numpy as np
import random as rnd
import seaborn as sns
import colorcet as cc
import matplotlib.pyplot as plt

from yaml_files import noisetest_evalairr_spec

input_file_name = '/home/mint/masters/data/noise_data/original.csv'
output_file_name = '/home/mint/masters/data/noise_data/original_unmodified.csv'
output_noise_file_name = '/home/mint/masters/data/noise_data/with_noise.csv'
runs = 4
rate_of_features = [0.02, 0.3, 0, 0]
rate_of_feature_noise = [0.03, 0.3, 0, 0]
rate_of_observations = [0, 0, 0.02, 0.3]
rate_of_observation_noise = [0, 0, 0.03, 0.3]
run_names = [
    'Low-level noise to 2% of features',
    'High-level noise to 30% of features',
    'Low-level noise to 2% of observations',
    'High-level noise to 30% of observations'
]

final_ks_feat = dict()
final_ks_obs = dict()
final_dist_feat = dict()
final_dist_obs = dict()
final_jenshan_feat = dict()
final_jenshan_obs = dict()

start_time = time.time()

for run in range(runs):
    print(f'[LOG] RUN {run+1}/{runs}')
    
    #region FUNCTIONS
    def read_encoded_csv(csv_path):
        print('[LOG] Reading file: ' + csv_path)
        try:
            data_file = open(csv_path, "r")
            features = data_file.readline().split(',')
            features = [f.replace('\n', '').strip() for f in features]
            print(f'[LOG] Number of features in "{csv_path}" :', len(features))
            data = []
            for row in data_file:
                row = row.replace('\n', '').split(',')
                float_row = []
                for x in row:
                    float_row.append(float(x))
                data.append(float_row)
            data_file.close()
            return np.array(features), np.array(data)
        except:
            print(f'[ERROR] Failed to read file {csv_path}')
            return None, None

    def add_feature_noise(data, rate_of_features, rate_of_noise):
        if rate_of_features <= 0 or rate_of_noise <= 0:
            return data
        indeces = list(range(data.shape[1]))
        rnd.shuffle(indeces)
        features_to_transform = indeces[:math.floor(data.shape[1] * rate_of_features)]
        with_noise = np.array(data)
        min_value = np.min(data)
        max_value = np.max(data)
        data_range = np.absolute(max_value - min_value)
        for j in features_to_transform:
            for i in range(data.shape[0]):
                min_limit = max(min_value, with_noise[i,j] - data_range * rate_of_noise)
                max_limit = min(max_value, with_noise[i,j] + data_range * rate_of_noise)
                with_noise[i,j] = rnd.uniform(min_limit, max_limit)
        return with_noise

    def add_observation_noise(data, rate_of_observations, rate_of_noise):
        if rate_of_observations <= 0 or rate_of_noise <= 0:
            return data
        indeces = list(range(data.shape[0]))
        rnd.shuffle(indeces)
        observations_to_transform = indeces[:math.floor(data.shape[0] * rate_of_observations)]
        with_noise = np.array(data)
        min_value = np.min(data)
        max_value = np.max(data)
        data_range = np.absolute(max_value - min_value)
        for i in observations_to_transform:
            for j in range(data.shape[1]):
                min_limit = max(min_value, with_noise[i,j] - data_range * rate_of_noise)
                max_limit = min(max_value, with_noise[i,j] + data_range * rate_of_noise)
                with_noise[i,j] = rnd.uniform(min_limit, max_limit)
        return with_noise

    def export_csv(file_name, features, data):
        with open(file_name, "w") as output_file:
            output_file.write(','.join(features) + '\n')
            for row in data:
                output_file.write(','.join([str(i) for i in row]) + '\n')
    #endregion
      
    #region TEST EXECUTION
    orig_features, orig_data = read_encoded_csv(input_file_name)
    noise_data = add_feature_noise(orig_data, rate_of_features[run], rate_of_feature_noise[run])
    noise_data = add_observation_noise(noise_data, rate_of_observations[run], rate_of_observation_noise[run])
    export_csv(output_file_name, orig_features, orig_data)
    export_csv(output_noise_file_name, orig_features, noise_data)
    print(f'[LOG] RUNNING EVALAIRR')
    subprocess.run(f'echo \"{noisetest_evalairr_spec(run)}\" > /home/mint/masters/data/noise_data/main_config_{run}.yaml', shell=True)
    subprocess.run(f'sudo evalairr -i /home/mint/masters/data/noise_data/main_config_{run}.yaml', shell=True)

    # READ CSV FILES
    with open(f'/home/mint/masters/data/noise_data/results/ks_feat.csv', 'r') as file:
        final_ks_feat[run] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
    with open(f'/home/mint/masters/data/noise_data/results/ks_obs.csv', 'r') as file:
        final_ks_obs[run] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
    with open(f'/home/mint/masters/data/noise_data/results/dist_feat.csv', 'r') as file:
        final_dist_feat[run] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
    with open(f'/home/mint/masters/data/noise_data/results/dist_obs.csv', 'r') as file:
        final_dist_obs[run] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
    with open(f'/home/mint/masters/data/noise_data/results/jenshan_feat.csv', 'r') as file:
        final_jenshan_feat[run] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])
    with open(f'/home/mint/masters/data/noise_data/results/jenshan_obs.csv', 'r') as file:
        final_jenshan_obs[run] = np.array([[float(val) for val in row.replace('\n', '').split(',')] for row in file.readlines()])

def draw_kdeplot(data, title, xlabel, file_name, stat=None):
    colours = sns.color_palette(cc.glasbey, runs).as_hex()
    f, ax = plt.subplots(1, 1)
    f.set_size_inches(9, 7)
    f.suptitle(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    for run in range(runs):
        sns.kdeplot(data[run].squeeze() if stat is None else data[run][stat].squeeze(), ax=ax, color=colours[run], label=run_names[run])
    ax.legend()
    f.savefig(f'/home/mint/masters/data/noise_data/results/{file_name}')
    del f
    plt.close()

# DRAW KS PLOTS
draw_kdeplot(final_ks_feat,
             'Distribution of feature KS statistic for original and noisy datasets', 
             'Feature Kolmogorov-Smirnov statistic', 
             'ks_feat.png', 0)
draw_kdeplot(final_ks_obs, 
             'Distribution of observation KS statistic for original and noisy datasets', 
             'Observation Kolmogorov-Smirnov statistic', 
             'ks_obs.png', 0)
draw_kdeplot(final_dist_feat, 
             'Distribution of Euclidean distance between features\nin the original and noisy datasets', 
             'Euclidean distance between features', 
             'dist_feat.png')
draw_kdeplot(final_dist_obs, 
             'Distribution of Euclidean distance between observations\nin the original and noisy datasets', 
             'Euclidean distance between observations', 
             'dist_obs.png')
draw_kdeplot(final_jenshan_feat, 
             'Distribution of feature Jensen-Shannon divergence for original and noisy datasets', 
             'Feature Jensen-Shannon divergence', 
             'jenshan_feat.png')
draw_kdeplot(final_jenshan_obs, 
             'Distribution of observation Jensen-Shannon divergence for original and noisy datasets', 
             'Observation Jensen-Shannon divergence', 
             'jenshan_obs.png')

print(f'[LOG] EXECUTION TIME {(time.time() - start_time) / 60} m')
#endregion