import math
import subprocess
import numpy as np
import random as rnd

input_file_name = '/home/mint/masters/data/noise_data/original.csv'
output_file_name = '/home/mint/masters/data/noise_data/with_noise.csv'
rate_of_features = 0.2
rate_of_feature_noise = 0.2
rate_of_observations = 0
rate_of_observation_noise = 0

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

def export_csv(features, data):
    with open(output_file_name, "w") as output_file:
        output_file.write(','.join(features) + '\n')
        for row in data:
            output_file.write(','.join([str(i) for i in row]) + '\n')
#endregion

#region TEST EXECUTION
orig_features, orig_data = read_encoded_csv(input_file_name)
noise_data = add_feature_noise(orig_data, rate_of_features, rate_of_feature_noise)
noise_data = add_observation_noise(noise_data, rate_of_observations, rate_of_observation_noise)
export_csv(orig_features, noise_data)
print(f'[LOG] RUNNING EVALAIRR')
subprocess.run(f'sudo evalairr -i /home/mint/masters/data/noise_data/main_config.yaml', shell=True)
#endregion