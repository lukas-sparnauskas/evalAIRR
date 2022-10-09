from scipy.stats import pearsonr
import argparse
import numpy as np
import subprocess

parser = argparse.ArgumentParser(prog='simAIRR_eval')
parser.add_argument('-r', '--real_dataset',
                    help='path to the real AIRR dataset', required=True)
parser.add_argument('-s', '--sim_dataset',
                    help='path to the simulated AIRR dataset', required=True)
args = parser.parse_args()

# subprocess.call(
#     'olga-generate_sequences --humanTRB -o ".\\data\\olga_repertoires\\rep_real.tsv" -n 10 --seed 121 >/dev/null 2>&1')
# subprocess.call(
#     'olga-generate_sequences --humanTRB -o ".\\data\\olga_repertoires\\rep_sim.tsv" -n 10 --seed 122 >/dev/null 2>&1')


# READ REAL ENCODED TSV

path_R = args.real_dataset
print('Real dataset: ' + path_R)
data_file_R = open(path_R, "r")

features_R = data_file_R.readline().split(',')
features_R.pop(0)
N_R = len(features_R)
print('Number of real features:', N_R)

data_R = []
for row in data_file_R:
    row = row.replace('\n', '').split(',')
    float_row = []
    for x in row:
        float_row.append(float(x))
    data_R.append(float_row)
    
data_R = np.array(data_R)

# READ SIMULATED ENCODED TSV

path_S = args.sim_dataset
print('Simulated dataset: ' + path_S)

data_file_S = open(path_S, "r")

features_S = data_file_S.readline().split(',')
features_S.pop(0)
N_S = len(features_S)
print('Number of simulated features:', N_S)

data_S = []
for row in data_file_S:
    row = row.replace('\n', '').split(',')
    float_row = []
    for x in row:
        float_row.append(float(x))
    data_S.append(float_row)

data_S = np.array(data_S)

################################
### CALCULATING CORRELATIONS ###
################################


def find_correlations(N_features, dataset):
    correlations = []
    # Iterate through features
    for i in range(N_features):
        # Iterate through every other feature
        for j in range(i + 1, N_features):
            # Calculate the Pearson correlation coefficient between the two features
            corr, _ = pearsonr(dataset[:, i], dataset[:, j])
            # If the coefficient is higher than the specified threshold, include the pair in the list
            correlations.append((i, j, corr))
            # If the list of correlations has reached the maximum number specified in the configuration, return the list
            # if len(correlations) == config['numcorr']:
            #     return correlations
    return correlations


corrs_R = find_correlations(N_R, data_R)
corrs_S = find_correlations(N_S, data_S)
print('Correlations (REAL):')
for corr in corrs_R:
    print(
        f'{features_R[corr[0]]} and {features_R[corr[1]]} correlate at a rate of', '%.3f' % corr[2])
print('Correlations (REAL):')
for corr in corrs_S:
    print(
        f'{features_S[corr[0]]} and {features_S[corr[1]]} correlate at a rate of', '%.3f' % corr[2])
