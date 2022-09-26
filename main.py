import argparse
import numpy as np
import subprocess

# parser = argparse.ArgumentParser(prog='simAIRR_eval')
# parser.add_argument('-r', '--real_dataset',
#                     help='path to the real AIRR dataset', required=True)
# parser.add_argument('-s', '--sim_dataset',
#                     help='path to the simulated AIRR dataset', required=True)
# args = parser.parse_args()

subprocess.call(
    'olga-generate_sequences --humanTRB -o ".\\data\\olga_repertoires\\rep_real.tsv" -n 10 --seed 121 >/dev/null 2>&1')
# subprocess.call(
#     'olga-generate_sequences --humanTRB -o ".\\data\\olga_repertoires\\rep_sim.tsv" -n 10 --seed 122 >/dev/null 2>&1')


# READ REAL ENCODED TSV

# path_R = args.real_dataset
# print('Real dataset: ' + path_R)
# data_file_R = open(path_R, "r")

# features_R = data_file_R.readline().split('\t')
# features_R.pop(0)
# N_R = len(features_R)
# print('Number of real features:', N_R)

# data_R = []
# for row in data_file_R:
#     row = row.replace('\n', '')
#     data_R.append(row.split('\t'))

# data_R = np.array(data_R)

# # READ SIMULATED ENCODED TSV

# path_S = args.sim_dataset
# print('Simulated dataset: ' + path_S)

# data_file_S = open(path_S, "r")

# features_S = data_file_S.readline().split('\t')
# features_S.pop(0)
# N_S = len(features_S)
# print('Number of simulated features:', N_S)

# data_S = []
# for row in data_file_S:
#     row = row.replace('\n', '')
#     data_S.append(row.split('\t'))

# data_S = np.array(data_S)
