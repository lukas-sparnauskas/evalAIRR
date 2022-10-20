import math
import seaborn
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

#######################
### PARSE ARGUMENTS ###
#######################

parser = argparse.ArgumentParser(prog='simAIRR_eval')
parser.add_argument('-r', '--real_dataset',
                    help='path to the real AIRR dataset', required=True)
parser.add_argument('-s', '--sim_dataset',
                    help='path to the simulated AIRR dataset', required=True)
args = parser.parse_args()


#############################
### READ REAL ENCODED CSV ###
#############################

path_R = args.real_dataset
print('Real dataset: ' + path_R)
data_file_R = open(path_R, "r")

features_R = data_file_R.readline().split(',')
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


##################################
### READ SIMULATED ENCODED CSV ###
##################################

path_S = args.sim_dataset
print('Simulated dataset: ' + path_S)

data_file_S = open(path_S, "r")

features_S = data_file_S.readline().split(',')
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


##################
### UNIVARIATE ###
##################

####################
### MULTIVARIATE ###
####################

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

# corrs_R = find_correlations(N_R, data_R)
# corrs_S = find_correlations(N_S, data_S)
# print('Correlations (REAL):')
# for corr in corrs_R:
#     print(
#         f'{features_R[corr[0]]} and {features_R[corr[1]]} correlate at a rate of', '%.3f' % corr[2])
# print('Correlations (REAL):')
# for corr in corrs_S:
#     print(
#         f'{features_S[corr[0]]} and {features_S[corr[1]]} correlate at a rate of', '%.3f' % corr[2])


# corr_R = pd.DataFrame(data_R, columns=features_R).corr()
# corr_S = pd.DataFrame(data_S, columns=features_S).corr()

# plt.figure(figsize=(20, 20))
# seaborn.heatmap(corr_R, cmap='YlGnBu')  # , mask=(np.abs(corr_R) >= 0.5))
# plt.show()

# plt.figure(figsize=(20, 20))
# seaborn.heatmap(corr_S, cmap='YlGnBu')  # , mask=(np.abs(corr_S) >= 0.5))
# plt.show()


###########
### PCA ###
###########

def center_data(A):
    # INPUT:
    # A    [NxM] numpy data matrix (N samples, M features)
    #
    # OUTPUT:
    # X    [NxM] numpy centered data matrix (N samples, M features)
    m = np.mean(A, axis=0)
    X = A - m
    return X

def compute_covariance_matrix(A):
    # INPUT:
    # A    [NxM] centered numpy data matrix (N samples, M features)
    #
    # OUTPUT:
    # C    [MxM] numpy covariance matrix (M features, M features)
    #
    # Do not apply centering here. We assume that A is centered before this function is called.
    C = np.cov(np.transpose(A))
    return C

def compute_eigenvalue_eigenvectors(A):
    # INPUT:
    # A    [DxD] numpy matrix
    #
    # OUTPUT:
    # eigval    [D] numpy vector of eigenvalues
    # eigvec    [DxD] numpy array of eigenvectors
    eigval, eigvec = np.linalg.eig(A)
    # Numerical roundoff can lead to (tiny) imaginary parts. We correct that here.
    eigval = eigval.real
    eigvec = eigvec.real
    return eigval, eigvec

def sort_eigenvalue_eigenvectors(eigval, eigvec):
    # INPUT:
    # eigval    [D] numpy vector of eigenvalues
    # eigvec    [DxD] numpy array of eigenvectors
    #
    # OUTPUT:
    # sorted_eigval    [D] numpy vector of eigenvalues
    # sorted_eigvec    [DxD] numpy array of eigenvectors
    indices = np.argsort(eigval)
    indices = indices[::-1]
    sorted_eigval = eigval[indices]
    sorted_eigvec = eigvec[:,indices]
    return sorted_eigval, sorted_eigvec

def encode_decode_pca(A,m):
    # INPUT:
    # A    [NxM] numpy data matrix (N samples, M features)
    # m    integer number denoting the number of learned features (m <= M)
    #
    # OUTPUT:
    # Ahat [NxM] numpy PCA reconstructed data matrix (N samples, M features)
    me = np.mean(A, axis=0)
    A = center_data(A)
    C = compute_covariance_matrix(A)
    eigval, eigvec = compute_eigenvalue_eigenvectors(C)
    eigval, eigvec = sort_eigenvalue_eigenvectors(eigval, eigvec)
    if m > 0:
        eigvec = eigvec[:,:m]
    for i in range(np.shape(eigvec)[1]):
        eigvec[:,i] / np.linalg.norm(eigvec[:,i]) * np.sqrt(eigval[i])
    P = np.dot(np.transpose(eigvec), np.transpose(A))
    Ahat = np.dot(P[:,:m].T, eigvec.T)
    Ahat += me  
    return Ahat

PCA_R = encode_decode_pca(data_R, math.floor(N_R / 4))

corr_PCA = pd.DataFrame(PCA_R).corr()
corr_OR = pd.DataFrame(data_R).corr()

# plt.figure(figsize=(20, 20))
# seaborn.heatmap(corr_PCA, cmap='YlGnBu')  # , mask=(np.abs(corr_R) >= 0.5))
# plt.title('PCA (with N/2 of features) compressed and decompressed corrs')
# plt.show()

# plt.figure(figsize=(20, 20))
# seaborn.heatmap(corr_OR, cmap='YlGnBu')  # , mask=(np.abs(corr_S) >= 0.5))
# plt.title('Original corrs')
# plt.show()

diff_corrs = corr_PCA.sub(corr_OR).abs()
print('Difference between corrs: \n', diff_corrs)

##########
### ML ###
##########

###############
### COPULAS ###
###############