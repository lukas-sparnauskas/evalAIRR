import numpy as np
import matplotlib.pyplot as plt

from util.univar import cdf
from util.univar import get_feature_data
from util.univar import get_observation_data

def export_copula_scatter_plot(feature_1, feature_2, data_R, data_S, features_R, features_S):
    data_R_f1 = get_feature_data(feature_1, data_R, features_R)
    data_R_f2 = get_feature_data(feature_2, data_R, features_R)
    if not any(data_R_f1) or not any(data_R_f2):
        return

    data_S_f1 = get_feature_data(feature_1, data_S, features_S)
    data_S_f2 = get_feature_data(feature_2, data_S, features_S)
    if not any(data_R_f2) or not any(data_S_f2):
        return

    cdf_R_f1_x, cdf_R_f1_y = cdf(data_R_f1)
    cdf_R_f2_x, cdf_R_f2_y = cdf(data_R_f2)

    cdf_S_f1_x, cdf_S_f1_y = cdf(data_S_f1)
    cdf_S_f2_x, cdf_S_f2_y = cdf(data_S_f2)

    cdf_R_f1 = np.concatenate((cdf_R_f1_x, cdf_R_f1_y))
    cdf_R_f2 = np.concatenate((cdf_R_f2_x, cdf_R_f2_y))

    cdf_S_f1 = np.concatenate((cdf_S_f1_x, cdf_S_f1_y))
    cdf_S_f2 = np.concatenate((cdf_S_f2_x, cdf_S_f2_y))

    f, [ax1, ax2] = plt.subplots(2, 1)
    f.set_size_inches(9, 7)
    f.suptitle(f'Distribution of {feature_1} and {feature_2} in copula space')

    ax1.set_title(f'Real dataset')
    ax1.scatter(cdf_R_f1, cdf_R_f2, c='#5480d1', linewidths=None, alpha=0.5)
    ax1.set_xlabel(f'{feature_1}')
    ax1.set_ylabel(f'{feature_2}')

    ax1.set_title(f'Simulated dataset')
    ax2.scatter(cdf_S_f1, cdf_S_f2, c='#d65161', linewidths=None, alpha=0.5)
    ax2.set_xlabel(f'{feature_1}')
    ax2.set_ylabel(f'{feature_2}')

    f.savefig(f'./output/temp_figures/copula_plot_{feature_1}_{feature_2}.svg')
    del f