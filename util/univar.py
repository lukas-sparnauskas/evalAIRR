import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

def cdf(data):
    data_sorted = np.sort(data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    # ax2 = fig.add_subplot(122)
    # ax2.plot(data_sorted, p)
    # ax2.set_xlabel('$x$')
    # ax2.set_ylabel('$p$')
    # return x, y
    return data_sorted, p

def get_feature_data(feature, data, features):
    idx = np.where(features == feature)[0][0]
    if not idx:
        print(f'[ERROR] Feature {feature} not found!')
        return np.array([])
    return data[:, idx].flatten()

def export_ks_test(feature, data_R, data_S, features_R, features_S):
    data_R_f = get_feature_data(feature, data_R, features_R)
    data_S_f = get_feature_data(feature, data_S, features_S)
    cdf_1_x, cdf_1_y = cdf(data_R_f)
    cdf_2_x, cdf_2_y = cdf(data_S_f)
    res = scipy.stats.ks_2samp(data_R_f, data_S_f)
    print(f'[RESULT] Feature {feature} KS statistic =', res.statistic)
    print(f'[RESULT] Feature {feature} P value =', res.pvalue)

    f,(ax1, ax2) = plt.subplots(1, 2)
    f.set_size_inches(10, 5)
    f.suptitle(f'CDF comparison of feature {feature} in  real and simulated datasets')
    ax1.plot(cdf_1_x, cdf_1_y, c='#1b24a8')
    ax1.set_title(f'CDF of feature {feature} in the real dataset')
    ax2.plot(cdf_2_x, cdf_2_y, c='#781010')
    ax2.set_title(f'CDF of feature {feature} in the simulated dataset')
    plt.show()