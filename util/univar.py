import matplotlib.pyplot as plt
import seaborn as sns
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

def export_distr_histogram(feature, data_R, data_S, features_R, features_S, n_bins=30):
    data_R_f = get_feature_data(feature, data_R, features_R)
    data_S_f = get_feature_data(feature, data_S, features_S)
    bins = np.linspace(min(min(data_R_f), min(data_S_f)), max(max(data_R_f), max(data_S_f)), n_bins)

    f, ax = plt.subplots(1, 1)
    f.set_size_inches(9, 7)
    f.suptitle(f'Distribution histograms of feature {feature}')
    ax.hist([data_R_f, data_S_f], bins, label=['Real dataset', 'Simulated dataset'])
    ax.legend(loc='upper right')
    plt.show()

def export_distr_boxplot(feature, data_R, data_S, features_R, features_S):
    data_R_f = get_feature_data(feature, data_R, features_R)
    data_S_f = get_feature_data(feature, data_S, features_S)

    f, ax = plt.subplots(1, 1)
    f.set_size_inches(9, 7)
    f.suptitle(f'Distribution boxplots of feature {feature}')
    ax.boxplot([data_R_f, data_S_f], labels=['Real dataset', 'Simulated dataset'])
    plt.show()

def export_distr_violinplot(feature, data_R, data_S, features_R, features_S):
    data_R_f = get_feature_data(feature, data_R, features_R)
    data_S_f = get_feature_data(feature, data_S, features_S)
    
    f, [ax1, ax2] = plt.subplots(2, 1)
    f.set_size_inches(9, 7)
    f.suptitle(f'Distribution violin plots of feature {feature}')
    ax1.violinplot(data_R_f, vert=False, widths=0.7,
                     showmeans=True, showextrema=True, showmedians=True)
    ax1.set_title('Real dataset')
    ax1.set_yticklabels([])
    ax1.set_yticks([])
    ax1.set
    ax2.violinplot(data_S_f, vert=False, widths=0.7,
                     showmeans=True, showextrema=True, showmedians=True)
    ax2.set_title('Simulated dataset')
    ax2.set_yticklabels([])
    ax2.set_yticks([])

    xbound = (min(ax1.get_xbound()[0], ax2.get_xbound()[0]), max(ax1.get_xbound()[1], ax2.get_xbound()[1]))
    ybound = (min(ax1.get_ybound()[0], ax2.get_ybound()[0]), max(ax1.get_ybound()[1], ax2.get_ybound()[1]))

    ax1.set_xbound(xbound)
    ax1.set_ybound(ybound)
    ax2.set_xbound(xbound)
    ax2.set_ybound(ybound)

    plt.show()

def export_distr_densityplot(feature, data_R, data_S, features_R, features_S):
    data_R_f = get_feature_data(feature, data_R, features_R)
    data_S_f = get_feature_data(feature, data_S, features_S)

    f, ax = plt.subplots(1, 1)
    f.set_size_inches(9, 7)
    f.suptitle(f'Distribution density plot of feature {feature}')
    sns.kdeplot(data_R_f, ax=ax, label='Real dataset', fill=True, common_norm=False, color='#5480d1', alpha=0.5, linewidth=0)
    sns.kdeplot(data_S_f, ax=ax, label='Simulated dataset', fill=True, common_norm=False, color='#d65161', alpha=0.5, linewidth=0)
    ax.legend()
    plt.show()