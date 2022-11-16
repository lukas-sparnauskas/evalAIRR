import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats

def cdf(data):
    data_sorted = np.sort(data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
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
    
    f,(ax1, ax2) = plt.subplots(1, 2)
    f.set_size_inches(10, 5)
    f.suptitle(f'CDF comparison of feature {feature} in  real and simulated datasets')
    ax1.plot(cdf_1_x, cdf_1_y, c='#1b24a8')
    ax1.set_title(f'CDF of feature {feature} in the real dataset')
    ax2.plot(cdf_2_x, cdf_2_y, c='#781010')
    ax2.set_title(f'CDF of feature {feature} in the simulated dataset')
    
    print(f'[RESULT] Feature {feature} KS statistic =', res.statistic)
    print(f'[RESULT] Feature {feature} P value =', res.pvalue)
    with open(f'./output/temp_results/ks_test_{feature}.txt', 'w', encoding="utf-8") as file:
        file.write('\t\t<tr colspan="2">\n')
        file.write(f'\t\t\t<td>Feature {feature} KS statistic</td>\n')
        file.write(f'\t\t\t<td>{res.statistic}</td>\n')
        file.write('\t\t</tr>\n')

        file.write('\t\t<tr colspan="2">\n')
        file.write(f'\t\t\t<td>Feature {feature} P value</td>\n')
        file.write(f'\t\t\t<td>{res.pvalue}</td>\n')
        file.write('\t\t</tr>\n')

    f.savefig(f'./output/temp_figures/ks_test_{feature}.svg')
    del f

def export_distr_histogram(feature, data_R, data_S, features_R, features_S, n_bins=30):
    data_R_f = get_feature_data(feature, data_R, features_R)
    data_S_f = get_feature_data(feature, data_S, features_S)

    bins = np.linspace(min(min(data_R_f), min(data_S_f)), max(max(data_R_f), max(data_S_f)), n_bins)

    f, ax = plt.subplots(1, 1)
    f.set_size_inches(9, 7)
    f.suptitle(f'Distribution histograms of feature {feature}')
    ax.hist([data_R_f, data_S_f], bins, label=['Real dataset', 'Simulated dataset'])
    ax.legend(loc='upper right')
    
    f.savefig(f'./output/temp_figures/histogram_{feature}.svg')
    del f

def export_distr_boxplot(feature, data_R, data_S, features_R, features_S):
    data_R_f = get_feature_data(feature, data_R, features_R)
    data_S_f = get_feature_data(feature, data_S, features_S)

    f, ax = plt.subplots(1, 1)
    f.set_size_inches(9, 7)
    f.suptitle(f'Distribution boxplots of feature {feature}')
    ax.boxplot([data_R_f, data_S_f], labels=['Real dataset', 'Simulated dataset'])
    
    f.savefig(f'./output/temp_figures/box_plot_{feature}.svg')
    del f

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

    f.savefig(f'./output/temp_figures/violin_plot_{feature}.svg')
    del f

def export_distr_densityplot(feature, data_R, data_S, features_R, features_S):
    data_R_f = get_feature_data(feature, data_R, features_R)
    data_S_f = get_feature_data(feature, data_S, features_S)

    f, ax = plt.subplots(1, 1)
    f.set_size_inches(9, 7)
    f.suptitle(f'Distribution density plot of feature {feature}')
    sns.kdeplot(data_R_f, ax=ax, label='Real dataset', fill=True, common_norm=False, color='#5480d1', alpha=0.5, linewidth=0)
    sns.kdeplot(data_S_f, ax=ax, label='Simulated dataset', fill=True, common_norm=False, color='#d65161', alpha=0.5, linewidth=0)
    ax.legend()

    f.savefig(f'./output/temp_figures/density_plot_{feature}.svg')
    del f

def export_avg_var_scatter_plot(data_R, data_S, axis=0):
    data_R_x = np.average(data_R, axis=axis)
    data_R_y = np.var(data_R, axis=axis)
    data_S_x = np.average(data_S, axis=axis)
    data_S_y = np.var(data_S, axis=axis)

    f, ax = plt.subplots(1, 1)
    f.set_size_inches(9, 7)
    f.suptitle('Feature average value vs variance' if axis == 0 else 'Observation average value vs variance')
    ax.scatter(data_R_x, data_R_y, c='#5480d1', linewidths=None, alpha=0.5)
    ax.scatter(data_S_x, data_S_y, c='#d65161', linewidths=None, alpha=0.5)
    ax.set_xlabel('Average value')
    ax.set_ylabel('Variance value')
    
    f.savefig(f'./output/temp_figures/avg_var_{"feat" if axis == 0 else "obs"}_scatter_plot.svg')
    del f

def export_distance(feature, data_R, data_S, features_R, features_S):
    data_R_f = get_feature_data(feature, data_R, features_R)
    data_S_f = get_feature_data(feature, data_S, features_S)

    dist = np.linalg.norm(data_R_f - data_S_f)
    print(f'[RESULT] Euclidean distance of feature {feature} : {dist}')

    with open(f'./output/temp_results/distance_{feature}.txt', 'w', encoding="utf-8") as file:
        file.write('\t\t<tr colspan="2">\n')
        file.write(f'\t\t\t<td>Euclidean distance of feature {feature}</td>\n')
        file.write(f'\t\t\t<td>{dist}</td>\n')
        file.write('\t\t</tr>\n')
    
def export_statistics(feature, data_R, data_S, features_R, features_S):
    data_R_f = get_feature_data(feature, data_R, features_R)
    data_S_f = get_feature_data(feature, data_S, features_S)

    avg = { 'real': np.average(data_R_f), 'sim': np.average(data_S_f)}
    median = { 'real': np.median(data_R_f), 'sim': np.median(data_S_f)}
    std = { 'real': np.std(data_R_f), 'sim': np.std(data_S_f)}
    var = { 'real': np.var(data_R_f), 'sim': np.var(data_S_f)}

    print('[RESULT] Average of feature {0:>16} : REAL = {1:>25}, SIMULATED = {2:>25}'.format(feature, avg['real'], avg['sim']))
    print('[RESULT] Median of feature {0:>17} : REAL = {1:>25}, SIMULATED = {2:>25}'.format(feature, median['real'], median['sim']))
    print('[RESULT] Standard deviation of feature {0:>5} : REAL = {1:>25}, SIMULATED = {2:>25}'.format(feature, std['real'], std['sim']))
    print('[RESULT] Variance of feature {0:>15} : REAL = {1:>25}, SIMULATED = {2:>25}'.format(feature, var['real'], var['sim']))

    with open(f'./output/temp_statistics/{feature}.txt', 'w', encoding="utf-8") as file:
        file.write('\t\t<tr>\n')
        file.write(f'\t\t\t<td>Average of feature {feature}</td>\n')
        file.write(f'\t\t\t<td>{avg["real"]}</td>\n')
        file.write(f'\t\t\t<td>{avg["sim"]}</td>\n')
        file.write('\t\t</tr>\n')
        file.write('\t\t<tr>\n')
        file.write(f'\t\t\t<td>Median of feature {feature}</td>\n')
        file.write(f'\t\t\t<td>{median["real"]}</td>\n')
        file.write(f'\t\t\t<td>{median["sim"]}</td>\n')
        file.write('\t\t</tr>\n')
        file.write('\t\t<tr>\n')
        file.write(f'\t\t\t<td>Standard deviation of feature {feature}</td>\n')
        file.write(f'\t\t\t<td>{std["real"]}</td>\n')
        file.write(f'\t\t\t<td>{std["sim"]}</td>\n')
        file.write('\t\t</tr>\n')
        file.write('\t\t<tr>\n')
        file.write(f'\t\t\t<td>Variance of feature {feature}</td>\n')
        file.write(f'\t\t\t<td>{var["real"]}</td>\n')
        file.write(f'\t\t\t<td>{var["sim"]}</td>\n')
        file.write('\t\t</tr>\n')
        