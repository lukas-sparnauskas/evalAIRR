import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from util.pca import pca

def export_corr_heatmap(data_real, data_sim, n_real_feat = 0, n_sim_feat = 0, pca_ratio = 0):

    if n_real_feat != 0 and n_sim_feat != 0 and pca_ratio != 0:
        _, data_real = pca(data_real, math.floor(n_real_feat * pca_ratio))
        _, data_sim = pca(data_sim, math.floor(n_sim_feat * pca_ratio))

    corr_real = pd.DataFrame(data_real).corr()
    corr_sim = pd.DataFrame(data_sim).corr()

    diff_corrs = corr_real.sub(corr_sim).abs()

    f,(ax1,ax2,ax3, axcb) = plt.subplots(1,4,gridspec_kw={'width_ratios':[1,1,1,0.08]})
    f.set_size_inches(15, 5)
    f.suptitle('Correlation heatmaps')
    ax1.get_shared_y_axes().join(ax2,ax3)
    g1 = sns.heatmap(corr_real,cmap="YlGnBu",cbar=False,ax=ax1)
    g1.set_title('Real dataset')
    g1.set_ylabel('')
    g1.set_xlabel('')
    g2 = sns.heatmap(corr_sim,cmap="YlGnBu",cbar=False,ax=ax2)
    g2.set_title('Simulated dataset')
    g2.set_ylabel('')
    g2.set_xlabel('')
    g2.set_yticks([])
    g3 = sns.heatmap(diff_corrs,cmap="YlGnBu",ax=ax3, cbar_ax=axcb)
    g3.set_title('Difference in correlation')
    g3.set_ylabel('')
    g3.set_xlabel('')
    g3.set_yticks([])
    
    plt.show()