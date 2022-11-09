import yaml
import argparse
import numpy as np
from util.input import read_encoded_csv
from util.corr import export_corr_heatmap
from util.pca import export_pca_2d_comparison
from util.univar import export_ks_test
from util.univar import export_distr_histogram
from util.univar import export_distr_boxplot
from util.univar import export_distr_violinplot

#######################
### PARSE ARGUMENTS ###
#######################

parser = argparse.ArgumentParser(prog='simAIRR_eval')
parser.add_argument('yaml_file', type=str, help='YAML file path')
YAML_FILE = parser.parse_args().yaml_file


#################
### READ YAML ###
#################

CONFIG = []
with open(YAML_FILE, "r") as stream:
    try:
        CONFIG = (yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)

REPORTS = CONFIG['reports']

#####################
### READ DATASETS ###
#####################

features_R, data_R = read_encoded_csv(CONFIG['datasets']['real']['path'])
features_S, data_S = read_encoded_csv(CONFIG['datasets']['sim']['path'])

###################
### CORR REPORT ###
###################

do_corr_report = 'corr' in REPORTS
if (do_corr_report):
    percent_features = REPORTS['corr']['percent_features']
    export_corr_heatmap(data_R, data_S, len(features_R), len(features_S), percent_features)

#####################
### PCA 2D REPORT ###
#####################

do_pca_2d_report = 'pca_2d' in REPORTS
if (do_pca_2d_report):
    export_pca_2d_comparison(data_R, data_S)

######################
### KS TEST REPORT ###
######################

do_ks_report = 'ks' in REPORTS
if (do_ks_report):
    ks_reports = REPORTS['ks']
    for feature in ks_reports:
        export_ks_test(feature, data_R, data_S, features_R, features_S)

#####################################
### DISTRIBUTION HISTOGRAM REPORT ###
#####################################

do_distr_histogram_report = 'distr_histogram' in REPORTS
if (do_distr_histogram_report):
    distr_histogram_reports = REPORTS['distr_histogram']
    for feature in distr_histogram_reports:
        export_distr_histogram(feature, data_R, data_S, features_R, features_S)

###################################
### DISTRIBUTION BOXPLOT REPORT ###
###################################

do_distr_boxplot_report = 'distr_boxplot' in REPORTS
if (do_distr_boxplot_report):
    distr_boxplot_reports = REPORTS['distr_boxplot']
    for feature in distr_boxplot_reports:
        export_distr_boxplot(feature, data_R, data_S, features_R, features_S)

######################################
### DISTRIBUTION VIOLINPLOT REPORT ###
######################################

do_distr_violinplot_report = 'distr_violinplot' in REPORTS
if (do_distr_violinplot_report):
    distr_violinplot_reports = REPORTS['distr_violinplot']
    for feature in distr_violinplot_reports:
        export_distr_violinplot(feature, data_R, data_S, features_R, features_S)