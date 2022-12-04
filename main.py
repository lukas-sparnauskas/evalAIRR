import yaml
import argparse

from util.input import read_encoded_csv
from util.corr import export_corr_heatmap
from util.pca import export_pca_2d_comparison
from util.univar import export_ks_test
from util.univar import export_distr_histogram, export_obs_distr_histogram
from util.univar import export_distr_boxplot, export_obs_distr_boxplot
from util.univar import export_distr_violinplot, export_obs_distr_violinplot
from util.univar import export_distr_densityplot, export_obs_distr_densityplot
from util.univar import export_avg_var_scatter_plot
from util.univar import export_distance, export_obs_distance
from util.univar import export_statistics, export_obs_statistics
from util.copula import export_copula_scatter_plot
from util.report import export_report

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
with open(YAML_FILE, 'r') as stream:
    try:
        CONFIG = (yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)

REPORTS = CONFIG['reports']

try:
    OUTPUT = CONFIG['output']['path']
except: 
    OUTPUT = './output/report.html'
    
#####################
### READ DATASETS ###
#####################

features_R, data_R = read_encoded_csv(CONFIG['datasets']['real']['path'])
features_S, data_S = read_encoded_csv(CONFIG['datasets']['sim']['path'])

###################
### CORR REPORT ###
###################

if ('corr' in REPORTS):
    percent_features = REPORTS['corr']['percent_features']
    export_corr_heatmap(data_R, data_S, len(features_R), len(features_S), percent_features)

#####################
### PCA 2D REPORT ###
#####################

if ('pca_2d' in REPORTS):
    export_pca_2d_comparison(data_R, data_S)

######################
### KS TEST REPORT ###
######################

if ('ks' in REPORTS):
    ks_reports = REPORTS['ks']
    for feature in ks_reports:
        export_ks_test(feature, data_R, data_S, features_R, features_S)

#####################################
### DISTRIBUTION HISTOGRAM REPORT ###
#####################################

if ('distr_histogram' in REPORTS):
    distr_histogram_reports = REPORTS['distr_histogram']
    for feature in distr_histogram_reports:
        export_distr_histogram(feature, data_R, data_S, features_R, features_S)

#################################################
### OBSERVATION DISTRIBUTION HISTOGRAM REPORT ###
#################################################

if ('observation_distr_histogram' in REPORTS):
    obs_distr_histogram_reports = REPORTS['observation_distr_histogram']
    for observation_index in obs_distr_histogram_reports:
        export_obs_distr_histogram(observation_index, data_R, data_S)

###################################
### DISTRIBUTION BOXPLOT REPORT ###
###################################

if ('distr_boxplot' in REPORTS):
    distr_boxplot_reports = REPORTS['distr_boxplot']
    for feature in distr_boxplot_reports:
        export_distr_boxplot(feature, data_R, data_S, features_R, features_S)


###############################################
### OBSERVATION DISTRIBUTION BOXPLOT REPORT ###
###############################################

if ('observation_distr_boxplot' in REPORTS):
    obs_distr_boxplot_reports = REPORTS['observation_distr_boxplot']
    for observation_index in obs_distr_boxplot_reports:
        export_obs_distr_boxplot(observation_index, data_R, data_S)

######################################
### DISTRIBUTION VIOLINPLOT REPORT ###
######################################

if ('distr_violinplot' in REPORTS):
    distr_violinplot_reports = REPORTS['distr_violinplot']
    for feature in distr_violinplot_reports:
        export_distr_violinplot(feature, data_R, data_S, features_R, features_S)

##################################################
### OBSERVATION DISTRIBUTION VIOLINPLOT REPORT ###
##################################################

if ('observation_distr_violinplot' in REPORTS):
    obs_distr_violinplot_reports = REPORTS['observation_distr_violinplot']
    for observation_index in obs_distr_violinplot_reports:
        export_obs_distr_violinplot(observation_index, data_R, data_S)

########################################
### DISTRIBUTION DENSITY PLOT REPORT ###
########################################

if ('distr_densityplot' in REPORTS):
    distr_densityplot_reports = REPORTS['distr_densityplot']
    for feature in distr_densityplot_reports:
        export_distr_densityplot(feature, data_R, data_S, features_R, features_S)

####################################################
### OBSERVATION DISTRIBUTION DENSITY PLOT REPORT ###
####################################################

if ('observation_distr_densityplot' in REPORTS):
    obs_distr_densityplot_reports = REPORTS['observation_distr_densityplot']
    for observation_index in obs_distr_densityplot_reports:
        export_obs_distr_densityplot(observation_index, data_R, data_S)

################################################
### FEATURE AVERAGE VALUE VS VARIANCE REPORT ###
################################################

if ('feature_average_vs_variance' in REPORTS):
    export_avg_var_scatter_plot(data_R, data_S, axis=0)

####################################################
### OBSERVATION AVERAGE VALUE VS VARIANCE REPORT ###
####################################################

if ('observation_average_vs_variance' in REPORTS):
    export_avg_var_scatter_plot(data_R, data_S, axis=1)

#######################
### DISTANCE REPORT ###
#######################

if ('distance' in REPORTS):
    distance_reports = REPORTS['distance']
    for feature in distance_reports:
        export_distance(feature, data_R, data_S, features_R, features_S)

###################################
### OBSERVATION DISTANCE REPORT ###
###################################

if ('observation_distance' in REPORTS):
    obs_distance_reports = REPORTS['observation_distance']
    for observation_index in obs_distance_reports:
        export_obs_distance(observation_index, data_R, data_S)

#########################
### STATISTICS REPORT ###
#########################

if ('statistics' in REPORTS):
    statistics_reports = REPORTS['statistics']
    for feature in statistics_reports:
        export_statistics(feature, data_R, data_S, features_R, features_S)

#####################################
### OBSERVATION STATISTICS REPORT ###
#####################################

if ('observation_statistics' in REPORTS):
    obs_statistics_reports = REPORTS['observation_statistics']
    for observation_index in obs_statistics_reports:
        export_obs_statistics(observation_index, data_R, data_S)

#####################
### COPULA REPORT ###
#####################

if ('copula' in REPORTS):
    copula_reports = REPORTS['copula']
    for copula_report in copula_reports:
        export_copula_scatter_plot(copula_reports[copula_report][0], copula_reports[copula_report][1], data_R, data_S, features_R, features_S)

##########################
### EXPORT HTML REPORT ###
##########################

export_report(OUTPUT)