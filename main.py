import yaml
import argparse
import numpy as np
from util.input import read_encoded_csv
from util.corr import export_corr_heatmap
from util.pca import export_pca_comparison
from util.univar import export_ks_test

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

do_corr_report = 'multivariate' in REPORTS
if (do_corr_report):
    export_corr_heatmap(data_R, data_S, len(features_R), len(features_S), 0.33)

##################
### PCA REPORT ###
##################

do_pca_report = 'pca' in REPORTS
if (do_pca_report):
    export_pca_comparison(data_R, data_S)

######################
### KS TEST REPORT ###
######################

do_ks_report = 'ks' in REPORTS
if (do_ks_report):
    ks_reports = REPORTS['ks']
    for feature in ks_reports:
        export_ks_test(feature, data_R, data_S, features_R, features_S)