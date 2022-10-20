import yaml
import argparse
from util.read_csv import read_encoded_csv

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

#####################
### READ DATASETS ###
#####################

features_R, data_R = read_encoded_csv(CONFIG['datasets']['real']['path'])
features_S, data_S = read_encoded_csv(CONFIG['datasets']['sim']['path'])