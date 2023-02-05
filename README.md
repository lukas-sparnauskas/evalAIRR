# evalAIRR

A tool that allows comparison of real and simulated datasets by providing different statistical indicators and dataset visualizations in one report.

## Installation

It is recommended to use a virtual python environment to run evalAIRR if another python environment is used. Here is a quick guide on how you can set up a virtual environment:

`https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments`

### Install using pip

Run this command to install the evalAIRR package:

`pip install evalairr`

## Quickstart

evalAIRR uses a YAML file for configuration. If you are unfamiliar with how YAML files are structured, read this guide to the syntax:

`https://docs.fileformat.com/programming/yaml/#syntax`

This is the stucture of a sample report configuration file you can use to start off with (it is included in the repository location ./yaml_files/quickstart.yaml):

```
datasets:
  real:
    path: ./data/encoded_real_1000_200.csv
  sim:
    path: ./data/encoded_sim_1000_200.csv
reports:
  feature_based:
    report1:
      features:
        - TGT
        - ANV
      report_types:
        - ks
        - distr_densityplot
output:
  path: './output/report.html'

```

This report will process the two provided datasets (real and simulated), and create an HTML report with feature-based report types - Kolmogorov–Smirnov test (indicated by `ks`) and a feature distribution density plot (indicated by `distr_densityplot`) for the features `TGT` and `ANV`. It will then export the report to the path `./output/report.html`. More details on what reports can be created can be found in the *YAML Configuration Guidelines* secion.

You can run the program by running this command within the installation directory:

`evalairr -i ./yaml_files/quickstart.yaml`

The report will be generated in the specified output path in the configuration file or, if a specific path is not provided, in `<CURRENT_DIRECTORY>/output/report.html`. The report is exported in the HTML format.

## YAML Configuration Guidelines

The configuration YAML file consists of 3 main sections: `datasets`, `reports` and `output`.

### Datasets

In the `datasets` section, you have to provide paths to a real and a simulated datasets that you are comparing. This can be done by specifying the file path of each file in the `path` variable under the sections `real` and `sim` respectively. Here is an example of how a configured `datasets` section looks like:

```
datasets:
  real:
    path: ./data/encoded_real_1000_200.csv
  sim:
    path: ./data/encoded_sim_1000_200.csv
```

### Reports

In the `reports` section, you can provide the list of report types you want to create and their parameters. There are three types of report groups depending on the different parameters: `feature_based`, `observation_based` and `generic`. Here is the list of reports you can create that compare the features of the real dataset with the simulated dataset:

#### Feature-based reports
- `ks` - Kolmogorov–Smirnov statistic. Parameters: list of features you are creating the report for.
- `distr_histogram` - feature distribution histogram. Parameters: list of features you are creating the report for.
- `distr_boxplot` - feature distribution boxplot. Parameters: list of features you are creating the report for.
- `distr_violinplot` - feature distribution violin plot. Parameters: list of features you are creating the report for.
- `distr_densityplot` - feature distribution density plot. Parameters: list of features you are creating the report for.
- `distance` - Euclidean distance between the real and simulated feature. Parameters: list of features you are creating the report for.
- `statistics` - statistical indicators (average, median, standard deviation and variance) of a feature in both real and simulated datasets. Parameters: list of features you are creating the report for.

#### Observation-based reports
- `observation_distr_histogram` - observation distribution histogram. Parameters: list of observations you are creating the report for.
- `observation_distr_boxplot` - observation distribution boxplot. Parameters: list of observations you are creating the report for.
- `observation_distr_violinplot` - observation distribution violin plot. Parameters: list of observations you are creating the report for.
- `observation_distr_densityplot` - observation distribution density plot. Parameters: list of observations you are creating the report for. The observation index 'all' can be used to report on all observations in one plot.
- `observation_distance` - Euclidean distance between the real and simulated observation. Parameters: list of observations you are creating the report for.
- `observation_statistics` - statistical indicators (average, median, standard deviation and variance) of an observation in both real and simulated datasets. Parameters: list of observations you are creating the report for.

#### General reports
- `copula_2d` - a 2D scatter plot that displays two features in a Gausian Multivariate copula space. Parameters: a report section of any name, under which the compared features are specified.
- `copula_3d` - a 3D scatter plot that displays three features in a Gausian Multivariate copula space. Parameters: a report section of any name, under which the compared features are specified.
- `feature_average_vs_variance` - a scatter plot that displays the average value of every feature on one axis and the variance of every feature on the other axis.
- `observation_average_vs_variance` - a scatter plot that displays the average value of every observation on one axis and the variance of every observation on the other axis.
- `corr` - correlation matrix heatmaps of the real and simulated datasets. Parameters: `percent_features` - an optional parameter for dimensionality reduction using PCA. A float value corresponding with the ratio of feature reduction (e.g. `percent_features` equal to 0.5 would reduce the feature count by half). 
- `pca_2d` - two scatter plots with both datasets reduced to two dimensions using PCA.

Here is a sample `reports` section of a configuration file containing all of the reports:

```
reports:
  feature_based:
    report1:
      features:
        - TGT
        - ANV
      report_types:
        - ks
        - distr_histogram
        - distr_boxplot
        - distr_violinplot
        - distr_densityplot
        - distance
        - statistics
  observation_based:
    report1:
      observations:
        - 0
      report_types:
        - observation_distr_histogram
        - observation_distr_boxplot
        - observation_distr_violinplot
        - observation_distr_densityplot
        - observation_distance
        - observation_statistics
    report2:
      observations:
        - all
      report_types:
        - observation_distr_densityplot
  general:
    copula_2d:
      report1:
        - TGT
        - ANV
    copula_3d:
      report1:
        - TGT
        - ANV
        - CAS
    feature_average_vs_variance:
    observation_average_vs_variance:
    corr:
      percent_features: 0.6
    pca_2d:
```

### Output

An optional section where you can specify the file path of the generated report. The default path of the generated report is `<INSTALL_DIRECTORY>/output/report.html`. The report is exported in the HTML format.

An example output section:

```
output:
  path: './output/report.html'
```