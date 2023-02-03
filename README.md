# simAIRR_evaluation

A tool that allows comparison of real and simulated datasets by providing different statistical indicators and dataset visualizations in one report.

## Installation

Clone the repository and install the required dependencies using this command:

`pip install -r <INSTALL_DIRECTORY>/requirements.txt`

## Quickstart

simAIRR_evaluation uses a YAML file for configuration. If you are unfamiliar with how YAML files are structured, read this guide to the syntax:

`https://docs.fileformat.com/programming/yaml/#syntax`

This is the stucture of a sample report configuration file you can use to start off with:

```
datasets:
  real:
    path: ./data/encoded_real_1000_200.csv
  sim:
    path: ./data/encoded_sim_1000_200.csv
reports:
  ks:
    - TGT
    - ANV
  distr_densityplot:
    - TGT
    - ANV
output:
  path: './output/report.html'
```

This report will process the two provided datasets (real and simulated), and create two reports - Kolmogorov–Smirnov test (indicated by `ks`) and a feature distribution density plot (indicated by `distr_densityplot`) for the features `TGT` and `ANV`. It will then export the report to the path `./output/report.html`. More details on what reports can be created can be found in the *YAML Configuration Guidelines* secion.

You can run the program by running this command within the installation directory:

`python main.py ./yaml_files/quickstart.yaml`

The report will be generated in the specified output path in the configuration file or, if a specific path is not provided, in `<INSTALL_DIRECTORY>/output/report.html`. The report is exported in the HTML format.

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

In the `reports` section, you can provide the list of report types you want to create and their parameters. Here is the list of reports you can create that compare the features of the real dataset with the simulated dataset:

- `ks` - Kolmogorov–Smirnov statistic. Parameters: list of features you are creating the report for.
- `distr_histogram` - feature distribution histogram. Parameters: list of features you are creating the report for.
- `observation_distr_histogram` - observation distribution histogram. Parameters: list of observations you are creating the report for.
- `distr_boxplot` - feature distribution boxplot. Parameters: list of features you are creating the report for.
- `observation_distr_boxplot` - observation distribution boxplot. Parameters: list of observations you are creating the report for.
- `distr_violinplot` - feature distribution violin plot. Parameters: list of features you are creating the report for.
- `observation_distr_violinplot` - observation distribution violin plot. Parameters: list of observations you are creating the report for.
- `distr_densityplot` - feature distribution density plot. Parameters: list of features you are creating the report for.
- `observation_distr_densityplot` - observation distribution density plot. Parameters: list of observations you are creating the report for.
- `distance` - Euclidean distance between the real and simulated feature. Parameters: list of features you are creating the report for.
- `observation_distance` - Euclidean distance between the real and simulated observation. Parameters: list of observations you are creating the report for.
- `statistics` - statistical indicators (average, median, standard deviation and variance) of a feature in both real and simulated datasets. Parameters: list of features you are creating the report for.
- `observation_statistics` - statistical indicators (average, median, standard deviation and variance) of an observation in both real and simulated datasets. Parameters: list of observations you are creating the report for.
- `copula_2d` - a 2D scatter plot that displays two features in a Gausian Multivariate copula space. Parameters: a report section of any name, under which the compared features are specified.
- `copula_3d` - a 3D scatter plot that displays three features in a Gausian Multivariate copula space. Parameters: a report section of any name, under which the compared features are specified.
- `feature_average_vs_variance` - a scatter plot that displays the average value of every feature on one axis and the variance of every feature on the other axis.
- `observation_average_vs_variance` - a scatter plot that displays the average value of every observation on one axis and the variance of every observation on the other axis.
- `corr` - correlation matrix heatmaps of the real and simulated datasets. Parameters: `percent_features` - an optional parameter for dimensionality reduction using PCA. A float value corresponding with the ratio of feature reduction (e.g. `percent_features` equal to 0.5 would reduce the feature count by half). 
- `pca_2d` - two scatter plots with both datasets reduced to two dimensions using PCA.

Here is a sample `reports` section of a configuration file with all of the reports:

```
reports:
  ks:
    - TGT
  distr_histogram:
    - TGT
  observation_distr_histogram:
    - 0
    - 199
  distr_boxplot:
    - TGT
  observation_distr_boxplot:
    - 0
    - 199
  distr_violinplot:
    - TGT
  observation_distr_violinplot:
    - 0
    - 199
  distr_densityplot:
    - TGT
  observation_distr_densityplot:
    - 0
    - 199
  distance:
    - TGT
  observation_distance:
    - 0
    - 199
  statistics:
    - TGT
  observation_statistics:
    - 0
    - 199
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
