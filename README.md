# evalAIRR

A tool that allows comparison of real and simulated AIRR datasets by providing different statistical indicators and dataset visualizations in one report.

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
    path: ./data/real_data.csv
  sim:
    path: ./data/sim_data.csv
reports:
  feature_based:
    report1:
      features:
        - AST
        - ASC
      report_types:
        - ks
        - distr_densityplot
        - distance
        - statistics
  observation_based:
    report1:
      observations:
        - all
      report_types:
        - distr_densityplot
  general:
    feat_mean_vs_variance:
    obs_mean_vs_variance:
    pca_2d_feat:
    pca_2d_obs:
    corr_feat_hist:
    corr_obs_hist:
output:
  path: ./output/report.html
```

This report will process the two provided datasets (real and simulated) with encoded kmer data, and create an HTML report with multiple report types. These include feature-based report types - Kolmogorov–Smirnov test (indicated by report type `ks`), a feature distribution density plot (indicated by report type `distr_densityplot`), Euclidean distance measures (indicated by report type `distance`) and descriptive statistics (mean, median, variance and standard deviation)(indicated by report type `statistics`) for the features `AST` and `ASC`. It will then export the report to the path `./output/report.html`. It will also create an observation-based report with the feature distribution density plot (indicated by report type `distr_densityplot`) for all observations (indicated by keyword `all` in the observation list). Finally, these general reports will be generated: feature mean compared with feature variance (indicated by report type `feat_mean_vs_variance`), observation mean compared with observation variance (indicated by report type `obs_mean_vs_variance`), two dimensional representation of all features using PCA (indicated by report type `pca_2d_feat`), two dimensional representation of all observations using PCA (indicated by report type `pca_2d_obs`), feature correlation coefficient distribution histogram (indicated by report type `corr_feat_hist`) and an observation correlation coefficient distribution histogram (indicated by report type `corr_obs_hist`). More details on what reports can be created can be found in the _YAML Configuration Guidelines_ section.

The repository contains sample datafiles and a quickstart YAML configuration files. You can clone the repository and run evalAIRR within it to use sample data.

Within the cloned repository run the command:

`evalairr -i ./yaml_files/quickstart.yaml`

The report will be generated in the specified output path in the configuration file or, if a specific path is not provided, in `<CURRENT_DIRECTORY>/output/report.html`. The report is exported in the HTML format.

## YAML Configuration Guidelines

The configuration YAML file consists of 3 main sections: `datasets`, `reports` and `output`.

### Datasets

In the `datasets` section, you have to provide paths to a real and a simulated datasets that you are comparing. CSV files with encoded kmer data are supported. This can be done by specifying the file path of each file in the `path` variable under the sections `real` and `sim` respectively. Here is an example of how a configured `datasets` section looks like:

```
datasets:
  real:
    path: ./data/real_data.csv
  sim:
    path: ./data/sim_data.csv
```

### Reports

In the `reports` section, you can provide the list of report types you want to create and their parameters. There are three types of report groups depending on the different parameters: `feature_based`, `observation_based` and `generic`. Here is the list of reports you can create that compare the features of the real dataset with the simulated dataset:

#### Feature-based reports

- <b>`ks`</b> - Kolmogorov–Smirnov statistic. Parameters: list of features you are creating the report for.
- <b>`distr_histogram`</b> - feature distribution histogram. Parameters: list of features you are creating the report for.
- <b>`distr_boxplot`</b> - feature distribution boxplot. Parameters: list of features you are creating the report for.
- <b>`distr_violinplot`</b> - feature distribution violin plot. Parameters: list of features you are creating the report for.
- <b>`distr_densityplot`</b> - feature distribution density plot. Parameters: list of features you are creating the report for.
- <b>`distance`</b> - Euclidean distance between the real and simulated feature. Parameters: list of features you are creating the report for.
- <b>`statistics`</b> - statistical indicators (mean, median, standard deviation and variance) of a feature in both real and simulated datasets. Parameters: list of features you are creating the report for.

#### Observation-based reports

- <b>`ks`</b> - Kolmogorov–Smirnov statistic. Parameters: list of observations you are creating the report for.
- <b>`distr_histogram`</b> - observation distribution histogram. Parameters: list of observations you are creating the report for.
- <b>`distr_boxplot`</b> - observation distribution boxplot. Parameters: list of observations you are creating the report for.
- <b>`distr_violinplot`</b> - observation distribution violin plot. Parameters: list of observations you are creating the report for.
- <b>`distr_densityplot`</b> - observation distribution density plot. Parameters: list of observations you are creating the report for. The observation index `all` can be used to report on all observations in one plot. `with_ml_sim` - optional parameter, which if True, instructs the report to include a comparison with a generated dataset using a GaussianProcessRegressor machine learning model trained on the real dataset (only applies in reports with `all` observations). `ml_random_state` - optional integer parameter, relevant only if `with_ml_sim` is set to True, which sets a seed in the machine learning model random number generation.
- <b>`distance`</b> - Euclidean distance between the real and simulated observation. Parameters: list of observations you are creating the report for.
- <b>`statistics`</b> - statistical indicators (mean, median, standard deviation and variance) of an observation in both real and simulated datasets. Parameters: list of observations you are creating the report for.

#### General reports

- <b>`ks_feat`</b> - Kolmogorov–Smirnov statistic for all features. Parameters: `output` - optional parameter, that specifies the path of text/csv file the results will be exported to (default value is set to `./output/ks_feat.csv`). The csv file contains two rows, with the first row containing the ks-statistic and the second one - the p-values.
- <b>`ks_obs`</b> - Kolmogorov–Smirnov statistic for all observations. Parameters: `output` - optional parameter, that specifies the path of text/csv file the results will be exported to (default value is set to `./output/ks_obs.csv`). The csv file contains two rows, with the first row containing the ks-statistic and the second one - the p-values.
- <b>`copula_2d`</b> - a 2D scatter plot that displays two features in a Gausian Multivariate copula space. Parameters: a report section of any name, under which the compared features are specified.
- <b>`copula_3d`</b> - a 3D scatter plot that displays three features in a Gausian Multivariate copula space. Parameters: a report section of any name, under which the compared features are specified.
- <b>`feat_mean_vs_variance`</b> - a scatter plot that displays the mean value of every feature on one axis and the variance of every feature on the other axis. Parameters: `with_ml_sim` - optional parameter, which if True, instructs the report to include a comparison with a generated dataset using a GaussianProcessRegressor machine learning model trained on the real dataset. `ml_random_state` - optional integer parameter, relevant only if `with_ml_sim` is set to True, which sets a seed in the machine learning model random number generation.
- <b>`obs_mean_vs_variance`</b> - a scatter plot that displays the mean value of every observation on one axis and the variance of every observation on the other axis. Parameters: `with_ml_sim` - optional parameter, which if True, instructs the report to include a comparison with a generated dataset using a GaussianProcessRegressor machine learning model trained on the real dataset. `ml_random_state` - optional integer parameter, relevant only if `with_ml_sim` is set to True, which sets a seed in the machine learning model random number generation.
- <b>`corr`</b> - correlation matrix heatmaps of the real and simulated datasets. Parameters: `reduce_to_n_features` - an optional parameter for dimensionality reduction using PCA. The number of features to reduce the dataset to (must be reduce_to_n_features < min(n_observations, n_features)). `with_ml_sim` - optional parameter, which if True, instructs the report to include a comparison with a generated dataset using a GaussianProcessRegressor machine learning model trained on the real dataset. `ml_random_state` - optional integer parameter, relevant only if `with_ml_sim` is set to True, which sets a seed in the machine learning model random number generation.
- <b>`corr_feat_hist`</b> - feature correlation matrix distribution histogram for the real and simulated datasets. Parameters: `n_bins` - an optional parameter that sets the number of bins in the histogram (default value is 30). `reduce_to_n_features` - an optional parameter for dimensionality reduction using PCA. The number of features to reduce the dataset to (must be reduce_to_n_features < min(n_observations, n_features)). `with_ml_sim` - optional parameter, which if True, instructs the report to include a comparison with a generated dataset using a GaussianProcessRegressor machine learning model trained on the real dataset. `ml_random_state` - optional integer parameter, relevant only if `with_ml_sim` is set to True, which sets a seed in the machine learning model random number generation.
- <b>`corr_obs_hist`</b> - observation correlation matrix distribution histogram for the real and simulated datasets. Parameters: `n_bins` - an optional parameter that sets the number of bins in the histogram (default value is 30). `reduce_to_n_obs` - an optional parameter for dimensionality reduction using PCA. The number of observations to reduce the dataset to (must be reduce_to_n_obs < min(n_observations, n_features)). `with_ml_sim` - optional parameter, which if True, instructs the report to include a comparison with a generated dataset using a GaussianProcessRegressor machine learning model trained on the real dataset. `ml_random_state` - optional integer parameter, relevant only if `with_ml_sim` is set to True, which sets a seed in the machine learning model random number generation.
- <b>`corr_csv`</b> - CSV file exporting of the difference between correlation matrices of the real and simulated datasets. Parameters: `output` - optional parameter, that specifies the path of text/csv file the results will be exported to (default value is set to `./output/corr.csv`).
- <b>`pca_2d_feat`</b> - two feature-level scatter plots with both datasets reduced to two dimensions using PCA. Parameters: `with_ml_sim` - optional parameter, which if True, instructs the report to include a comparison with a generated dataset using a GaussianProcessRegressor machine learning model trained on the real dataset. `ml_random_state` - optional integer parameter, relevant only if `with_ml_sim` is set to True, which sets a seed in the machine learning model random number generation.
- <b>`pca_2d_obs`</b> - two observation-level scatter plots with both datasets reduced to two dimensions using PCA. Parameters: `with_ml_sim` - optional parameter, which if True, instructs the report to include a comparison with a generated dataset using a GaussianProcessRegressor machine learning model trained on the real dataset. `ml_random_state` - optional integer parameter, relevant only if `with_ml_sim` is set to True, which sets a seed in the machine learning model random number generation.
- <b>`distance_feat`</b> - Euclidean distance between the real and simulated features. Parameters: `output` - optional parameter, that specifies the path of text/csv file the results will be exported to (default value is set to `./output/dist.csv`).
- <b>`statistics_feat`</b> - statistical indicators (mean, median, standard deviation and variance) of all features in both real and simulated datasets. Parameters: `output_dir` - optional parameter, that specifies the directory for the csv files in which the csv result files `real_stat.csv` and `sim_stat.csv` will be exported to (default value is set to `./output/`). Each csv file contain four rows, each with a different statistic: 1 - mean, 2 - median, 3 - standard deviation, 4 - variance.
- <b>`distance_obs`</b> - Euclidean distance between the real and simulated observations. Parameters: `output` - optional parameter, that specifies the path of text/csv file the results will be exported to (default value is set to `./output/obs_dist.csv`).
- <b>`statistics_obs`</b> - statistical indicators (mean, median, standard deviation and variance) of all observation in both real and simulated datasets. Parameters: `output_dir` - optional parameter, that specifies the directory for the csv files in which the csv result files `real_obs_stat.csv` and `sim_obs_stat.csv` will be exported to (default value is set to `./output/`). Each csv file contain four rows, each with a different statistic: 1 - mean, 2 - median, 3 - standard deviation, 4 - variance.
- <b>`jensen_shannon_feat`</b> - Jensen-Shannon divergence metric between the real and simulated features. Parameters: `output` - optional parameter, that specifies the path of text/csv file the results will be exported to (default value is set to `./output/jenshan.csv`).
- <b>`jensen_shannon_obs`</b> - Jensen-Shannon divergence metric between the real and simulated observations. Parameters: `output` - optional parameter, that specifies the path of text/csv file the results will be exported to (default value is set to `./output/obs_jenshan.csv`).

Here is a sample `reports` section of a configuration file containing all of the reports:

```
reports:
  feature_based:
    report1:
      features:
        - AST
        - ASC
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
        - 20
      report_types:
        - ks
        - distr_histogram
        - distr_boxplot
        - distr_violinplot
        - distr_densityplot
        - distance
        - statistics
    report2:
      observations:
        - all
      report_types:
        - distr_densityplot
  general:
    copula_2d:
      report1:
        - AST
        - ASC
    copula_3d:
      report1:
        - AST
        - ASC
        - ASA
    feat_mean_vs_variance:
    obs_mean_vs_variance:
    corr:
      reduce_to_n_features: 150
    corr_feat_hist:
      n_bins: 30
    corr_obs_hist:
      n_bins: 20
      with_ml_sim: True
      ml_random_state: 0
    pca_2d_feat:
    pca_2d_obs:
    corr_csv:
      output: ./output/corr.csv
    ks_feat:
      output: ./output/ks_feat.csv
    ks_obs:
      output: ./output/ks_obs.csv
    statistics_feat:
      output_dir: ./output/
    statistics_obs:
      output_dir: ./output/
    distance_feat:
      output: ./output/dist.csv
    distance_obs:
      output: ./output/obs_dist.csv
    jensen_shannon_feat:
      output: ./output/jenshan.csv
    jensen_shannon_obs:
      output: ./output/obs_jenshan.csv
```

### Output

An optional section where you can specify the file path of the generated report. The default path of the generated report is `<CURRENT_DIRECTORY>/output/report.html`. The report is exported in the HTML format. If you declare the path as 'NONE', the report will not be created.

An example output section:

```
output:
  path: ./output/report.html
```

For example, this output section would result in a report file not being created:

```
output:
  path: NONE
```

## Example of full HTML report

![full_report](https://github.com/lukas-sparnauskas/evalAIRR/assets/58486048/3c8771f9-5540-4b08-902f-44e9ef867c72)
