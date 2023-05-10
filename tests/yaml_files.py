def immuneml_spec(run, timestamp):
  return f'''definitions:
  datasets:
    sim_dataset:
      format: OLGA
      params:
        number_of_processes: 6
        path: \"/data/sim_data/temp/\"
        is_repertoire: True
        metadata_file: \"/data/sim_data/metadata_{run}_{timestamp}.csv\"
    real_dataset:
      format: AIRR
      params:
        number_of_processes: 6
        path: \"/data/real_data/temp/\"
        is_repertoire: True
        metadata_file: \"/data/real_data/metadata_{timestamp}.csv\"
  encodings:
    continuous_kmer:
      KmerFrequency:
        reads: ALL
        sequence_encoding: CONTINUOUS_KMER
        k: 3
        scale_to_unit_variance: True
        scale_to_zero_mean: True
  reports:
    dme_report:
      DesignMatrixExporter:
        file_format: csv
instructions:
  expl_analysis_instruction:
    type: ExploratoryAnalysis
    analyses:
      real:
        dataset: real_dataset
        encoding: continuous_kmer
        report: dme_report
      sim:
        dataset: sim_dataset
        encoding: continuous_kmer
        report: dme_report
output:
  format: HTML
  '''

def evalairr_spec(run, timestamp):
  return f'''datasets:
  real:
    path: \"/home/mint/masters/data/immunemldata/output_{timestamp}/expl_analysis_instruction/analysis_real/report/design_matrix.csv\"
  sim:
    path: \"/home/mint/masters/data/immunemldata/output_{timestamp}/expl_analysis_instruction/analysis_sim/report/design_matrix.csv\"
reports:
  general:
    ks_feat:
      output: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/ks_feat.csv\"
    ks_obs:
      output: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/ks_obs.csv\"
    statistics_feat:
      output_dir: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/\"
    statistics_obs:
      output_dir: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/\"
    distance_feat:
      output: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/dist.csv\"
    distance_obs:
      output: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/dist_obs.csv\"
    jensen_shannon_feat:
      output: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/jenshan.csv\"
    jensen_shannon_obs:
      output: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/jenshan_obs.csv\"
    corr_feat_hist:
output:
  path: /home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/report.html
  '''
  
def evalairr_preencoded_spec(data_folder, run, timestamp):
  return f'''datasets:
  real:
    path: \"/home/mint/masters/data/immunemldata/{data_folder}/expl_analysis_instruction/analysis_real/report/design_matrix.csv\"
  sim:
    path: \"/home/mint/masters/data/immunemldata/{data_folder}/expl_analysis_instruction/analysis_sim/report/design_matrix.csv\"
reports:
  general:
    ks_feat:
      output: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/ks_feat.csv\"
    ks_obs:
      output: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/ks_obs.csv\"
    statistics_feat:
      output_dir: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/\"
    statistics_obs:
      output_dir: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/\"
    distance_feat:
      output: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/dist.csv\"
    distance_obs:
      output: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/dist_obs.csv\"
    jensen_shannon_feat:
      output: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/jenshan.csv\"
    jensen_shannon_obs:
      output: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/jenshan_obs.csv\"
    corr_feat_hist:
output:
  path: /home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/report.html
  '''

def threshold_test_immuneml_spec(run, timestamp):
  return f'''definitions:
  datasets:
    data1:
      format: AIRR
      params:
        number_of_processes: 6
        path: \"/data/th_data_1/temp/\"
        is_repertoire: True
        metadata_file: \"/data/th_data_1/metadata_{timestamp}.csv\"
    data2:
      format: AIRR
      params:
        number_of_processes: 6
        path: \"/data/th_data_2/temp/\"
        is_repertoire: True
        metadata_file: \"/data/th_data_2/metadata_{timestamp}.csv\"
  encodings:
    continuous_kmer:
      KmerFrequency:
        reads: ALL
        sequence_encoding: CONTINUOUS_KMER
        k: 3
        scale_to_unit_variance: True
        scale_to_zero_mean: True
  reports:
    dme_report:
      DesignMatrixExporter:
        file_format: csv
instructions:
  expl_analysis_instruction:
    type: ExploratoryAnalysis
    analyses:
      data1:
        dataset: data1
        encoding: continuous_kmer
        report: dme_report
      data2:
        dataset: data2
        encoding: continuous_kmer
        report: dme_report
output:
  format: HTML
  '''

def threshold_test_evalairr_spec(run, timestamp):
  return f'''datasets:
  real:
    path: \"/home/mint/masters/data/immunemldata/th_output_{timestamp}/expl_analysis_instruction/analysis_data1/report/design_matrix.csv\"
  sim:
    path: \"/home/mint/masters/data/immunemldata/th_output_{timestamp}/expl_analysis_instruction/analysis_data2/report/design_matrix.csv\"
reports:
  general:
    ks_feat:
      output: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/ks_feat.csv\"
    ks_obs:
      output: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/ks_obs.csv\"
    statistics_feat:
      output_dir: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/\"
    statistics_obs:
      output_dir: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/\"
    distance_feat:
      output: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/dist.csv\"
    distance_obs:
      output: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/dist_obs.csv\"
    jensen_shannon_feat:
      output: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/jenshan.csv\"
    jensen_shannon_obs:
      output: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/jenshan_obs.csv\"
    corr_feat_hist:
output:
  path: /home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/report.html
  '''
  
def threshold_test_preencoded_evalairr_spec(data_folder, run, timestamp):
  return f'''datasets:
  real:
    path: \"/home/mint/masters/data/immunemldata/{data_folder}/expl_analysis_instruction/analysis_data1/report/design_matrix.csv\"
  sim:
    path: \"/home/mint/masters/data/immunemldata/{data_folder}/expl_analysis_instruction/analysis_data2/report/design_matrix.csv\"
reports:
  general:
    ks_feat:
      output: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/ks_feat.csv\"
    ks_obs:
      output: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/ks_obs.csv\"
    statistics_feat:
      output_dir: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/\"
    statistics_obs:
      output_dir: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/\"
    distance_feat:
      output: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/dist.csv\"
    distance_obs:
      output: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/dist_obs.csv\"
    jensen_shannon_feat:
      output: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/jenshan.csv\"
    jensen_shannon_obs:
      output: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/jenshan_obs.csv\"
    corr_feat_hist:
output:
  path: /home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/report.html
  '''
  
def noisetest_evalairr_spec(run):
  return f'''datasets:
  real:
    path: /home/mint/masters/data/noise_data/original_unmodified.csv
  sim:
    path: /home/mint/masters/data/noise_data/with_noise.csv
reports:
  general:
    corr_feat_hist:
    corr_obs_hist:
      n_bins: 20
    ks_feat:
      output: /home/mint/masters/data/noise_data/results/ks_feat.csv
    ks_obs:
      output: /home/mint/masters/data/noise_data/results/ks_obs.csv
    jensen_shannon_feat:
      output: /home/mint/masters/data/noise_data/results/jenshan_feat.csv
    jensen_shannon_obs:
      output: /home/mint/masters/data/noise_data/results/jenshan_obs.csv
    pca_2d_feat:
    pca_2d_obs:
output:
  path: /home/mint/masters/data/noise_data/results/report_{run}.html
'''