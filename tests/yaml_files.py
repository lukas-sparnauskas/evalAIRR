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
    statistics:
      output_dir: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/\"
    observation_statistics:
      output_dir: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/\"
    distance:
      output: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/dist.csv\"
    observation_distance:
      output: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/dist_obs.csv\"
    jensen_shannon:
      output: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/jenshan.csv\"
    observation_jensen_shannon:
      output: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/jenshan_obs.csv\"
output:
  path: NONE
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
    statistics:
      output_dir: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/\"
    observation_statistics:
      output_dir: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/\"
    distance:
      output: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/dist.csv\"
    observation_distance:
      output: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/dist_obs.csv\"
    jensen_shannon:
      output: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/jenshan.csv\"
    observation_jensen_shannon:
      output: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/jenshan_obs.csv\"
output:
  path: NONE
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
    statistics:
      output_dir: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/\"
    observation_statistics:
      output_dir: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/\"
    distance:
      output: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/dist.csv\"
    observation_distance:
      output: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/dist_obs.csv\"
    jensen_shannon:
      output: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/jenshan.csv\"
    observation_jensen_shannon:
      output: \"/home/mint/masters/data/evalairrdata/th_run_{run}/results_{timestamp}/jenshan_obs.csv\"
output:
  path: NONE
  '''