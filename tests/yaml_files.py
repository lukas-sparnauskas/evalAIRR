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
    ks:
      output: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/ks.csv\"
    statistics:
      output_dir: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/\"
    observation_statistics:
      output_dir: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/\"
    distance:
      output: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/dist.csv\"
    observation_distance:
      output: \"/home/mint/masters/data/evalairrdata/run_{run}/results_{timestamp}/dist_obs.csv\"
output:
  path: NONE
  '''