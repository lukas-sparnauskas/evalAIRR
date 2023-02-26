def immuneml_spec(timestamp):
  return f'''
definitions:
  datasets:
    real_dataset:
      format: OLGA
      params:
        path: \"./data/real_dataset_{timestamp}\"
        is_repertoire: True # whether to import a RepertoireDataset (True) or a SequenceDataset (False)
        metadata_file: "./data/metadata_real.csv" # metadata file for RepertoireDataset
        import_illegal_characters: False # remove sequences with illegal characters for the sequence_type being used
        import_empty_nt_sequences: True # keep sequences even though the nucleotide sequence might be empty
        import_empty_aa_sequences: False # filter out sequences if they don't have sequence_aa set
        # Optional fields with OLGA-specific defaults, only change when different behavior is required:
        # separator: "\t" # column separator
        # region_type: IMGT_CDR3 # what part of the sequence to import
    sim_dataset:
      format: OLGA
      params:
        path: \"./data/sim_dataset_{timestamp}\"
        is_repertoire: True # whether to import a RepertoireDataset (True) or a SequenceDataset (False)
        metadata_file: "./data/metadata_sim.csv" # metadata file for RepertoireDataset
        import_illegal_characters: False # remove sequences with illegal characters for the sequence_type being used
        import_empty_nt_sequences: True # keep sequences even though the nucleotide sequence might be empty
        import_empty_aa_sequences: False # filter out sequences if they don't have sequence_aa set
        # Optional fields with OLGA-specific defaults, only change when different behavior is required:
        # separator: "\t" # column separator
        # region_type: IMGT_CDR3 # what part of the sequence to import
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
  export_instruction:
    type: DatasetExport
    datasets:
      - real_dataset
      - sim_dataset
    number_of_processes: 1
    export_formats:
      - AIRR
      - ImmuneML
  expl_analysis_instruction:
    type: ExploratoryAnalysis
    analyses:
      analysis_1:
        dataset: real_dataset
        encoding: continuous_kmer
        report: dme_report
      analysis_2:
        dataset: sim_dataset
        encoding: continuous_kmer
        report: dme_report
    number_of_processes: 1
output:
  format: HTML
  '''

def simairr_spec(timestamp, n_seq): 
  return f'''
mode: baseline_repertoire_generation
olga_model: humanTRB
output_path: \"/home/mint/Desktop/masters/evalAIRR/data/sim_dataset_{timestamp}\"
n_repertoires: 200
n_sequences: {n_seq}
n_threads: 1
  '''

def evalairr_spec(timestamp):
  return f'''
datasets:
  real:
    path: \"/data/encoded_output_{timestamp}/encoded_real_1000_200.csv\"
  sim:
    path: \"/data/encoded_output_{timestamp}/encoded_sim_1000_200.csv\"
reports:
  general:
    ks:
      output: \"data/yaml_files/main_yaml_{timestamp}.yaml\"
  '''