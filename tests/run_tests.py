import time  
import subprocess
from yaml_files import immuneml_spec, simairr_spec, evalairr_spec

n_seq = 5000
n_runs = 5

ks_results = []

for n in range(1, n_runs):
    print(f'[LOG] RUNNING ITERATION {n}/{n_runs}')
    output_timestamp = int(time.time())

    ### SELECT REAL DATA
    # TODO
    
    ### RUN SIMAIRR
    subprocess.run(f'echo \"{simairr_spec(output_timestamp, n_runs)}\" > /data/yaml_files/olga_yaml_{output_timestamp}.yaml')
    subprocess.run(f'sim_airr -i /data/yaml_files/olga_yaml_{output_timestamp}.yaml')

    ### RUN IMMUNEML ENCODING
    subprocess.run(f'echo \"{immuneml_spec(output_timestamp)}\" > /data/yaml_files/immuneml_spec_{output_timestamp}.yaml')
    subprocess.run(f'docker run -it -v $(pwd):/data --name immunemlcontainer milenapavlovic/immuneml immune-ml /data/yaml_files/immuneml_spec_{output_timestamp}.yaml /data/encoded_output_{output_timestamp}/')
    
    ### RUN EVALAIRR
    subprocess.run(f'echo \"{evalairr_spec(output_timestamp)}\" > /data/yaml_files/main_yaml_{output_timestamp}.yaml')
    subprocess.run(f'evalairr -i /data/yaml_files/main_yaml_{output_timestamp}.yaml')

    ## COLLECT DATA
    # TODO