# Imports
import shutil
import platform
import ray
import pearl
import yaml
import pkg_resources
import subprocess
from pathlib import Path
import argparse
from datetime import datetime


@ray.remote
def run(group_name_run, replication_run):
    replication_run_str = str(replication_run).zfill(len(str(config['replications'])))
    output_path = output_root_path/'csv_output'/group_name_run/f'replication_{replication_run_str}'
    rerun_path = rerun_root_path/'csv_output'/group_name_run/f'replication_{replication_run_str}' if rerun_root_path is not None else None
    parameters = pearl.Parameters(path=param_file_path, rerun_folder=rerun_path, output_folder=output_path,
                                  group_name=group_name_run, comorbidity_flag=config['comorbidity_flag'], new_dx=config['new_dx'],
                                  final_year=config['final_year'], mortality_model=config['mortality_model'],
                                  mortality_threshold_flag=config['mortality_threshold_flag'], idu_threshold=config['idu_threshold'],
                                  verbose=config['verbose'], bmi_intervention=config['bmi_intervention'], bmi_intervention_probability=config['bmi_intervention_probability'])
    pearl.Pearl(parameters, group_name_run, replication_run)


@ray.remote
def run_sa(sa_variable_run, sa_value_run, group_name_run, replication_run):
    replication_run_str = str(replication_run).zfill(len(str(config['replications'])))
    output_path = output_root_path/'csv_output'/f'{sa_variable_run}_{sa_value_run}'/group_name_run/f'replication_{replication_run_str}'
    rerun_path = rerun_root_path/'csv_output'/f'{sa_variable_run}_{sa_value_run}'/group_name_run/f'replication_{replication_run_str}' if rerun_root_path is not None else None
    parameters = pearl.Parameters(path=param_file_path, rerun_folder=rerun_path, output_folder=output_path,
                                  group_name=group_name_run, comorbidity_flag=config['comorbidity_flag'], new_dx=config['new_dx'],
                                  final_year=config['final_year'], mortality_model=config['mortality_model'],
                                  mortality_threshold_flag=config['mortality_threshold_flag'], idu_threshold=config['idu_threshold'],
                                  verbose=config['verbose'], sa_type=config['sa_type'], sa_variable=sa_variable_run,
                                  sa_value=sa_value_run, bmi_intervention=config['bmi_intervention'], bmi_intervention_probability=config['bmi_intervention_probability'])
    pearl.Pearl(parameters, group_name_run, replication_run)


start_time = datetime.now()

# Define the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--rerun')
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()

pearl_path = Path('..')
date_string = datetime.today().strftime('%Y-%m-%d')
param_file_path = pearl_path/'param_files/parameters.h5'

# Check that requirements.txt is met
with open(pearl_path/'requirements.txt', 'r') as requirements:
    pkg_resources.require(requirements)

# Original run, rerun, or test run?
rerun_root_path = None
if args.config:
    config_file_path = pearl_path/'config'/args.config
    output_root_path = pearl_path/f'out/{config_file_path.stem}_{date_string}'
elif args.rerun:
    rerun_root_path = pearl_path/'out'/args.rerun
    config_file_path = rerun_root_path/'config.yaml'
    output_root_path = pearl_path/f'out/{args.rerun}_rerun_{date_string}'
else:
    config_file_path = pearl_path/'config/test.yaml'
    output_root_path = pearl_path/f'out/{config_file_path.stem}_{date_string}'

# Load config_file
with open(config_file_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

# If it's a rerun check that python version and commit hash are correct else save those details for future runs
if args.rerun:
    print('This is a rerun')
    python_version = config['python_version']
    if python_version != platform.python_version():
        raise EnvironmentError("Incorrect python version for rerun")
    commit_hash = config['commit_hash']
    if commit_hash != subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip():
        raise EnvironmentError("Incorrect commit hash for rerun")
else:
    config['python_version'] = platform.python_version()
    config['commit_hash'] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

# Load sensitivity analysis variables
if config['sa_type'] == 'none':
    sa_variables = None
    sa_values = None
elif config['sa_type'] == 'type1':
    sa_variables = pearl.sa_type1_var
    sa_values = ['low', 'high']
elif config['sa_type'] == 'type2':
    sa_variables = pearl.sa_type2_var
    sa_values = [0.8, 1.2]
elif (config['sa_type'] == 'aim2_inc') | (config['sa_type'] == 'aim2_prev'):
    sa_variables = pearl.sa_aim2_var
    sa_values = [0.75, 1.25]
elif config['sa_type'] == 'aim2_mort':
    sa_variables = pearl.sa_aim2_mort_var
    sa_values = [0.75, 1.25]
else:
    raise ValueError("Unrecognized sensitivity analysis type")

# Create Output folder structure
if output_root_path.is_dir():
    if (config_file_path.stem == 'test') | args.overwrite:
        shutil.rmtree(output_root_path)
    else:
        raise FileExistsError("Output folder already exists")

if sa_variables is None:
    for group_name in config['group_names']:
        for replication in range(config['replications']):
            replication_str = str(replication).zfill(len(str(config['replications'])))
            output_path = output_root_path/'csv_output'/group_name/f'replication_{replication_str}'
            output_path.mkdir(parents=True)
else:
    for sa_variable in sa_variables:
        for sa_value in sa_values:
            for group_name in config['group_names']:
                for replication in range(config['replications']):
                    replication_str = str(replication).zfill(len(str(config['replications'])))
                    output_path = output_root_path/'csv_output'/f'{sa_variable}_{sa_value}'/group_name/f'replication_{replication_str}'
                    output_path.mkdir(parents=True)

# Copy config file to output dir
with open(output_root_path/'config.yaml', 'w') as yaml_file:
    yaml.safe_dump(config, yaml_file)

# Initialize ray with the desired number of threads
ray.init(num_cpus=config['num_cpus'])
if sa_variables is None:
    ray.get([run.remote(group_name, replication)
             for group_name in config['group_names']
             for replication in range(config['replications'])])
else:
    ray.get([run_sa.remote(sa_variable, sa_value, group_name, replication)
             for sa_variable in sa_variables
             for sa_value in sa_values
             for group_name in config['group_names']
             for replication in range(config['replications'])])

end_time = datetime.now()
print(f'Elapsed Time: {end_time - start_time}')