import os
import yaml
import shutil
import numpy as np
import pandas as pd
import statsmodels.stats.weightstats as weightstats
from pathlib import Path
from datetime import datetime
import argparse

sa_types = ['type1', 'type2', 'aim2_inc', 'aim2_prev', 'aim2_mort']

start_time = datetime.now()

# Define the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dir')
args = parser.parse_args()

pearl_path = Path('..')
in_dir = pearl_path/'out'/args.dir/'csv_output'
out_dir = pearl_path/'out'/args.dir/'hdf_output'
if out_dir.is_dir():
    shutil.rmtree(out_dir)
out_dir.mkdir()

print(in_dir)
print(out_dir)

combinable_tables = ['in_care_age', 'out_care_age', 'reengaged_age', 'ltfu_age', 'dead_in_care_age',
                     'dead_out_care_age', 'new_init_age', 'years_out', 'cd4_inits', 'cd4_in_care', 'cd4_out_care',
                     'incidence_in_care', 'incidence_out_care', 'prevalence_in_care', 'prevalence_out_care',
                     'prevalence_inits', 'prevalence_dead', 'mm_in_care', 'mm_out_care', 'mm_inits', 'mm_dead',
                     'mm_detail_in_care', 'mm_detail_out_care', 'mm_detail_inits', 'mm_detail_dead', 'pre_art_bmi',
                     'post_art_bmi']

# Load config_file
with open(in_dir/'../config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

if config['sa_type'] in sa_types:
    model_names = next(os.walk(in_dir))[1]
    group_names = next(os.walk(in_dir/model_names[0]))[1]
    replications = next(os.walk(in_dir / model_names[0] / group_names[0]))[1]
    output_tables = [x for x in next(os.walk(in_dir/model_names[0]/group_names[0]/replications[0]))[2] if x != 'random.state']
else:
    model_names = ['base']
    group_names = next(os.walk(in_dir))[1]
    replications = next(os.walk(in_dir/group_names[0]))[1]
    output_tables = [x for x in next(os.walk(in_dir/group_names[0]/replications[0]))[2] if x != 'random.state']

print(output_tables)
for output_table in output_tables:
    print(output_table)
    chunk_list = []
    for model_name in model_names:
        print('   ', model_name)
        for group_name in group_names:
            print('   ', '   ', group_name)
            for replication in replications:
                replication_int = int(replication.split(sep='_')[1])
                if config['sa_type'] in sa_types:
                    chunk_list.append(
                        pd.read_csv(in_dir/model_name/group_name/replication/output_table).assign(model=model_name,
                                                                                                  group=group_name,
                                                                                                  replication=replication_int))
                else:
                    chunk_list.append(
                        pd.read_csv(in_dir/group_name/replication/output_table).assign(model=model_name,
                                                                                       group=group_name,
                                                                                       replication=replication_int))
    df = pd.concat(chunk_list, ignore_index=True)
    measured_var = df.columns[-4]
    table_cols = df.columns[:-4]
    if output_table[:-4] in combinable_tables:
        groupby_cols = list(df.columns.drop(['group', measured_var]))
        ov = df.groupby(groupby_cols)[measured_var].sum().reset_index().assign(group='overall')
        df = pd.concat([df, ov], ignore_index=True)
    df = df.set_index(['model', 'group', 'replication'] + list(table_cols)).sort_index()
    if config['sa_type'] not in ['type1', 'type2', 'aim2']:
        df.index = df.index.droplevel()
    df.to_hdf(out_dir/f'{Path(output_table).stem}.h5', 'df', mode='w', format='table')

end_time = datetime.now()
print(f'Elapsed Time: {end_time - start_time}')
