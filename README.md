# The PEARL Model

The ProjEcting Age, multimoRbidity, and poLypharmacy (PEARL) model is an agent-based simulation model of persons living with HIV (PWH) using antiretroviral therapy (ART) in the US (2009 - 2030). Due to effective treatment, PWH accessing care in the US now have a life expectancy approaching the general population. As these people survive to older ages, the burden of multimorbidity and polypharmacy will change to reflect this older age distribution and non-HIV-related healthcare costs will become increasingly important in caring for these individuals. Since the relevant results vary greatly by demographic among PWH in the US, the PEARL model accounts race, sex, HIV risk group explicitly. Among these key populations, outcomes depend further upon age, age at ART initiation, CD4 count at ART initiation, current CD4 count, and BMI. For most of its machinery, the PEARL model utilizes data from th North American AIDS Cohort Collaboration on Research and Design (NA-ACCORD). The NA-ACCORD is comprised of data from over 20 cohorts, 200 sites, and 190,000 HIV-infected participants, providing a data set that is widely representative of HIV care in North America. Data on overall PWH population size comes mostly from CDC surveillance data. The PEARL model has been constructed to achieve the following:

**Aim 1:** To fill the gap in knowledge by projecting age distributions of PWH using ART in the US through 2030 broken down by key population.

**Aim 2:** To project the burden of multimorbidity and polypharmacy among PWH using ART in the US through 2030.

**Aim 3:** To project the annual costs of non-HIV-related healthcare for PWH using ART in the US through 2030.

## Installation and First Run

Clone the repository onto your machine and enter the directory.

```
git clone git@github.com:PearlHivModelingTeam/pearlModel.git pearl
cd pearl
```

Here you will see 4 folder as well as a requirements.txt file.

```
code
config
out
param_files
requirements.txt
```

The ``code`` folder holds 3 numbered python scripts as well as a library with the PEARL classes and variables. The python files are numbered so that they can be run one after another to run a simulation. The ``config`` folder holds yaml files for specifying run configurations, the ``param_files`` folder holds the input parameters for use by PEARL and simulation results are generated in the ``out`` folder.

To start your first test run you must set up and activate a python virtual environment

```
python3 -m venv venv
source venv/bin/activate
```

You can then use the ``requirements.txt`` to install the necessary packages for running PEARL.

```
pip install --upgrade pip
pip install -r requirements.txt
```

Finally, enter the code folder and run the first two numbered scripts. This will generate parameters and run a simulation using the test.yaml config file. The simulation output can be found in ``out/test_yyyy_mm_dd`` with the date corresponding to the initiation of the run. The 3rd numbered script will convert the output from nested folders of csv files into hdf files which can be useful for analysis later. The third script takes the name of the output folder as a command line argument.

```
cd code
python 1_create_param_file.py
python 2_simulate.py
python 3_convert_csv_to_hdf.py --dir test_yyyy_mm_dd
```

## Configuration

The simulation script can be called with a command line argument pointing to a config file in order to run a simulation with different parameters or attributes. A template file lives ``config/template.yaml`` which contains all of the options available. In order to run a simulation with a specific config file simply call the simulation script as:
```
python 2_simulate --config my_config.yaml
```
and the output will be created at ``out/my_config_yyyy_mm_dd``

### ``group_names``
A list of the sex, race, and hiv-acquisition groups to include in the simulation. Can be any number of 
```
['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
 'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
 'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']
```

### ``num_cpus``
Number of cpus to use for the run, any positive integer. You should have access to this many cpus.

### ``replications``
Number of replications of each simulation to run with different seeds. Any positive integer.

### ``comorbidity_flag``
Flag indicating whether the simulation should include comorbidities, risk factors, and bmi. 0 or 1.

### ``new_dx``
String indicating which set of parameters to use for number of new diagnoses. ``base``, ``ehe``, ``sa``. The alternate models correspond to models used in some previous papers.

### ``mortality_model``
String corresponding to which model to use for mortality. ``by_sex_race_risk``, ``by_sex_race``, ``by_sex``, ``overall``. These models are presented in the mortality paper.

### ``mortality_threshold_flag``
Flag indicating whether simulation should include the mortality threshold functionality. 0 or 1.

### ``final_year``
Year to end the simulation. Integer between 2010 and 2035.

### ``verbose``
Flag whether to print yearly summary data while running for use while debugging. 0 or 1.

### ``sa_type``
For use when running groups of runs with certain parameters for various sensitivity analyses. ``none``, ``type1``, ``type2``, ``aim2_inc``, ``aim2_prev``, ``aim2_mort``. Produces a slightly different output structure and may not work with certain non-baseline parameters selected above.

### ``idu_threshold``
String corresponding to the different multipliers available for setting the mortality threshold for the idu population above other risk groups. ``2x``, ``5x``, ``10x``.

### ``bmi_intervention``
Flag indicating whether to simulate with the bmi intervention. 0 or 1.

### ``bmi_intervention_probability``
Proportion of eligible population to apply the bmi intervention to. Number between 0.0 and 1.0.
