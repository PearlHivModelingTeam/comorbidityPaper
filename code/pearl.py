# Imports
import os
import pickle
import numpy as np
import pandas as pd
import scipy.stats as stats
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

###############################################################################
# Constants                                                                   #
###############################################################################

# Status Constants
ART_NAIVE = 0
DELAYED = 1
ART_USER = 2
ART_NONUSER = 3
REENGAGED = 4
LTFU = 5
DYING_ART_USER = 6
DYING_ART_NONUSER = 7
DEAD_ART_USER = 8
DEAD_ART_NONUSER = 9


# Comorbidity stages
STAGE0 = ['hcv', 'smoking']
STAGE1 = ['anx', 'dpr']
STAGE2 = ['ckd', 'lipid', 'dm', 'ht']
STAGE3 = ['malig', 'esld', 'mi']


# Sensitivity analysis default values
sa_type1_var = ['lambda1', 'mu1', 'mu2', 'sigma1', 'sigma2', 'mortality_in_care', 'mortality_out_care',
                'loss_to_follow_up', 'cd4_increase', 'cd4_decrease', 'new_pop_size']
sa_type1_default_dict = {i: j for i, j in zip(sa_type1_var, len(sa_type1_var) * [None])}

sa_type2_var = ['users_2009_n', 'users_2009_age', 'users_2009_cd4', 'nonusers_2009_n', 'nonusers_2009_age',
                'nonusers_2009_cd4', 'initiators_n', 'initiators_age', 'initiators_cd4', 'disengagement',
                'reengagement', 'mortality_in_care', 'mortality_out_care', 'cd4_increase', 'cd4_decrease']
sa_type2_default_dict = {i: j for i, j in zip(sa_type2_var, len(sa_type2_var) * [1.0])}

sa_aim2_var = STAGE1 + STAGE2 + STAGE3
sa_aim2_default_dict = {i: j for i, j in zip(sa_aim2_var, len(sa_aim2_var) * [1.0])}

sa_aim2_mort_var = ['mortality_in_care', 'mortality_out_care']
sa_aim2_mort_default_dict = {i: j for i, j in zip(sa_aim2_mort_var, len(sa_aim2_mort_var) * [1.0])}


###############################################################################
# Functions                                                                   #
###############################################################################
def restricted_cubic_spline_var(x, t, i):
    """Return a numpy array of the i'th restricted cubic spline variable for numpy array x with knots t.
    Adapted from: https:github.com/harrelfe/Hmisc/blob/master/R/rcspline.eval.s
    """
    kd = (t[3] - t[0]) ** (2 / 3)
    y = (np.maximum(0, (x - t[i - 1]) / kd) ** 3
         - (np.maximum(0, (x - t[2]) / kd) ** 3) * (t[3] - t[i - 1]) / (t[3] - t[2])
         + (np.maximum(0, (x - t[3]) / kd) ** 3) * (t[2] - t[i - 1]) / (t[3] - t[2]))
    return y


def restricted_quadratic_spline_var(x, t, i):
    """Return a numpy array of the i'th restricted quadratic spline variable for numpy array x with knots t."""
    y = (np.maximum(0, x - t[i - 1]) ** 2 - np.maximum(0, x - t[3]) ** 2) / (t[3] - t[0])
    return y


def draw_from_trunc_norm(a, b, mu, sigma, n):
    """Return a numpy array filled with n values drawn from a truncated normal with the given parameters. If n=0 return
    an empty list.
    """
    y = []
    if n != 0:
        a_mod = (a - mu) / sigma
        b_mod = (b - mu) / sigma
        y = stats.truncnorm.rvs(a_mod, b_mod, loc=mu, scale=sigma, size=n)
    return y


def make_binary(x, n):
    """Return a binary representation of a decimal number x with n digits"""
    return ''.join(reversed([str((x >> i) & 1) for i in range(n)]))


def create_mm_detail_stats(pop):
    """Encode all comorbidity information as an 11 bit integer and return a dataframe counting the number of agents with every
    unique set of comorbidities.
    """
    all_comorbidities = STAGE0 + STAGE1 + STAGE2 + STAGE3
    df = pop[['age_cat'] + all_comorbidities].copy()
    df['multimorbidity'] = df[all_comorbidities].apply(lambda row: ''.join(row.values.astype(str)), axis='columns').apply(int, base=2)

    # Count how many people have each unique set of comorbidities
    df = df.groupby(['multimorbidity']).size().reset_index(name='n')
    return df


def simulate_ages(coeffs, pop_size):
    """Return numpy array of ages with length pop_size drawn from a mixed gaussian of given coefficients truncated
    at 18 and 85.
    """
    # Draw population size of each normal from the binomial distribution
    pop_size_1 = np.random.binomial(pop_size, coeffs.loc['lambda1', 'estimate'])
    pop_size_2 = pop_size - pop_size_1

    # Draw ages from truncated normal
    ages_1 = draw_from_trunc_norm(18, 85, coeffs.loc['mu1', 'estimate'], coeffs.loc['sigma1', 'estimate'], pop_size_1)
    ages_2 = draw_from_trunc_norm(18, 85, coeffs.loc['mu2', 'estimate'], coeffs.loc['sigma2', 'estimate'], pop_size_2)
    ages = np.concatenate((ages_1, ages_2))
    return ages


def calculate_cd4_increase(pop, knots, coeffs, vcov, sa):
    """Return new cd4 count of the given population as calculated via a linear function of time since art initiation modeled as a spline, initial
    cd4 count category, age category and cross terms.
    """
    # Calculate spline variables
    pop['time_from_h1yy'] = pop['year'] - pop['last_h1yy']
    pop['time_from_h1yy_'] = restricted_quadratic_spline_var(pop['time_from_h1yy'], knots.to_numpy(), 1)
    pop['time_from_h1yy__'] = restricted_quadratic_spline_var(pop['time_from_h1yy'], knots.to_numpy(), 2)
    pop['time_from_h1yy___'] = restricted_quadratic_spline_var(pop['time_from_h1yy'], knots.to_numpy(), 3)

    # Calculate CD4 Category Variables
    pop['cd4_cat_349'] = (pop['last_init_sqrtcd4n'].ge(np.sqrt(200.0)) & pop['last_init_sqrtcd4n'].lt(np.sqrt(350.0))).astype(int)
    pop['cd4_cat_499'] = (pop['last_init_sqrtcd4n'].ge(np.sqrt(350.0)) & pop['last_init_sqrtcd4n'].lt(np.sqrt(500.0))).astype(int)
    pop['cd4_cat_500'] = pop['last_init_sqrtcd4n'].ge(np.sqrt(500.0)).astype(int)

    # Create cross term variables
    pop['timecd4cat349_'] = pop['time_from_h1yy_'] * pop['cd4_cat_349']
    pop['timecd4cat499_'] = pop['time_from_h1yy_'] * pop['cd4_cat_499']
    pop['timecd4cat500_'] = pop['time_from_h1yy_'] * pop['cd4_cat_500']
    pop['timecd4cat349__'] = pop['time_from_h1yy__'] * pop['cd4_cat_349']
    pop['timecd4cat499__'] = pop['time_from_h1yy__'] * pop['cd4_cat_499']
    pop['timecd4cat500__'] = pop['time_from_h1yy__'] * pop['cd4_cat_500']
    pop['timecd4cat349___'] = pop['time_from_h1yy___'] * pop['cd4_cat_349']
    pop['timecd4cat499___'] = pop['time_from_h1yy___'] * pop['cd4_cat_499']
    pop['timecd4cat500___'] = pop['time_from_h1yy___'] * pop['cd4_cat_500']

    # Create numpy matrix
    pop_matrix = pop[['intercept', 'time_from_h1yy', 'time_from_h1yy_', 'time_from_h1yy__', 'time_from_h1yy___', 'cd4_cat_349', 'cd4_cat_499', 'cd4_cat_500', 'age_cat', 'timecd4cat349_',
                      'timecd4cat499_', 'timecd4cat500_', 'timecd4cat349__', 'timecd4cat499__', 'timecd4cat500__', 'timecd4cat349___', 'timecd4cat499___', 'timecd4cat500___']].to_numpy(dtype=float)

    # Perform matrix multiplication
    new_cd4 = np.matmul(pop_matrix, coeffs)

    # If this is a sensitivity analysis run take the upper or lower confidence interval prediction
    if sa is not None:
        # Calculate variance of prediction using matrix multiplication
        se = np.sqrt(np.sum(np.matmul(pop_matrix, vcov) * pop_matrix, axis=1))
        if sa == 'low:':
            new_cd4 -= 1.96 * se
        elif sa == 'high':
            new_cd4 += 1.96 * se

    new_cd4 = np.clip(new_cd4, 0, np.sqrt(2000))
    return new_cd4


def calculate_cd4_decrease(pop, coeffs, sa, vcov, smearing):
    """ Calculate out of care cd4 count via a linear function of years out of care and sqrt cd4 count at exit from care."""
    # Calculate the time_out variable and perform the matrix multiplication
    pop['time_out'] = pop['year'] - pop['ltfu_year']
    pop_matrix = pop[['intercept', 'time_out', 'sqrtcd4n_exit']].to_numpy(dtype=float)
    diff = np.matmul(pop_matrix, coeffs)

    # If this is a sensitivity analysis run take the upper or lower confidence interval prediction
    if sa is not None:
        se = np.sqrt(np.sum(np.matmul(pop_matrix, vcov) * pop_matrix, axis=1))
        if sa == 'low':
            diff -= 1.96 * se
        elif sa == 'high':
            diff += 1.96 * se

    new_cd4 = np.sqrt((pop['sqrtcd4n_exit'].to_numpy(dtype=float) ** 2) * np.exp(diff) * smearing)
    new_cd4 = np.clip(new_cd4, 0, np.sqrt(2000))
    return new_cd4


def create_mortality_in_care_pop_matrix(pop, comorbidity_flag, parameters):
    """Return the population matrix as a numpy array for calculating mortality in care. This log odds of mortality are a linear function of calendar year,
    ART init year category modeled as two binary variables, and age and sqrt initial cd4 count modeled as restricted cubic splines. If using comorbidities,
    log odds of mortality are a linear function of calendar year, age category, initial cd4 count, delta bmi and post art bmi modeled as restricted cubic
    splines, and presence of each individual comorbidity modeled as binary variables.
    """
    if comorbidity_flag:
        pop['post_art_bmi_'] = restricted_cubic_spline_var(pop['post_art_bmi'], parameters.mortality_in_care_post_art_bmi, 1)
        pop['post_art_bmi__'] = restricted_cubic_spline_var(pop['post_art_bmi'], parameters.mortality_in_care_post_art_bmi, 2)
        return pop[['age_cat', 'anx', 'post_art_bmi', 'post_art_bmi_', 'post_art_bmi__', 'ckd', 'dm', 'dpr', 'esld', 'h1yy',
                    'hcv', 'ht', 'intercept', 'lipid', 'malig', 'mi', 'smoking', 'init_sqrtcd4n', 'year']].to_numpy(dtype=float)
    else:
        pop['age_'] = restricted_cubic_spline_var(pop['age'], parameters.mortality_in_care_age, 1)
        pop['age__'] = restricted_cubic_spline_var(pop['age'], parameters.mortality_in_care_age, 2)
        pop['h1yy_cat_1'] = 0
        pop['h1yy_cat_2'] = 0
        pop.loc[pop['h1yy'].isin(np.arange(2009, 2013)), 'h1yy_cat_1'] = 1
        pop.loc[pop['h1yy'] >= 2013, 'h1yy_cat_2'] = 1
        pop['init_sqrtcd4n_'] = restricted_cubic_spline_var(pop['init_sqrtcd4n'], parameters.mortality_in_care_sqrtcd4, 1)
        pop['init_sqrtcd4n__'] = restricted_cubic_spline_var(pop['init_sqrtcd4n'], parameters.mortality_in_care_sqrtcd4, 2)
        return pop[['intercept', 'year', 'age', 'age_', 'age__', 'init_sqrtcd4n', 'init_sqrtcd4n_', 'init_sqrtcd4n__', 'h1yy_cat_1', 'h1yy_cat_2']].to_numpy(dtype=float)


def create_mortality_out_care_pop_matrix(pop, comorbidity_flag, parameters):
    """Return the population matrix as a numpy array for calculating mortality out of care. This log odds of mortality are a linear function of calendar
    year and age and sqrt cd4 count modeled as restricted cubic splines. If using comorbidities, log odds of mortality are a linear function of calendar
    year, age category, sqrt cd4 count, delta bmi and post art bmi modeled as restricted cubic splines, and presence of each individual comorbidity modeled
    as binary variables.
    """
    if comorbidity_flag:
        pop['post_art_bmi_'] = restricted_cubic_spline_var(pop['post_art_bmi'], parameters.mortality_out_care_post_art_bmi, 1)
        pop['post_art_bmi__'] = restricted_cubic_spline_var(pop['post_art_bmi'], parameters.mortality_out_care_post_art_bmi, 2)
        return pop[['age_cat', 'anx', 'post_art_bmi', 'post_art_bmi_', 'post_art_bmi__', 'ckd', 'dm', 'dpr', 'esld',
                    'hcv', 'ht', 'intercept', 'lipid', 'malig', 'mi', 'smoking', 'time_varying_sqrtcd4n', 'year']].to_numpy(dtype=float)
    else:
        pop['age_'] = restricted_cubic_spline_var(pop['age'], parameters.mortality_out_care_age, 1)
        pop['age__'] = restricted_cubic_spline_var(pop['age'], parameters.mortality_out_care_age, 2)
        pop['time_varying_sqrtcd4n_'] = restricted_cubic_spline_var(pop['time_varying_sqrtcd4n'], parameters.mortality_out_care_tv_sqrtcd4, 1)
        pop['time_varying_sqrtcd4n__'] = restricted_cubic_spline_var(pop['time_varying_sqrtcd4n'], parameters.mortality_out_care_tv_sqrtcd4, 2)
        return pop[['intercept', 'year', 'age', 'age_', 'age__', 'time_varying_sqrtcd4n', 'time_varying_sqrtcd4n_', 'time_varying_sqrtcd4n__']].to_numpy(dtype=float)


def create_comorbidity_pop_matrix(pop, condition, parameters):
    """Create and return the population matrix as a numpy array for calculating the probability of incidence of any of the 9 comorbidities.
    Each comorbidity has a unique set of variables as listed below.
    """
    # Calculate some intermediate variables needed for the population matrix
    pop['time_since_art'] = pop['year'] - pop['h1yy']
    pop['out_care'] = (pop['status'] == ART_NONUSER).astype(int)
    if condition in STAGE2 + STAGE3:
        pop['delta_bmi_'] = restricted_cubic_spline_var(pop['delta_bmi'], parameters.delta_bmi_dict[condition], 1)
        pop['delta_bmi__'] = restricted_cubic_spline_var(pop['delta_bmi'], parameters.delta_bmi_dict[condition], 2)
        pop['post_art_bmi_'] = restricted_cubic_spline_var(pop['post_art_bmi'], parameters.post_art_bmi_dict[condition], 1)
        pop['post_art_bmi__'] = restricted_cubic_spline_var(pop['post_art_bmi'], parameters.post_art_bmi_dict[condition], 2)

    if condition == 'anx':
        return pop[['age', 'init_sqrtcd4n', 'dpr', 'time_since_art', 'hcv', 'intercept', 'smoking', 'year']].to_numpy(dtype=float)
    elif condition == 'dpr':
        return pop[['age', 'anx', 'init_sqrtcd4n', 'time_since_art', 'hcv', 'intercept', 'smoking', 'year']].to_numpy(dtype=float)
    elif condition == 'ckd':
        return pop[['age', 'anx', 'delta_bmi_', 'delta_bmi__', 'delta_bmi', 'post_art_bmi', 'post_art_bmi_', 'post_art_bmi__', 'init_sqrtcd4n',
                    'dm', 'dpr', 'time_since_art', 'hcv', 'ht', 'intercept', 'lipid', 'smoking', 'year']].to_numpy(dtype=float)
    elif condition == 'lipid':
        return pop[['age', 'anx', 'delta_bmi_', 'delta_bmi__', 'delta_bmi', 'post_art_bmi', 'post_art_bmi_', 'post_art_bmi__', 'init_sqrtcd4n',
                    'ckd', 'dm', 'dpr', 'time_since_art', 'hcv', 'ht', 'intercept', 'smoking', 'year']].to_numpy(dtype=float)
    elif condition == 'dm':
        return pop[['age', 'anx', 'delta_bmi_', 'delta_bmi__', 'delta_bmi', 'post_art_bmi', 'post_art_bmi_', 'post_art_bmi__', 'init_sqrtcd4n',
                    'ckd', 'dpr', 'time_since_art', 'hcv', 'ht', 'intercept', 'lipid', 'smoking', 'year']].to_numpy(dtype=float)
    elif condition == 'ht':
        return pop[['age', 'anx', 'delta_bmi_', 'delta_bmi__', 'delta_bmi', 'post_art_bmi', 'post_art_bmi_', 'post_art_bmi__', 'init_sqrtcd4n',
                    'ckd', 'dm', 'dpr', 'time_since_art', 'hcv', 'intercept', 'lipid', 'smoking', 'year']].to_numpy(dtype=float)
    elif condition in ['malig', 'esld', 'mi']:
        return pop[['age', 'anx', 'delta_bmi_', 'delta_bmi__', 'delta_bmi', 'post_art_bmi', 'post_art_bmi_', 'post_art_bmi__', 'init_sqrtcd4n',
                    'ckd', 'dm', 'dpr', 'time_since_art', 'hcv', 'ht', 'intercept', 'lipid', 'smoking', 'year']].to_numpy(dtype=float)


def calculate_pre_art_bmi(pop, model, coeffs, t_age, t_h1yy, rse, smearing):
    """Calculate and return pre art bmi for a given population dataframe. Each subpopulation can use a different model of pre art bmi."""
    # Calculate pre art bmi using one of 5 different models depending on subpopulation
    pre_art_bmi = np.nan
    if model == 6:
        pop['age_'] = restricted_cubic_spline_var(pop['init_age'], t_age, 1)
        pop['age__'] = restricted_cubic_spline_var(pop['init_age'], t_age, 2)
        h1yy = pop['h1yy'].values
        pop['h1yy_'] = restricted_cubic_spline_var(h1yy, t_h1yy, 1)
        pop['h1yy__'] = restricted_cubic_spline_var(h1yy, t_h1yy, 2)
        pop_matrix = pop[['init_age', 'age_', 'age__', 'h1yy', 'h1yy_', 'h1yy__', 'intercept']].to_numpy(dtype=float)
        log_pre_art_bmi = np.matmul(pop_matrix, coeffs.to_numpy(dtype=float))

    elif model == 5:
        pop['age_'] = restricted_cubic_spline_var(pop['init_age'], t_age, 1)
        pop['age__'] = restricted_cubic_spline_var(pop['init_age'], t_age, 2)
        pop_matrix = pop[['init_age', 'age_', 'age__', 'h1yy', 'intercept']].to_numpy(dtype=float)
        log_pre_art_bmi = np.matmul(pop_matrix, coeffs.to_numpy(dtype=float))

    elif model == 3:
        pop_matrix = pop[['init_age', 'h1yy', 'intercept']].to_numpy(dtype=float)
        log_pre_art_bmi = np.matmul(pop_matrix, coeffs.to_numpy(dtype=float))

    elif model == 2:
        pop['age_'] = (pop['init_age'] >= 30) & (pop['init_age'] < 40)
        pop['age__'] = (pop['init_age'] >= 40) & (pop['init_age'] < 50)
        pop['age___'] = (pop['init_age'] >= 50) & (pop['init_age'] < 60)
        pop['age____'] = pop['init_age'] >= 60
        h1yy = pop['h1yy'].values
        pop['h1yy_'] = restricted_cubic_spline_var(h1yy, t_h1yy, 1)
        pop['h1yy__'] = restricted_cubic_spline_var(h1yy, t_h1yy, 2)
        pop_matrix = pop[['age_', 'age__', 'age___', 'age____',
                          'h1yy', 'h1yy_', 'h1yy__', 'intercept']].to_numpy(dtype=float)
        log_pre_art_bmi = np.matmul(pop_matrix, coeffs.to_numpy(dtype=float))

    elif model == 1:
        pop['age_'] = (pop['init_age'] >= 30) & (pop['init_age'] < 40)
        pop['age__'] = (pop['init_age'] >= 40) & (pop['init_age'] < 50)
        pop['age___'] = (pop['init_age'] >= 50) & (pop['init_age'] < 60)
        pop['age____'] = pop['init_age'] >= 60
        pop_matrix = pop[['age_', 'age__', 'age___', 'age____', 'h1yy', 'intercept']].to_numpy(dtype=float)
        log_pre_art_bmi = np.matmul(pop_matrix, coeffs.to_numpy(dtype=float))

    log_pre_art_bmi = np.log10(smearing) + log_pre_art_bmi
    log_pre_art_bmi = log_pre_art_bmi.T[0]
    log_pre_art_bmi = np.vectorize(draw_from_trunc_norm)(np.log10(10), np.log10(65),
                                                         log_pre_art_bmi, rse, 1)
    pre_art_bmi = 10.0 ** log_pre_art_bmi
    return pre_art_bmi


def calculate_post_art_bmi(pop, parameters, intervention=False):
    """Calculate and return post art bmi for a population. Sqrt of post art bmi is modeled as a linear function of ART initiation year and
    age, sqrt of pre art bmi, sqrt initial cd4 count, and sqrt of cd4 count 2 years after art initiation all modeled as restricted cubic splines.
    """
    # Copy coefficients and knots to more reasonable variable names
    coeffs = parameters.post_art_bmi
    t_age = parameters.post_art_bmi_age_knots
    t_pre_sqrt = parameters.post_art_bmi_pre_art_bmi_knots
    t_sqrtcd4 = parameters.post_art_bmi_cd4_knots
    t_sqrtcd4_post = parameters.post_art_bmi_cd4_post_knots
    rse = parameters.post_art_bmi_rse

    # Calculate spline variables
    pop['age_'] = restricted_cubic_spline_var(pop['init_age'], t_age, 1)
    pop['age__'] = restricted_cubic_spline_var(pop['init_age'], t_age, 2)
    pop['pre_sqrt'] = pop['pre_art_bmi'] ** 0.5
    pop['pre_sqrt_'] = restricted_cubic_spline_var(pop['pre_sqrt'], t_pre_sqrt, 1)
    pop['pre_sqrt__'] = restricted_cubic_spline_var(pop['pre_sqrt'], t_pre_sqrt, 2)
    pop['sqrtcd4'] = pop['init_sqrtcd4n']
    pop['sqrtcd4_'] = restricted_cubic_spline_var(pop['sqrtcd4'], t_sqrtcd4, 1)
    pop['sqrtcd4__'] = restricted_cubic_spline_var(pop['sqrtcd4'], t_sqrtcd4, 2)

    # Calculate cd4 count 2 years after art initiation and its spline terms
    pop_future = pop.copy().assign(age=pop['init_age'] + 2)
    pop_future['year'] = pop['h1yy'] + 2
    pop_future['age_cat'] = np.floor(pop_future['age'] / 10)
    pop_future.loc[pop_future['age_cat'] < 2, 'age_cat'] = 2
    pop_future.loc[pop_future['age_cat'] > 7, 'age_cat'] = 7
    pop['sqrtcd4_post'] = calculate_cd4_increase(pop_future, parameters.cd4_increase_knots, parameters.cd4_increase.to_numpy(dtype=float),
                                                 parameters.cd4_increase_vcov.to_numpy(dtype=float), parameters.sa_type1_dict['cd4_increase'])

    pop['sqrtcd4_post_'] = restricted_cubic_spline_var(pop['sqrtcd4_post'], t_sqrtcd4_post, 1)
    pop['sqrtcd4_post__'] = restricted_cubic_spline_var(pop['sqrtcd4_post'], t_sqrtcd4_post, 2)

    # Create the population matrix and perform the matrix multiplication
    pop_matrix = pop[['init_age', 'age_', 'age__', 'h1yy', 'intercept', 'pre_sqrt', 'pre_sqrt_', 'pre_sqrt__', 'sqrtcd4',
                      'sqrtcd4_', 'sqrtcd4__', 'sqrtcd4_post', 'sqrtcd4_post_', 'sqrtcd4_post__']].to_numpy(dtype=float)
    sqrt_post_art_bmi = np.matmul(pop_matrix, coeffs.to_numpy(dtype=float))
    sqrt_post_art_bmi = sqrt_post_art_bmi.T[0]

    if intervention:
        sqrt_post_art_bmi = np.vectorize(draw_from_trunc_norm)(np.sqrt(10), np.sqrt(30),
                                                               sqrt_post_art_bmi, rse, 1)
    else:
        sqrt_post_art_bmi = np.vectorize(draw_from_trunc_norm)(np.sqrt(10), np.sqrt(65),
                                                               sqrt_post_art_bmi, rse, 1)
    post_art_bmi = sqrt_post_art_bmi ** 2.0

    return post_art_bmi


def create_ltfu_pop_matrix(pop, knots):
    """Create and return the population matrix as a numpy array for use in calculating probability of loss to follow up."""
    # Create all needed intermediate variables
    pop['age_'] = restricted_quadratic_spline_var(pop['age'], knots.to_numpy(), 1)
    pop['age__'] = restricted_quadratic_spline_var(pop['age'], knots.to_numpy(), 2)
    pop['age___'] = restricted_quadratic_spline_var(pop['age'], knots.to_numpy(), 3)
    pop['haart_period'] = (pop['h1yy'].values > 2010).astype(int)
    return pop[['intercept', 'age', 'age_', 'age__', 'age___', 'year', 'init_sqrtcd4n', 'haart_period']].to_numpy(dtype=float)


def calculate_prob(pop, coeffs, sa, vcov):
    """Calculate and return a numpy array of individual probabilities from logistic regression given the population and coefficient matrices.
    Used for multiple logistic regression functions.
    """
    # Calculate log odds using a matrix multiplication
    log_odds = np.matmul(pop, coeffs)

    if sa is not None:
        # Calculate variance of prediction using matrix multiplication
        a = np.matmul(pop, vcov)

        b = a * pop
        c = np.sum(b, axis=1)

        se = np.sqrt(c)
        if sa == 'low':
            log_odds = log_odds - (1.96 * se)
        elif sa == 'high':
            log_odds = log_odds + (1.96 * se)

    # Convert to probability
    prob = np.exp(log_odds) / (1.0 + np.exp(log_odds))
    return prob


def simulate_new_dx(new_dx, linkage_to_care):
    """Return the number of ART non-users in 2009 as an integer and the number of agents entering the model each year as art users and non-users
    as a dataframe. Draw number of new diagnoses from a uniform distribution between upper and lower bounds. Calculate number of new art
    initiators by assuming a certain number link in the first year as estimated by a linear regression on CDC data, capped at 95%. We assume
    that 40% of the remaining population links to care over the next 3 years. We assume that 70% of those linking to care begin ART, rising
    to 85% in 2011 and 97% afterwards. We take the number of people not initiating ART 2006 - 2009 in this calculation to be the out of
    care population size in 2009 for our simulation.
    """

    # Draw new dx from a uniform distribution between upper and lower for 2016-final_year
    new_dx['n_dx'] = new_dx['lower'] + (new_dx['upper'] - new_dx['lower']) * np.random.uniform()

    # Only a proportion of new diagnoses link to care and 40% of the remaining link in the next 3 years
    new_dx['unlinked'] = new_dx['n_dx'] * (1 - linkage_to_care['link_prob'])
    new_dx['gardner_per_year'] = new_dx['unlinked'] * 0.4 / 3.0
    new_dx['year0'] = new_dx['n_dx'] * linkage_to_care['link_prob']
    new_dx['year1'] = new_dx['gardner_per_year'].shift(1, fill_value=0)
    new_dx['year2'] = new_dx['gardner_per_year'].shift(2, fill_value=0)
    new_dx['year3'] = new_dx['gardner_per_year'].shift(3, fill_value=0)
    new_dx['total_linked'] = new_dx['year0'] + new_dx['year1'] + new_dx['year2'] + new_dx['year3']

    # Proportion of those linked to care start ART
    new_dx['art_initiators'] = (new_dx['total_linked'] * linkage_to_care['art_prob']).astype(int)
    new_dx['art_delayed'] = (new_dx['total_linked'] * (1 - linkage_to_care['art_prob'])).astype(int)

    # Count those not starting art 2006 - 2009 as initial ART nonusers
    n_initial_nonusers = new_dx.loc[np.arange(2006, 2010), 'art_delayed'].sum()

    # Compile list of number of new agents to be introduced in the model
    new_agents = new_dx.loc[np.arange(2010, 2036), ['art_initiators', 'art_delayed']]

    return n_initial_nonusers, new_agents


def apply_bmi_intervention(pop, parameters):
    pop['post_art_bmi_inter'] = calculate_post_art_bmi(pop.copy(), parameters, intervention=True)
    pop['eligible'] = (pop['pre_art_bmi'] >= 18.5) & (pop['pre_art_bmi'] <= 30)
    pop['become_obese'] = pop['post_art_bmi'] > 30
    pop['inter_effective'] = np.random.choice([1, 0], size=len(pop), replace=True,
                                              p=[parameters.bmi_intervention_probability, 1 - parameters.bmi_intervention_probability])
    pop['intervention_year'] = pop['h1yy'].isin(range(2021, 2027))
    pop['intervention'] = pop['intervention_year'] & pop['eligible'] & pop['become_obese'] & pop['inter_effective']
    pop.loc[pop['intervention'], 'post_art_bmi'] = pop.loc[pop['intervention'], 'post_art_bmi_inter']
    return pop['post_art_bmi']


###############################################################################
# Pearl Class                                                                 #
###############################################################################

class Pearl:
    """The PEARL class runs a simulation when initialized."""
    def __init__(self, parameters, group_name, replication):
        """Takes an instance of the Parameters class, the group name and replication number and runs a simulation."""
        self.group_name = group_name
        self.replication = replication
        self.year = 2009
        self.parameters = parameters

        # If this is a rerun, reload the random state
        if self.parameters.rerun_folder is not None:
            with open(self.parameters.rerun_folder/'random.state', 'rb') as state_file_load, \
                    open(self.parameters.output_folder/'random.state', 'wb') as state_file_save:
                state = pickle.load(state_file_load)
                pickle.dump(state, state_file_save)
            np.random.set_state(state)
        else:
            state = np.random.get_state()
            with open(self.parameters.output_folder/'random.state', 'wb') as state_file:
                pickle.dump(state, state_file)

        # Initiate output class
        self.stats = Statistics(output_folder=self.parameters.output_folder,
                                comorbidity_flag=self.parameters.comorbidity_flag,
                                sa_type=self.parameters.sa_type)

        # Simulate number of new art initiators and initial nonusers
        n_initial_nonusers, n_new_agents = simulate_new_dx(self.parameters.new_dx.copy(), self.parameters.linkage_to_care)

        # Initialize population dataframe
        self.population = pd.DataFrame()

        # Create art using 2009 population
        self.make_user_pop_2009()

        # Create art non-using 2009 population
        self.make_nonuser_pop_2009(n_initial_nonusers)

        # Create population of new art initiators
        self.make_new_population(n_new_agents)

        # First recording of stats
        self.record_stats()

        # Print populations
        if self.parameters.verbose:
            print(self)

        # Move to 2010
        self.year += 1

        # Run
        self.run()

        # Save output
        self.stats.save()

    def __str__(self):
        """Output diagnostic information when the verbose flag is true."""
        total = len(self.population.index)
        in_care = len(self.population.loc[self.population['status'] == ART_USER])
        out_care = len(self.population.loc[self.population['status'] == ART_NONUSER])
        dead_in_care = len(self.population.loc[self.population['status'] == DEAD_ART_USER])
        dead_out_care = len(self.population.loc[self.population['status'] == DEAD_ART_NONUSER])
        uninitiated_user = len(self.population.loc[self.population['status'].isin([ART_NAIVE])])

        string = 'Year End: ' + str(self.year) + '\n'
        string += 'Total Population Size: ' + str(total) + '\n'
        string += 'In Care Size: ' + str(in_care) + '\n'
        string += 'Out Care Size: ' + str(out_care) + '\n'
        string += 'Dead In Care Size: ' + str(dead_in_care) + '\n'
        string += 'Dead Out Care Size: ' + str(dead_out_care) + '\n'
        string += 'Uninitiated User Size: ' + str(uninitiated_user) + '\n'
        return string

    def make_user_pop_2009(self):
        """ Create and return initial 2009 population dataframe. Draw ages from a mixed normal distribution truncated at 18
        and 85. Assign ART initiation year using proportions from NA-ACCORD data. Draw sqrt CD4 count from a normal
        distribution truncated at 0 and sqrt(2000). If doing an Aim 2 simulation, assign bmi, comorbidities, and multimorbidity
        using their respective models.
        """
        # Create population dataframe
        population = pd.DataFrame()

        # Draw ages from the truncated mixed gaussian
        n_initial_users = self.parameters.on_art_2009[0]
        n_initial_users = int(self.parameters.sa_type2_dict['users_2009_n'] * n_initial_users)  # Sensitivity analysis
        population['age'] = simulate_ages(self.parameters.age_in_2009, n_initial_users)
        population['age'] = self.parameters.sa_type2_dict['users_2009_age'] * population['age']  # Sensitivity analysis
        population.loc[population['age'] < 18, 'age'] = 18
        population.loc[population['age'] > 85, 'age'] = 85

        # Create age categories
        population['age'] = np.floor(population['age'])
        population['age_cat'] = np.floor(population['age'] / 10)
        population.loc[population['age_cat'] > 7, 'age_cat'] = 7
        population['id'] = range(population.index.size)
        population = population.sort_values('age')
        population = population.set_index(['age_cat', 'id'])

        # Assign H1YY to match NA-ACCORD distribution from h1yy_by_age_2009
        for age_cat, grouped in population.groupby('age_cat'):
            h1yy_data = self.parameters.h1yy_by_age_2009.loc[age_cat].reset_index()
            population.loc[age_cat, 'h1yy'] = np.random.choice(h1yy_data['h1yy'], size=len(grouped), p=h1yy_data['pct'])

        # Reindex for group operation
        population['h1yy'] = population['h1yy'].astype(int)
        population = population.reset_index().set_index(['h1yy', 'id']).sort_index()

        # For each h1yy draw values of sqrt_cd4n from a normal truncated at 0 and sqrt 2000
        for h1yy, group in population.groupby(level=0):
            mu = self.parameters.cd4n_by_h1yy_2009.loc[(h1yy, 'mu'), 'estimate']
            sigma = self.parameters.cd4n_by_h1yy_2009.loc[(h1yy, 'sigma'), 'estimate']
            size = group.shape[0]
            sqrt_cd4n = draw_from_trunc_norm(0, np.sqrt(2000.0), mu, sigma, size)
            population.loc[(h1yy,), 'init_sqrtcd4n'] = sqrt_cd4n
        population = population.reset_index().set_index('id').sort_index()

        # Sensitivity Analysis
        population['init_sqrtcd4n'] = np.sqrt(self.parameters.sa_type2_dict['users_2009_cd4'] * np.power(population['init_sqrtcd4n'], 2))

        # Toss out age_cat < 2
        population.loc[population['age_cat'] < 2, 'age_cat'] = 2

        # Add final columns used for calculations and output
        population['last_h1yy'] = population['h1yy']
        population['last_init_sqrtcd4n'] = population['init_sqrtcd4n']
        population['init_age'] = population['age'] - (2009 - population['h1yy'])
        population['n_lost'] = 0
        population['years_out'] = 0
        population['year_died'] = np.nan
        population['sqrtcd4n_exit'] = 0
        population['ltfu_year'] = 0
        population['return_year'] = 0
        population['intercept'] = 1.0
        population['year'] = 2009

        # Calculate time varying cd4 count
        population['time_varying_sqrtcd4n'] = calculate_cd4_increase(population.copy(), self.parameters.cd4_increase_knots, self.parameters.cd4_increase.to_numpy(dtype=float),
                                                                     self.parameters.cd4_increase_vcov.to_numpy(dtype=float), self.parameters.sa_type1_dict['cd4_increase'])

        # Set status and initiate out of care variables
        population['status'] = ART_USER

        # If doing a comorbidity simulation, add bmi, comorbidity, and multimorbidity columns
        if self.parameters.comorbidity_flag:
            # Bmi
            population['pre_art_bmi'] = calculate_pre_art_bmi(population.copy(), self.parameters.pre_art_bmi_model, self.parameters.pre_art_bmi,
                                                              self.parameters.pre_art_bmi_age_knots, self.parameters.pre_art_bmi_h1yy_knots,
                                                              self.parameters.pre_art_bmi_rse, self.parameters.pre_art_bmi_smearing)
            population['post_art_bmi'] = calculate_post_art_bmi(population.copy(), self.parameters)
            population['delta_bmi'] = population['post_art_bmi'] - population['pre_art_bmi']

            # Apply comorbidities
            for condition in STAGE0 + STAGE1 + STAGE2 + STAGE3:
                population[condition] = (np.random.rand(len(population.index)) < self.parameters.prev_users_dict[condition].values).astype(int)
            population['mm'] = population[STAGE2 + STAGE3].sum(axis=1)

        # Sort columns alphabetically
        population = population.reindex(sorted(population), axis=1)

        # Record classic one-way sa input
        sa_initial_cd4_in_care = pd.DataFrame(data={'mean_cd4': (population['init_sqrtcd4n'] ** 2).mean(),
                                                    'n': len(population)}, index=[0])
        self.stats.sa_initial_cd4_in_care = self.stats.sa_initial_cd4_in_care.append(sa_initial_cd4_in_care, ignore_index=True)

        # Save population as pearl population
        self.population = population

    def make_nonuser_pop_2009(self, n_initial_nonusers):
        """ Create and return initial 2009 population dataframe. Draw ages from a mixed normal distribution truncated at 18
        and 85. Assign ART initiation year using proportions from NA-ACCORD data. Draw sqrt CD4 count from a normal
        distribution truncated at 0 and sqrt(2000). If doing an Aim 2 simulation, assign bmi, comorbidities, and multimorbidity
        using their respective models.
        """
        # Create population dataframe
        population = pd.DataFrame()

        # Draw ages from the truncated mixed gaussian
        n_initial_nonusers = int(self.parameters.sa_type2_dict['nonusers_2009_n'] * n_initial_nonusers)  # Sensitivity analysis
        population['age'] = simulate_ages(self.parameters.age_in_2009, n_initial_nonusers)
        population['age'] = self.parameters.sa_type2_dict['nonusers_2009_age'] * population['age']  # Sensitivity analysis
        population.loc[population['age'] < 18, 'age'] = 18
        population.loc[population['age'] > 85, 'age'] = 85

        # Create age categories
        population['age'] = np.floor(population['age'])
        population['age_cat'] = np.floor(population['age'] / 10)
        population.loc[population['age_cat'] > 7, 'age_cat'] = 7
        population['id'] = range(population.index.size)
        population = population.sort_values('age')
        population = population.set_index(['age_cat', 'id'])

        # Assign H1YY to match NA-ACCORD distribution from h1yy_by_age_2009
        for age_cat, grouped in population.groupby('age_cat'):
            h1yy_data = self.parameters.h1yy_by_age_2009.loc[age_cat].reset_index()
            population.loc[age_cat, 'h1yy'] = np.random.choice(h1yy_data['h1yy'], size=len(grouped), p=h1yy_data['pct'])

        # Reindex for group operation
        population['h1yy'] = population['h1yy'].astype(int)
        population = population.reset_index().set_index(['h1yy', 'id']).sort_index()

        # For each h1yy draw values of sqrt_cd4n from a normal truncated at 0 and sqrt 2000
        for h1yy, group in population.groupby(level=0):
            mu = self.parameters.cd4n_by_h1yy_2009.loc[(h1yy, 'mu'), 'estimate']
            sigma = self.parameters.cd4n_by_h1yy_2009.loc[(h1yy, 'sigma'), 'estimate']
            size = group.shape[0]
            sqrt_cd4n = draw_from_trunc_norm(0, np.sqrt(2000.0), mu, sigma, size)
            population.loc[(h1yy,), 'init_sqrtcd4n'] = sqrt_cd4n
        population = population.reset_index().set_index('id').sort_index()

        # Sensitivity Analysis
        population['init_sqrtcd4n'] = np.sqrt(self.parameters.sa_type2_dict['nonusers_2009_cd4'] * np.power(population['init_sqrtcd4n'], 2))

        # Toss out age_cat < 2
        population.loc[population['age_cat'] < 2, 'age_cat'] = 2

        # Add final columns used for calculations and output
        population['last_h1yy'] = population['h1yy']
        population['last_init_sqrtcd4n'] = population['init_sqrtcd4n']
        population['init_age'] = population['age'] - (2009 - population['h1yy'])
        population['n_lost'] = 0
        population['years_out'] = 0
        population['year_died'] = np.nan
        population['sqrtcd4n_exit'] = 0
        population['ltfu_year'] = 0
        population['return_year'] = 0
        population['intercept'] = 1.0
        population['year'] = 2009

        # Calculate time varying cd4 count
        population['time_varying_sqrtcd4n'] = calculate_cd4_increase(population.copy(), self.parameters.cd4_increase_knots, self.parameters.cd4_increase.to_numpy(dtype=float),
                                                                     self.parameters.cd4_increase_vcov.to_numpy(dtype=float), self.parameters.sa_type1_dict['cd4_increase'])

        # Set status and initiate out of care variables
        years_out_of_care = np.random.choice(a=self.parameters.years_out_of_care['years'], size=n_initial_nonusers, p=self.parameters.years_out_of_care['probability'])
        population['status'] = ART_NONUSER
        population['sqrtcd4n_exit'] = population['time_varying_sqrtcd4n']
        population['ltfu_year'] = 2009
        population['return_year'] = 2009 + years_out_of_care
        population['n_lost'] += 1

        # If doing a comorbidity simulation, add bmi, comorbidity, and multimorbidity columns
        if self.parameters.comorbidity_flag:
            # Bmi
            population['pre_art_bmi'] = calculate_pre_art_bmi(population.copy(), self.parameters.pre_art_bmi_model, self.parameters.pre_art_bmi,
                                                              self.parameters.pre_art_bmi_age_knots, self.parameters.pre_art_bmi_h1yy_knots,
                                                              self.parameters.pre_art_bmi_rse, self.parameters.pre_art_bmi_smearing)
            population['post_art_bmi'] = calculate_post_art_bmi(population.copy(), self.parameters)
            population['delta_bmi'] = population['post_art_bmi'] - population['pre_art_bmi']

            # Apply comorbidities
            for condition in STAGE0:
                population[condition] = (np.random.rand(len(population.index)) < self.parameters.prev_users_dict[condition].values).astype(int)
            for condition in STAGE1 + STAGE2 + STAGE3:
                population[condition] = (np.random.rand(len(population.index)) < (self.parameters.prev_users_dict[condition].values) *
                                         self.parameters.sa_aim2_prev_dict[condition]).astype(int)
            population['mm'] = population[STAGE2 + STAGE3].sum(axis=1)

        # Sort columns alphabetically
        population = population.reindex(sorted(population), axis=1)

        # Record classic one-way sa input
        sa_initial_cd4_out_care = pd.DataFrame(data={'mean_cd4': (population['init_sqrtcd4n'] ** 2).mean(),
                                                     'n': len(population)}, index=[0])
        self.stats.sa_initial_cd4_out_care = self.stats.sa_initial_cd4_out_care.append(sa_initial_cd4_out_care, ignore_index=True)

        # Append new population to pearl population
        self.population = self.population.append(population)

    def make_new_population(self, n_new_agents):
        """Create and return the population initiating ART during the simulation. Age and CD4 count distribution parameters are taken from a
        linear regression until 2018 and drawn from a uniform distribution between the 2018 values and the predicted values thereafter. Ages
        are drawn from the two-component mixed normal distribution truncated at 18 and 85 defined by the generated parameters. The n_new_agents
        dataframe defines the population size of ART initiators and those not initiating immediately. The latter population begins ART some years
        later as drawn from a normalized, truncated Poisson distribution. The sqrt CD4 count at ART initiation for each agent is drawn from a normal
        distribution truncated at 0 and sqrt 2000 as defined by the generated parameters. If this is an Aim 2 simulation, generate bmi, comorbidities,
        and multimorbidity from their respective distributions.
        """
        # Draw a random value between predicted and 2018 predicted value for years greater than 2018
        rand = np.random.rand(len(self.parameters.age_by_h1yy.index))
        self.parameters.age_by_h1yy['estimate'] = (rand * (self.parameters.age_by_h1yy['high_value'] - self.parameters.age_by_h1yy['low_value'])) + self.parameters.age_by_h1yy['low_value']
        self.stats.art_coeffs = self.parameters.age_by_h1yy[['estimate']].assign(variable='age').reset_index()

        rand = np.random.rand(len(self.parameters.cd4n_by_h1yy.index))
        self.parameters.cd4n_by_h1yy['estimate'] = (rand * (self.parameters.cd4n_by_h1yy['high_value'] - self.parameters.cd4n_by_h1yy['low_value'])) + self.parameters.cd4n_by_h1yy['low_value']
        art_coeffs_cd4 = self.parameters.cd4n_by_h1yy[['estimate']].assign(variable='cd4').reset_index()
        self.stats.art_coeffs = pd.concat([self.stats.art_coeffs, art_coeffs_cd4]).rename(columns={'h1yy': 'year'})[['year', 'variable', 'param', 'estimate']]

        # Create population
        population = pd.DataFrame()

        n_new_agents = (self.parameters.sa_type2_dict['initiators_n'] * n_new_agents).astype(int)  # Sensitivity analysis

        # Generate ages and art status for each new initiator based on year of initiation
        for h1yy in self.parameters.age_by_h1yy.index.levels[0]:
            grouped_pop = pd.DataFrame()
            n_initiators = n_new_agents.loc[h1yy, 'art_initiators']
            n_delayed = n_new_agents.loc[h1yy, 'art_delayed']
            grouped_pop['age'] = simulate_ages(self.parameters.age_by_h1yy.loc[h1yy], n_initiators + n_delayed)
            grouped_pop['h1yy'] = h1yy
            grouped_pop['status'] = ART_NAIVE
            delayed = np.random.choice(a=len(grouped_pop.index), size=n_delayed, replace=False)
            grouped_pop.loc[delayed, 'status'] = DELAYED
            population = pd.concat([population, grouped_pop])

        population['age'] = self.parameters.sa_type2_dict['initiators_age'] * population['age']  # Sensitivity analysis
        population.loc[population['age'] < 18, 'age'] = 18
        population.loc[population['age'] > 85, 'age'] = 85

        # Generate number of years for delayed initiators to wait before beginning care and modify their start year accordingly
        delayed = population['status'] == DELAYED
        years_out_of_care = np.random.choice(a=self.parameters.years_out_of_care['years'], size=len(population.loc[delayed]), p=self.parameters.years_out_of_care['probability'])
        population.loc[delayed, 'h1yy'] = population.loc[delayed, 'h1yy'] + years_out_of_care
        population.loc[delayed, 'status'] = ART_NAIVE
        population = population[population['h1yy'] <= self.parameters.final_year].copy()

        # Create age_cat variable
        population['age'] = np.floor(population['age'])
        population['age_cat'] = np.floor(population['age'] / 10)
        population.loc[population['age_cat'] < 2, 'age_cat'] = 2
        population.loc[population['age_cat'] > 7, 'age_cat'] = 7

        # Add id number
        population['id'] = np.arange(len(self.population), (len(self.population) + population.index.size))

        # For each h1yy draw values of sqrt_cd4n from a normal truncated at 0 and sqrt 2000
        population = population.set_index('h1yy')
        for h1yy, group in population.groupby(level=0):
            mu = self.parameters.cd4n_by_h1yy.loc[(h1yy, 'mu'), 'estimate']
            sigma = self.parameters.cd4n_by_h1yy.loc[(h1yy, 'sigma'), 'estimate']
            size = group.shape[0]
            sqrt_cd4n = draw_from_trunc_norm(0, np.sqrt(2000.0), mu, sigma, size)
            population.loc[h1yy, 'init_sqrtcd4n'] = sqrt_cd4n
        population = population.reset_index().set_index('id').sort_index()

        # Sensitivity Analysis
        population['init_sqrtcd4n'] = np.sqrt(self.parameters.sa_type2_dict['initiators_cd4'] * np.power(population['init_sqrtcd4n'], 2))

        # Calculate time varying cd4 count and other needed variables
        population['last_h1yy'] = population['h1yy']
        population['time_varying_sqrtcd4n'] = population['init_sqrtcd4n']
        population['last_init_sqrtcd4n'] = population['init_sqrtcd4n']
        population['init_age'] = population['age']
        population['n_lost'] = 0
        population['years_out'] = 0
        population['year_died'] = np.nan
        population['sqrtcd4n_exit'] = 0
        population['ltfu_year'] = 0
        population['return_year'] = 0
        population['intercept'] = 1.0
        population['year'] = 2009

        # If doing an Aim 2 simulation calculate bmi variables for the population
        if self.parameters.comorbidity_flag:
            population['pre_art_bmi'] = calculate_pre_art_bmi(population.copy(), self.parameters.pre_art_bmi_model, self.parameters.pre_art_bmi,
                                                              self.parameters.pre_art_bmi_age_knots, self.parameters.pre_art_bmi_h1yy_knots,
                                                              self.parameters.pre_art_bmi_rse, self.parameters.pre_art_bmi_smearing)
            population['post_art_bmi'] = calculate_post_art_bmi(population.copy(), self.parameters)

            # Apply post_art_bmi intervention
            if self.parameters.bmi_intervention:
                population['post_art_bmi'] = apply_bmi_intervention(population.copy(), self.parameters)

            population['delta_bmi'] = population['post_art_bmi'] - population['pre_art_bmi']

            # Apply comorbidities
            for condition in STAGE0:
                population[condition] = (np.random.rand(len(population.index)) < self.parameters.prev_inits_dict[condition].values).astype(int)
            for condition in STAGE1 + STAGE2 + STAGE3:
                population[condition] = (np.random.rand(len(population.index)) < (self.parameters.prev_inits_dict[condition].values) *
                                         self.parameters.sa_aim2_prev_dict[condition]).astype(int)
            population['mm'] = population[STAGE2 + STAGE3].sum(axis=1)

        # Sort columns alphabetically
        population = population.reindex(sorted(population), axis=1)

        # Record classic one-way sa input
        sa_initial_cd4_inits = population.groupby('h1yy')['init_sqrtcd4n'].agg(mean_cd4=lambda x: (x ** 2).mean(), n='size')
        self.stats.sa_initial_cd4_inits = self.stats.sa_initial_cd4_inits.append(sa_initial_cd4_inits, ignore_index=True)

        # Append new population to pearl population
        self.population = self.population.append(population)

    def run(self):
        """ Simulate from 2010 to final_year """
        while self.year <= self.parameters.final_year:
            # Increment calendar year, ages, age_cat and years out of care
            self.increment_years()

            # Apply comorbidities
            if self.parameters.comorbidity_flag:
                self.apply_comorbidity_incidence()
                self.update_mm()

            # In care operations
            self.increase_cd4_count()  # Increase cd4n in people in care
            self.add_new_user()  # Add in newly diagnosed ART initiators
            self.kill_in_care()  # Kill some people in care
            self.lose_to_follow_up()  # Lose some people to follow up

            # Out of care operations
            self.decrease_cd4_count()  # Decrease cd4n in people out of care
            self.kill_out_care()  # Kill some people out of care
            self.reengage()  # Reengage some people out of care

            # Record output statistics
            self.record_stats()

            # Append changed populations to their respective DataFrames
            self.append_new()

            # Print population breakdown if verbose
            if self.parameters.verbose:
                print(self)

            # Increment year
            self.year += 1

        # Record output statistics for the end of the simulation
        self.record_final_stats()

    def increment_years(self):
        """Increment calendar year for all agents, increment age and age_cat for those alive in the model,
        and increment the number of years spent out of care for the ART non-using population.
        """
        alive_and_initiated = self.population['status'].isin([ART_USER, ART_NONUSER])
        out_care = self.population['status'] == ART_NONUSER
        self.population['year'] = self.year
        self.population.loc[alive_and_initiated, 'age'] += 1
        self.population['age_cat'] = np.floor(self.population['age'] / 10)
        self.population.loc[self.population['age_cat'] < 2, 'age_cat'] = 2
        self.population.loc[self.population['age_cat'] > 7, 'age_cat'] = 7
        self.population.loc[out_care, 'years_out'] += 1

    def increase_cd4_count(self):
        """Calculate and set new CD4 count for ART using population."""
        in_care = self.population['status'] == ART_USER

        new_sqrt_cd4 = calculate_cd4_increase(
            self.population.loc[in_care].copy(),
            self.parameters.cd4_increase_knots,
            self.parameters.cd4_increase.to_numpy(dtype=float),
            self.parameters.cd4_increase_vcov.to_numpy(dtype=float),
            self.parameters.sa_type1_dict['cd4_increase'])

        # Sensitivity Analysis
        new_sqrt_cd4 = np.sqrt(self.parameters.sa_type2_dict['cd4_increase'] * np.power(new_sqrt_cd4, 2))

        old_sqrt_cd4 = self.population.loc[in_care, 'time_varying_sqrtcd4n']
        diff_cd4 = (new_sqrt_cd4**2 - old_sqrt_cd4**2).mean()

        # Record classic one-way sa input
        sa_cd4_increase_in_care = pd.DataFrame(data={'year': self.year,
                                                     'mean_diff': diff_cd4,
                                                     'n': len(old_sqrt_cd4)}, index=[0])
        self.stats.sa_cd4_increase_in_care = self.stats.sa_cd4_increase_in_care.append(sa_cd4_increase_in_care, ignore_index=True)

        self.population.loc[in_care, 'time_varying_sqrtcd4n'] = new_sqrt_cd4

    def decrease_cd4_count(self):
        """Calculate and set new CD4 count for ART non-using population."""
        out_care = self.population['status'] == ART_NONUSER
        new_sqrt_cd4 = calculate_cd4_decrease(
            self.population.loc[out_care].copy(),
            self.parameters.cd4_decrease.to_numpy(dtype=float),
            self.parameters.sa_type1_dict['cd4_decrease'],
            self.parameters.cd4_decrease_vcov.to_numpy(dtype=float),
            self.parameters.cd4_decrease_smearing.to_numpy(dtype=float)[0])

        # Sensitivity Analysis
        new_sqrt_cd4 = np.sqrt(self.parameters.sa_type2_dict['cd4_decrease'] * np.power(new_sqrt_cd4, 2))

        old_sqrt_cd4 = self.population.loc[out_care, 'time_varying_sqrtcd4n']
        diff_cd4 = (new_sqrt_cd4**2 - old_sqrt_cd4**2).mean()

        # Record classic one-way sa input
        sa_cd4_decrease_out_care = pd.DataFrame(data={'year': self.year,
                                                      'mean_diff': diff_cd4,
                                                      'n': len(old_sqrt_cd4)}, index=[0])
        self.stats.sa_cd4_decrease_out_care = self.stats.sa_cd4_decrease_out_care.append(sa_cd4_decrease_out_care, ignore_index=True)

        self.population.loc[out_care, 'time_varying_sqrtcd4n'] = new_sqrt_cd4

    def add_new_user(self):
        """Add newly initiating ART users."""
        new_user = (self.population['status'] == ART_NAIVE) & (self.population['h1yy'] == self.year)
        self.population.loc[new_user, 'status'] = ART_USER

    def kill_in_care(self):
        """Calculate probability of mortality for in care population. Optionally, use the general population mortality threshold
        to increase age category grouped probability of mortality to have the same mean as the general population. Draw random
        numbers to determine who will die.
        """
        # Calculate death probability
        in_care = self.population['status'] == ART_USER
        pop = self.population.copy()
        coeff_matrix = self.parameters.mortality_in_care_co.to_numpy(dtype=float) if self.parameters.comorbidity_flag else self.parameters.mortality_in_care.to_numpy(dtype=float)
        pop_matrix = create_mortality_in_care_pop_matrix(pop.copy(), self.parameters.comorbidity_flag, parameters=self.parameters)
        vcov_matrix = self.parameters.mortality_in_care_vcov.to_numpy(dtype=float)
        pop['death_prob'] = calculate_prob(pop_matrix, coeff_matrix, self.parameters.sa_type1_dict['mortality_in_care'], vcov_matrix)

        # Increase mortality to general population threshold
        if self.parameters.mortality_threshold_flag:
            pop['mortality_age_group'] = pd.cut(pop['age'], bins=[0, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 85], right=True, labels=np.arange(14))
            mean_mortality = pd.DataFrame(pop.loc[in_care].groupby(['mortality_age_group'])['death_prob'].mean())
            mean_mortality['p'] = self.parameters.mortality_threshold['p'] - mean_mortality['death_prob']
            mean_mortality.loc[mean_mortality['p'] <= 0, 'p'] = 0
            for mortality_age_group in np.arange(14):
                excess_mortality = mean_mortality.loc[mortality_age_group, 'p']
                pop.loc[in_care & (pop['mortality_age_group'] == mortality_age_group), 'death_prob'] += excess_mortality

        # Sensitivity Analysis
        pop['death_prob'] = self.parameters.sa_type2_dict['mortality_in_care'] * pop['death_prob']
        pop['death_prob'] = self.parameters.sa_aim2_mort_dict['mortality_in_care'] * pop['death_prob']

        # Record classic one-way sa input
        sa_mortality_in_care_prob = pd.DataFrame(data={'year': self.year,
                                                       'mean_prob': pop.loc[in_care]['death_prob'].mean(),
                                                       'n': len(pop.loc[in_care])}, index=[0])
        self.stats.sa_mortality_in_care_prob = self.stats.sa_mortality_in_care_prob.append(sa_mortality_in_care_prob, ignore_index=True)

        # Draw for mortality
        died = ((pop['death_prob'] > np.random.rand(len(self.population.index))) | (self.population['age'] > 85)) & in_care
        self.population.loc[died, 'status'] = DYING_ART_USER
        self.population.loc[died, 'year_died'] = self.year

    def kill_out_care(self):
        """Calculate probability of mortality for out of care population. Optionally, use the general population mortality threshold
        to increase age category grouped probability of mortality to have the same mean as the general population. Draw random
        numbers to determine who will die.
        """
        # Calculate death probability
        out_care = self.population['status'] == ART_NONUSER
        pop = self.population.copy()
        coeff_matrix = self.parameters.mortality_out_care_co.to_numpy(dtype=float) if self.parameters.comorbidity_flag else self.parameters.mortality_out_care.to_numpy(dtype=float)
        pop_matrix = create_mortality_out_care_pop_matrix(pop.copy(), self.parameters.comorbidity_flag, parameters=self.parameters)
        vcov_matrix = self.parameters.mortality_out_care_vcov.to_numpy(dtype=float)
        pop['death_prob'] = calculate_prob(pop_matrix, coeff_matrix, self.parameters.sa_type1_dict['mortality_out_care'], vcov_matrix)

        # Increase mortality to general population threshold
        if self.parameters.mortality_threshold_flag:
            pop['mortality_age_group'] = pd.cut(pop['age'], bins=[0, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 85], right=True, labels=np.arange(14))
            mean_mortality = pd.DataFrame(pop.loc[out_care].groupby(['mortality_age_group'])['death_prob'].mean())
            mean_mortality['p'] = self.parameters.mortality_threshold['p'] - mean_mortality['death_prob']
            mean_mortality.loc[mean_mortality['p'] <= 0, 'p'] = 0
            for mortality_age_group in np.arange(14):
                excess_mortality = mean_mortality.loc[mortality_age_group, 'p']
                pop.loc[out_care & (pop['mortality_age_group'] == mortality_age_group), 'death_prob'] += excess_mortality

        pop['death_prob'] = self.parameters.sa_type2_dict['mortality_out_care'] * pop['death_prob']
        pop['death_prob'] = self.parameters.sa_aim2_mort_dict['mortality_out_care'] * pop['death_prob']

        # Record classic one-way sa input
        sa_mortality_out_care_prob = pd.DataFrame(data={'year': self.year,
                                                        'mean_prob': pop.loc[out_care]['death_prob'].mean(),
                                                        'n': len(pop.loc[out_care])}, index=[0])
        self.stats.sa_mortality_out_care_prob = self.stats.sa_mortality_out_care_prob.append(sa_mortality_out_care_prob, ignore_index=True)

        # Draw for mortality
        died = ((pop['death_prob'] > np.random.rand(len(self.population.index))) | (self.population['age'] > 85)) & out_care
        self.population.loc[died, 'status'] = DYING_ART_NONUSER
        self.population.loc[died, 'year_died'] = self.year
        self.population.loc[died, 'return_year'] = 0

    def lose_to_follow_up(self):
        """Calculate probability of in care agents leaving care. Draw random number to decide who leaves care. For those
        leaving care, draw the number of years to spend out of care from a normalized, truncated Poisson distribution.
        """
        # Calculate probability and draw
        in_care = self.population['status'] == ART_USER
        pop = self.population.copy()
        coeff_matrix = self.parameters.loss_to_follow_up.to_numpy(dtype=float)
        vcov_matrix = self.parameters.loss_to_follow_up_vcov.to_numpy(dtype=float)
        pop_matrix = create_ltfu_pop_matrix(pop.copy(), self.parameters.ltfu_knots)
        pop['ltfu_prob'] = calculate_prob(pop_matrix, coeff_matrix, self.parameters.sa_type1_dict['loss_to_follow_up'], vcov_matrix)
        pop['ltfu_prob'] = self.parameters.sa_type2_dict['disengagement'] * pop['ltfu_prob']

        lost = (pop['ltfu_prob'] > np.random.rand(len(self.population.index))) & in_care

        # Record classic one-way sa input
        sa_ltfu_prob = pd.DataFrame(data={'year': self.year,
                                          'mean_prob': pop.loc[in_care]['ltfu_prob'].mean(),
                                          'n': len(pop.loc[in_care])}, index=[0])
        self.stats.sa_ltfu_prob = self.stats.sa_ltfu_prob.append(sa_ltfu_prob, ignore_index=True)

        # Draw years spent out of care for those lost
        if self.parameters.sa_type2_dict['reengagement'] == 1.2:
            p = self.parameters.years_out_of_care['prob_1.2']
        elif self.parameters.sa_type2_dict['reengagement'] == 1.1:
            p = self.parameters.years_out_of_care['prob_1.1']
        elif self.parameters.sa_type2_dict['reengagement'] == 0.9:
            p = self.parameters.years_out_of_care['prob_0.9']
        elif self.parameters.sa_type2_dict['reengagement'] == 0.8:
            p = self.parameters.years_out_of_care['prob_0.8']
        else:
            p = self.parameters.years_out_of_care['probability']

        years_out_of_care = np.random.choice(a=self.parameters.years_out_of_care['years'], size=len(self.population.loc[lost]), p=p)

        sa_years_out_input = pd.DataFrame(data={'year': self.year,
                                                'mean_years': years_out_of_care.mean(),
                                                'n': len(years_out_of_care)}, index=[0])
        self.stats.sa_years_out_input = self.stats.sa_years_out_input.append(sa_years_out_input, ignore_index=True)

        # Set variables for lost population
        self.population.loc[lost, 'return_year'] = self.year + years_out_of_care
        self.population.loc[lost, 'status'] = LTFU
        self.population.loc[lost, 'sqrtcd4n_exit'] = self.population.loc[lost, 'time_varying_sqrtcd4n']
        self.population.loc[lost, 'ltfu_year'] = self.year
        self.population.loc[lost, 'n_lost'] += 1

    def reengage(self):
        """Move out of care population scheduled to reenter care."""
        out_care = self.population['status'] == ART_NONUSER
        reengaged = (self.year == self.population['return_year']) & out_care
        self.population.loc[reengaged, 'status'] = REENGAGED

        # Set new initial sqrtcd4n to current time varying cd4n and h1yy to current year
        self.population.loc[reengaged, 'last_init_sqrtcd4n'] = self.population.loc[reengaged, 'time_varying_sqrtcd4n']
        self.population.loc[reengaged, 'last_h1yy'] = self.year
        self.population.loc[reengaged, 'return_year'] = 0

        # Save years out of care
        years_out = (pd.DataFrame(self.population.loc[reengaged, 'years_out'].value_counts())
                     .reindex(range(1, 16), fill_value=0).reset_index()
                     .rename(columns={'index': 'years_out', 'years_out': 'n'})
                     .assign(year=self.year))[['year', 'years_out', 'n']]

        self.stats.years_out = self.stats.years_out.append(years_out)
        self.population.loc[reengaged, 'years_out'] = 0

    def append_new(self):
        """Move agents from the temporary, statuses to the main statuses at the end of the year."""
        reengaged = self.population['status'] == REENGAGED
        ltfu = self.population['status'] == LTFU
        dying_art_user = self.population['status'] == DYING_ART_USER
        dying_art_nonuser = self.population['status'] == DYING_ART_NONUSER

        self.population.loc[reengaged, 'status'] = ART_USER
        self.population.loc[ltfu, 'status'] = ART_NONUSER
        self.population.loc[dying_art_user, 'status'] = DEAD_ART_USER
        self.population.loc[dying_art_nonuser, 'status'] = DEAD_ART_NONUSER

    def apply_comorbidity_incidence(self):
        """Calculate probability of incidence of all comorbidities and then draw to determine which agents
        experience incidence. Record incidence data stratified by care status and age category.
        """
        in_care = self.population['status'] == ART_USER
        out_care = self.population['status'] == ART_NONUSER

        # Iterate over all comorbidities
        for condition in STAGE1 + STAGE2 + STAGE3:
            # Calculate probability
            coeff_matrix = self.parameters.comorbidity_coeff_dict[condition].to_numpy(dtype=float)
            pop_matrix = create_comorbidity_pop_matrix(self.population.copy(), condition=condition, parameters=self.parameters)
            prob = calculate_prob(pop_matrix, coeff_matrix, None, None)
            prob = self.parameters.sa_aim2_inc_dict[condition] * prob

            # Draw for incidence
            rand = prob > np.random.rand(len(self.population.index))
            old = self.population[condition]
            new = rand & (in_care | out_care) & ~old
            self.population[condition] = (old | new).astype(int)

            # Save incidence statistics
            incidence_in_care = (self.population.loc[new & in_care].groupby(['age_cat']).size()
                                 .reindex(index=self.parameters.AGE_CATS, fill_value=0).reset_index(name='n')
                                 .assign(year=self.year, condition=condition))[['condition', 'year', 'age_cat', 'n']]
            self.stats.incidence_in_care = self.stats.incidence_in_care.append(incidence_in_care)

            incidence_out_care = (self.population.loc[new & out_care].groupby(['age_cat']).size()
                                  .reindex(index=self.parameters.AGE_CATS, fill_value=0).reset_index(name='n')
                                  .assign(year=self.year, condition=condition))[['condition', 'year', 'age_cat', 'n']]
            self.stats.incidence_out_care = self.stats.incidence_out_care.append(incidence_out_care)

    def update_mm(self):
        """Calculate and update the multimorbidity, defined as the number of stage 2 and 3 comorbidities in each agent."""
        self.population['mm'] = self.population[STAGE2 + STAGE3].sum(axis=1)

    def record_stats(self):
        """"Record in care age breakdown, out of care age breakdown, reengaging pop age breakdown, leaving care age breakdown, and CD4
        statistics for both in and out of care populations. If it is an Aim 2 simulation, record the prevalence of all comorbidities, and the
        multimorbidity for the in care, out of care, initiating, and dying populations. Record the detailed comorbidity information if the multimorbidity detail flag is set.
        """
        stay_in_care = self.population['status'] == ART_USER
        stay_out_care = self.population['status'] == ART_NONUSER
        reengaged = self.population['status'] == REENGAGED
        ltfu = self.population['status'] == LTFU
        dying_art_user = self.population['status'] == DYING_ART_USER
        dying_art_nonuser = self.population['status'] == DYING_ART_NONUSER
        dying = dying_art_nonuser | dying_art_user
        in_care = stay_in_care | ltfu | dying_art_user
        out_care = stay_out_care | reengaged | dying_art_nonuser
        initiating = self.population['h1yy'] == self.year

        # Count of those in care by age and year
        in_care_age = (self.population.loc[in_care].groupby(['age']).size()
                       .reindex(index=self.parameters.AGES, fill_value=0).reset_index(name='n')
                       .assign(year=self.year))[['year', 'age', 'n']]
        self.stats.in_care_age = self.stats.in_care_age.append(in_care_age)

        # Count of those in care by age and year
        out_care_age = (self.population.loc[out_care].groupby(['age']).size()
                        .reindex(index=self.parameters.AGES, fill_value=0).reset_index(name='n')
                        .assign(year=self.year))[['year', 'age', 'n']]
        self.stats.out_care_age = self.stats.out_care_age.append(out_care_age)

        # Count of those reengaging in care by age and year
        reengaged_age = (self.population.loc[reengaged].groupby(['age']).size()
                         .reindex(index=self.parameters.AGES, fill_value=0).reset_index(name='n')
                         .assign(year=self.year))[['year', 'age', 'n']]
        self.stats.reengaged_age = self.stats.reengaged_age.append(reengaged_age)

        # Count of those lost to care by age and year
        ltfu_age = (self.population.loc[ltfu].groupby(['age']).size()
                    .reindex(index=self.parameters.AGES, fill_value=0).reset_index(name='n')
                    .assign(year=self.year))[['year', 'age', 'n']]
        self.stats.ltfu_age = self.stats.ltfu_age.append(ltfu_age)

        # Discretize cd4 count and count those in care
        cd4_in_care = pd.DataFrame(np.power(self.population.loc[in_care, 'time_varying_sqrtcd4n'], 2).round(0).astype(int)).rename(columns={'time_varying_sqrtcd4n': 'cd4_count'})
        cd4_in_care = cd4_in_care.groupby('cd4_count').size()
        cd4_in_care = cd4_in_care.reset_index(name='n').assign(year=self.year)[['year', 'cd4_count', 'n']]
        self.stats.cd4_in_care = self.stats.cd4_in_care.append(cd4_in_care)

        # Discretize cd4 count and count those out care
        cd4_out_care = pd.DataFrame(np.power(self.population.loc[out_care, 'time_varying_sqrtcd4n'], 2).round(0).astype(int)).rename(columns={'time_varying_sqrtcd4n': 'cd4_count'})
        cd4_out_care = cd4_out_care.groupby('cd4_count').size()
        cd4_out_care = cd4_out_care.reset_index(name='n').assign(year=self.year)[['year', 'cd4_count', 'n']]
        self.stats.cd4_out_care = self.stats.cd4_out_care.append(cd4_out_care)

        if self.parameters.comorbidity_flag:
            for condition in STAGE0 + STAGE1 + STAGE2 + STAGE3:
                has_condition = self.population[condition] == 1

                # Record prevalence for in care, out of care, initiator, and dead populations
                prevalence_in_care = (self.population.loc[in_care & has_condition].groupby(['age_cat']).size()
                                      .reindex(index=self.parameters.AGE_CATS, fill_value=0).reset_index(name='n')
                                      .assign(year=self.year, condition=condition))[['condition', 'year', 'age_cat', 'n']]
                self.stats.prevalence_in_care = self.stats.prevalence_in_care.append(prevalence_in_care)
                prevalence_out_care = (self.population.loc[out_care & has_condition].groupby(['age_cat']).size()
                                       .reindex(index=self.parameters.AGE_CATS, fill_value=0).reset_index(name='n')
                                       .assign(year=self.year, condition=condition))[['condition', 'year', 'age_cat', 'n']]
                self.stats.prevalence_out_care = self.stats.prevalence_out_care.append(prevalence_out_care)
                prevalence_inits = (self.population.loc[initiating & has_condition].groupby(['age_cat']).size()
                                    .reindex(index=self.parameters.AGE_CATS, fill_value=0).reset_index(name='n')
                                    .assign(year=self.year, condition=condition))[['condition', 'year', 'age_cat', 'n']]
                self.stats.prevalence_inits = self.stats.prevalence_inits.append(prevalence_inits)
                prevalence_dead = (self.population.loc[dying & has_condition].groupby(['age_cat']).size()
                                   .reindex(index=self.parameters.AGE_CATS, fill_value=0).reset_index(name='n')
                                   .assign(year=self.year, condition=condition))[['condition', 'year', 'age_cat', 'n']]
                self.stats.prevalence_dead = self.stats.prevalence_dead.append(prevalence_dead)

            # Record the multimorbidity information for the in care, out of care, initiating, and dead populations
            mm_in_care = (self.population.loc[in_care].groupby(['age_cat', 'mm']).size()
                          .reindex(index=pd.MultiIndex.from_product([self.parameters.AGE_CATS, np.arange(0, 8)], names=['age_cat', 'mm']), fill_value=0)
                          .reset_index(name='n').assign(year=self.year))[['year', 'age_cat', 'mm', 'n']]
            self.stats.mm_in_care = self.stats.mm_in_care.append(mm_in_care)
            mm_out_care = (self.population.loc[out_care].groupby(['age_cat', 'mm']).size()
                           .reindex(index=pd.MultiIndex.from_product([self.parameters.AGE_CATS, np.arange(0, 8)], names=['age_cat', 'mm']), fill_value=0)
                           .reset_index(name='n').assign(year=self.year))[['year', 'age_cat', 'mm', 'n']]
            self.stats.mm_out_care = self.stats.mm_out_care.append(mm_out_care)
            mm_inits = (self.population.loc[initiating].groupby(['age_cat', 'mm']).size()
                        .reindex(index=pd.MultiIndex.from_product([self.parameters.AGE_CATS, np.arange(0, 8)], names=['age_cat', 'mm']), fill_value=0)
                        .reset_index(name='n').assign(year=self.year))[['year', 'age_cat', 'mm', 'n']]
            self.stats.mm_inits = self.stats.mm_inits.append(mm_inits)
            mm_dead = (self.population.loc[dying].groupby(['age_cat', 'mm']).size()
                       .reindex(index=pd.MultiIndex.from_product([self.parameters.AGE_CATS, np.arange(0, 8)], names=['age_cat', 'mm']), fill_value=0)
                       .reset_index(name='n').assign(year=self.year))[['year', 'age_cat', 'mm', 'n']]
            self.stats.mm_dead = self.stats.mm_dead.append(mm_dead)

            # Record the detailed comorbidity information
            mm_detail_in_care = create_mm_detail_stats(self.population.loc[in_care].copy())
            mm_detail_in_care = mm_detail_in_care.assign(year=self.year)[['year', 'multimorbidity', 'n']]
            self.stats.mm_detail_in_care = self.stats.mm_detail_in_care.append(mm_detail_in_care)

            mm_detail_out_care = create_mm_detail_stats(self.population.loc[out_care].copy())
            mm_detail_out_care = mm_detail_out_care.assign(year=self.year)[['year', 'multimorbidity', 'n']]
            self.stats.mm_detail_out_care = self.stats.mm_detail_out_care.append(mm_detail_out_care)

            mm_detail_inits = create_mm_detail_stats(self.population.loc[initiating].copy())
            mm_detail_inits = mm_detail_inits.assign(year=self.year)[['year', 'multimorbidity', 'n']]
            self.stats.mm_detail_inits = self.stats.mm_detail_inits.append(mm_detail_inits)

            mm_detail_dead = create_mm_detail_stats(self.population.loc[dying].copy())
            mm_detail_dead = mm_detail_dead.assign(year=self.year)[['year', 'multimorbidity', 'n']]
            self.stats.mm_detail_dead = self.stats.mm_detail_dead.append(mm_detail_dead)

    def record_final_stats(self):
        """Record some stats that are better calculated at the end of the simulation. A count of new initiators, those dying in care, and
        those dying out of care is recorded as well as the cd4 count of ART initiators.
        """
        dead_in_care = self.population['status'] == DEAD_ART_USER
        dead_out_care = self.population['status'] == DEAD_ART_NONUSER
        new_inits = self.population['h1yy'] >= 2010

        # Count of new initiators by year and age
        new_init_age = self.population.loc[new_inits].groupby(['h1yy', 'init_age']).size()
        new_init_age = new_init_age.reindex(pd.MultiIndex.from_product([self.parameters.SIMULATION_YEARS, self.parameters.AGES], names=['year', 'age']), fill_value=0)
        self.stats.new_init_age = new_init_age.reset_index(name='n')

        # Count of those that died in care by age and year
        dead_in_care_age = self.population.loc[dead_in_care].groupby(['year_died', 'age']).size()
        dead_in_care_age = dead_in_care_age.reindex(pd.MultiIndex.from_product([self.parameters.SIMULATION_YEARS, self.parameters.AGES], names=['year', 'age']), fill_value=0)
        self.stats.dead_in_care_age = dead_in_care_age.reset_index(name='n')

        # Count of those that died out of care by age and year
        dead_out_care_age = self.population.loc[dead_out_care].groupby(['year_died', 'age']).size()
        dead_out_care_age = dead_out_care_age.reindex(pd.MultiIndex.from_product([self.parameters.SIMULATION_YEARS, self.parameters.AGES], names=['year', 'age']), fill_value=0)
        self.stats.dead_out_care_age = dead_out_care_age.reset_index(name='n')

        # Count of discretized cd4 count at ART initiation
        cd4_inits = self.population[['init_sqrtcd4n', 'h1yy']].copy()
        cd4_inits['cd4_count'] = np.power(cd4_inits['init_sqrtcd4n'], 2).round(0).astype(int)
        cd4_inits = cd4_inits.groupby(['h1yy', 'cd4_count']).size()
        self.stats.cd4_inits = cd4_inits.reset_index(name='n').rename(columns={'h1yy': 'year'})

        if self.parameters.comorbidity_flag:
            pre_art_bmi = self.population[['pre_art_bmi', 'h1yy']].round(0).astype(int)
            pre_art_bmi = pre_art_bmi.groupby(['h1yy', 'pre_art_bmi']).size()
            self.stats.pre_art_bmi = pre_art_bmi.reset_index(name='n').rename(columns={'h1yy': 'year'})

            post_art_bmi = self.population[['post_art_bmi', 'h1yy']].round(0).astype(int)
            post_art_bmi = post_art_bmi.groupby(['h1yy', 'post_art_bmi']).size()
            self.stats.post_art_bmi = post_art_bmi.reset_index(name='n').rename(columns={'h1yy': 'year'})


###############################################################################
# Parameter and Statistics Classes                                            #
###############################################################################

class Parameters:
    """This class holds all the parameters needed for PEARL to run."""
    def __init__(self, path, rerun_folder, output_folder, group_name, comorbidity_flag, new_dx, final_year,
                 mortality_model, mortality_threshold_flag, idu_threshold, verbose, sa_type=None, sa_variable=None,
                 sa_value=None, bmi_intervention=False, bmi_intervention_probability=1.0):
        """Takes the path to the parameters.h5 file, the path to the folder containing rerun data if the run is a rerun,
        the output folder, the group name, a flag indicating if the simulation is for aim 2, a flag indicating whether to
        record detailed comorbidity information, the type of new_dx parameter to use, the final year of the model, the
        mortality model to use, whether to use a mortality threshold, verbosity, the sensitivity analysis dict, the classic
        sensitivity analysis dict, and the aim 2 sensitivity analysis dict.
        """

        # Save inputs as class attributes
        self.rerun_folder = rerun_folder
        self.output_folder = output_folder
        self.comorbidity_flag = comorbidity_flag
        self.final_year = final_year
        self.mortality_threshold_flag = mortality_threshold_flag
        self.verbose = verbose
        self.sa_type = sa_type
        self.sa_variable = sa_variable
        self.sa_value = sa_value

        # 2009 population
        self.on_art_2009 = pd.read_hdf(path, 'on_art_2009').loc[group_name]
        self.age_in_2009 = pd.read_hdf(path, 'age_in_2009').loc[group_name]
        self.h1yy_by_age_2009 = pd.read_hdf(path, 'h1yy_by_age_2009').loc[group_name]
        self.cd4n_by_h1yy_2009 = pd.read_hdf(path, 'cd4n_by_h1yy_2009').loc[group_name]

        # New initiator statistics
        self.linkage_to_care = pd.read_hdf(path, 'linkage_to_care').loc[group_name]
        self.age_by_h1yy = pd.read_hdf(path, 'age_by_h1yy').loc[group_name]
        self.cd4n_by_h1yy = pd.read_hdf(path, 'cd4n_by_h1yy').loc[group_name]

        # Choose new ART initiator model
        if new_dx == 'base':
            self.new_dx = pd.read_hdf(path, 'new_dx').loc[group_name]
        elif new_dx == 'ehe':
            self.new_dx = pd.read_hdf(path, 'new_dx_ehe').loc[group_name]
        elif new_dx == 'sa':
            self.new_dx = pd.read_hdf(path, 'new_dx_sa').loc[group_name]
        else:
            raise ValueError('Invalid new diagnosis file specified')

        # Choose mortality model
        if mortality_model == 'by_sex_race_risk':
            mortality_model_str = ''
        else:
            mortality_model_str = '_' + mortality_model
            if sa_type is not None:
                raise NotImplementedError('Using alternative mortality models with sensitivity analysis is not implemented')

        if (mortality_model != 'by_sex_race_risk') & (mortality_model != 'by_sex_race_risk_2015'):
                if idu_threshold != '2x':
                    raise NotImplementedError('Using alternative mortality models with idu threshold changes is not implemented')

        # Mortality In Care
        self.mortality_in_care = pd.read_hdf(path, f'mortality_in_care{mortality_model_str}').loc[group_name]
        self.mortality_in_care_age = pd.read_hdf(path, f'mortality_in_care_age{mortality_model_str}').loc[group_name]
        self.mortality_in_care_sqrtcd4 = pd.read_hdf(path, f'mortality_in_care_sqrtcd4{mortality_model_str}').loc[group_name]
        self.mortality_in_care_vcov = pd.read_hdf(path, 'mortality_in_care_vcov').loc[group_name]

        # Mortality Out Of Care
        self.mortality_out_care = pd.read_hdf(path, f'mortality_out_care{mortality_model_str}').loc[group_name]
        self.mortality_out_care_age = pd.read_hdf(path, f'mortality_out_care_age{mortality_model_str}').loc[group_name]
        self.mortality_out_care_tv_sqrtcd4 = pd.read_hdf(path, f'mortality_out_care_tv_sqrtcd4{mortality_model_str}').loc[group_name]
        self.mortality_out_care_vcov = pd.read_hdf(path, 'mortality_out_care_vcov').loc[group_name]

        # Mortality Threshold
        if idu_threshold != '2x':
            self.mortality_threshold = pd.read_hdf(path, f'mortality_threshold_idu_{idu_threshold}').loc[group_name]
        else:
            self.mortality_threshold = pd.read_hdf(path, f'mortality_threshold{mortality_model_str}').loc[group_name]

        # Loss To Follow Up
        self.loss_to_follow_up = pd.read_hdf(path, 'loss_to_follow_up').loc[group_name]
        self.ltfu_knots = pd.read_hdf(path, 'ltfu_knots').loc[group_name]
        self.loss_to_follow_up_vcov = pd.read_hdf(path, 'loss_to_follow_up_vcov').loc[group_name]

        # Cd4 Increase
        self.cd4_increase = pd.read_hdf(path, 'cd4_increase').loc[group_name]
        self.cd4_increase_knots = pd.read_hdf(path, 'cd4_increase_knots').loc[group_name]
        self.cd4_increase_vcov = pd.read_hdf(path, 'cd4_increase_vcov').loc[group_name]

        # Cd4 Decrease
        self.cd4_decrease = pd.read_hdf(path, 'cd4_decrease').loc['all']
        self.cd4_decrease_smearing = pd.read_hdf(path, 'cd4_decrease_smearing').loc['all']
        self.cd4_decrease_vcov = pd.read_hdf(path, 'cd4_decrease_vcov')

        # Years out of Care
        self.years_out_of_care = pd.read_hdf(path, 'years_out_of_care')

        # Set up sensitivity analysis
        self.sa_type1_dict = sa_type1_default_dict.copy()
        self.sa_type2_dict = sa_type2_default_dict.copy()
        self.sa_aim2_inc_dict = sa_aim2_default_dict.copy()
        self.sa_aim2_prev_dict = sa_aim2_default_dict.copy()
        self.sa_aim2_mort_dict = sa_aim2_mort_default_dict.copy()

        if sa_type == 'type1':
            self.sa_type1_dict[sa_variable] = sa_value
            if sa_variable in ['lambda1', 'mu1', 'mu2', 'sigma1', 'sigma2']:
                self.age_in_2009.loc[sa_variable, 'estimate'] = self.age_in_2009.loc[sa_variable, f'conf_{sa_value}']
            elif sa_variable == 'new_pop_size':
                if sa_value == 'low':
                    self.new_dx['upper'] = self.new_dx['lower']
                elif sa_value == 'high':
                    self.new_dx['lower'] = self.new_dx['upper']
        elif sa_type == 'type2':
            self.sa_type2_dict[sa_variable] = sa_value
        elif sa_type == 'aim2_inc':
            self.sa_aim2_inc_dict[sa_variable] = sa_value
        elif sa_type == 'aim2_prev':
            self.sa_aim2_prev_dict[sa_variable] = sa_value
        elif sa_type == 'aim2_mort':
            self.sa_aim2_mort_dict[sa_variable] = sa_value

        # BMI
        self.pre_art_bmi = pd.read_hdf(path, 'pre_art_bmi').loc[group_name]
        self.pre_art_bmi_model = pd.read_hdf(path, 'pre_art_bmi_model').loc[group_name].values[0]
        self.pre_art_bmi_age_knots = pd.read_hdf(path, 'pre_art_bmi_age_knots').loc[group_name]
        self.pre_art_bmi_h1yy_knots = pd.read_hdf(path, 'pre_art_bmi_h1yy_knots').loc[group_name]
        self.pre_art_bmi_rse = pd.read_hdf(path, 'pre_art_bmi_rse').loc[group_name].values[0]
        self.pre_art_bmi_smearing = pd.read_hdf(path, 'pre_art_bmi_smearing').loc[group_name].values[0]
        self.post_art_bmi = pd.read_hdf(path, 'post_art_bmi').loc[group_name]
        self.post_art_bmi_age_knots = pd.read_hdf(path, 'post_art_bmi_age_knots').loc[group_name]
        self.post_art_bmi_pre_art_bmi_knots = pd.read_hdf(path, 'post_art_bmi_pre_art_bmi_knots').loc[group_name]
        self.post_art_bmi_cd4_knots = pd.read_hdf(path, 'post_art_bmi_cd4_knots').loc[group_name]
        self.post_art_bmi_cd4_post_knots = pd.read_hdf(path, 'post_art_bmi_cd4_post_knots').loc[group_name]
        self.post_art_bmi_rse = pd.read_hdf(path, 'post_art_bmi_rse').loc[group_name].values[0]

        # BMI Intervention Probability
        self.bmi_intervention = bmi_intervention
        self.bmi_intervention_probability = bmi_intervention_probability

        # Comorbidities
        self.prev_users_dict = {comorbidity: pd.read_hdf(path, f'{comorbidity}_prev_users').loc[group_name] for comorbidity in STAGE0 + STAGE1 + STAGE2 + STAGE3}
        self.prev_inits_dict = {comorbidity: pd.read_hdf(path, f'{comorbidity}_prev_inits').loc[group_name] for comorbidity in STAGE0 + STAGE1 + STAGE2 + STAGE3}
        self.comorbidity_coeff_dict = {comorbidity: pd.read_hdf(path, f'{comorbidity}_coeff').loc[group_name] for comorbidity in STAGE1 + STAGE2 + STAGE3}
        self.delta_bmi_dict = {comorbidity: pd.read_hdf(path, f'{comorbidity}_delta_bmi').loc[group_name] for comorbidity in STAGE2 + STAGE3}
        self.post_art_bmi_dict = {comorbidity: pd.read_hdf(path, f'{comorbidity}_post_art_bmi').loc[group_name] for comorbidity in STAGE2 + STAGE3}

        # Aim 2 Mortality
        self.mortality_in_care_co = pd.read_hdf(path, 'mortality_in_care_co').loc[group_name]
        self.mortality_in_care_post_art_bmi = pd.read_hdf(path, 'mortality_in_care_post_art_bmi').loc[group_name]
        self.mortality_out_care_co = pd.read_hdf(path, 'mortality_out_care_co').loc[group_name]
        self.mortality_out_care_post_art_bmi = pd.read_hdf(path, 'mortality_out_care_post_art_bmi').loc[group_name]

        # Year and age ranges
        self.AGES = np.arange(18, 87)
        self.AGE_CATS = np.arange(2, 8)
        self.SIMULATION_YEARS = np.arange(2010, final_year+1)
        self.ALL_YEARS = np.arange(2000, final_year+1)
        self.INITIAL_YEARS = np.arange(2000, 2010)
        self.CD4_BINS = np.arange(2001)


class Statistics:
    """A class housing the output from a PEARL run."""
    def __init__(self, output_folder, comorbidity_flag=None, sa_type=None):
        """The init function operates on two levels. If called with no out_list a new Statistics class is initialized, with empty dataframes to fill with data.
        Otherwise it concatenates the out_list dataframes so that the results of all replications and groups are stored in a single dataframe.
        """

        self.output_folder = output_folder

        self.in_care_age = pd.DataFrame()
        self.out_care_age = pd.DataFrame()
        self.reengaged_age = pd.DataFrame()
        self.ltfu_age = pd.DataFrame()
        self.dead_in_care_age = pd.DataFrame()
        self.dead_out_care_age = pd.DataFrame()
        self.new_init_age = pd.DataFrame()
        self.years_out = pd.DataFrame()
        self.cd4_inits = pd.DataFrame()
        self.cd4_in_care = pd.DataFrame()
        self.cd4_out_care = pd.DataFrame()
        self.art_coeffs = pd.DataFrame()
        if comorbidity_flag:
            self.incidence_in_care = pd.DataFrame()
            self.incidence_out_care = pd.DataFrame()
            self.prevalence_in_care = pd.DataFrame()
            self.prevalence_out_care = pd.DataFrame()
            self.prevalence_inits = pd.DataFrame()
            self.prevalence_dead = pd.DataFrame()
            self.mm_in_care = pd.DataFrame()
            self.mm_out_care = pd.DataFrame()
            self.mm_inits = pd.DataFrame()
            self.mm_dead = pd.DataFrame()
            self.mm_detail_in_care = pd.DataFrame()
            self.mm_detail_out_care = pd.DataFrame()
            self.mm_detail_inits = pd.DataFrame()
            self.mm_detail_dead = pd.DataFrame()
            self.pre_art_bmi = pd.DataFrame()
            self.post_art_bmi = pd.DataFrame()

        # Type2 sensitivity analysis output
        self.sa_initial_cd4_in_care = pd.DataFrame()
        self.sa_initial_cd4_out_care = pd.DataFrame()
        self.sa_initial_cd4_inits = pd.DataFrame()
        self.sa_mortality_in_care_prob = pd.DataFrame()
        self.sa_mortality_out_care_prob = pd.DataFrame()
        self.sa_ltfu_prob = pd.DataFrame()
        self.sa_years_out_input = pd.DataFrame()
        self.sa_cd4_increase_in_care = pd.DataFrame()
        self.sa_cd4_decrease_out_care = pd.DataFrame()

    def save(self):
        """Save all internal dataframes as csv files."""
        for name, item in self.__dict__.items():
            if isinstance(item, pd.DataFrame):
                item.to_csv(self.output_folder/f'{name}.csv', index=False)
