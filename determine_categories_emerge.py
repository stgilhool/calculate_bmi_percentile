import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from plotnine import *
import pdb
import argparse
import math
import pdb


# make cag-to-rhythm lookup table
#lkdf = pd.read_csv('/home/gilhools/Rhythm_project/data/batch2_pool_lookup.txt', header=None, names=['subject_id', 'rhythm_id'], sep=' ')
#lkdf = pd.read_csv('/mnt/isilon/cag/ngs/hiseq/gilhools/Rhythm_project/data/lookup13000.csv', header=0)
### Set up paths
current_path = os.path.realpath(os.path.curdir)
root_path = os.path.dirname(current_path)
if os.path.basename(root_path) != 'eMERGE_project':
    raise OSError("Root path should be 'eMERGE_project'")

data_path = '/'.join([root_path, 'data'])
code_path = '/'.join([root_path, 'dev'])
out_path = '/'.join([root_path, 'out'])
print(root_path)
print(data_path)
print(code_path)
print(out_path)

bmi_table_cdc_filename = 'bmi_table_cdc.csv'
bmi_table_boys_who_filename = 'bmi_table_boys_who.txt'
bmi_table_girls_who_filename = 'bmi_table_girls_who.txt'

#Read in the CDC percentile data
pctl_table_cdc = pd.read_csv('/'.join([data_path, bmi_table_cdc_filename]), header=0)
cdc_pctl_boys = pctl_table_cdc.loc[pctl_table_cdc.Sex == 1]
cdc_pctl_girls = pctl_table_cdc.loc[pctl_table_cdc.Sex == 2]

who_pctl_boys = pd.read_csv('/'.join([data_path, bmi_table_boys_who_filename]),
                            sep='\t',
                            header=0)
who_pctl_boys.rename({'Month':'Agemos'},axis=1,inplace=True)
who_pctl_girls = pd.read_csv('/'.join([data_path, bmi_table_girls_who_filename]),
                            sep='\t',
                            header=0)
who_pctl_girls.rename({'Month':'Agemos'},axis=1,inplace=True)

print(who_pctl_boys.columns)
print(who_pctl_boys.shape)
print(cdc_pctl_boys.columns)
print(cdc_pctl_boys.shape)
#Go through bmi table and do the following:
'''
Classify the bmi as 97th, 98th, 120*95th, 140*95th
-Not sure if should make a 1,1,0,0; 1,0,0,0; 1,1,1,1; etc OR classify into one cat for each measurement

Create new table (groupby patient id)
-For each patient, count the number of >97, >98, >120, >140
-Create category column
'''

def zptile(z_score):
    return .5 * (math.erf(z_score / 2. ** .5) + 1)

def calc_z_score(bmi, L, M, S):
    '''
    Calculate z-score given bmi and L, M, S parameters (scalar or vector)
    '''
    return (((bmi/M)**L) - 1.)/(L*S)


def calc_percentile(row, overwrite=True):
    '''
    Calculate the percentile of the bmi for a given visit
    '''
    age_days = row.visit_agedays
    sex = row.gender
    
    age_months = (age_days/365.25) * 12.
    
    bmi = row.bmi

    if age_months >= 24:
        table_type = 'cdc'
        if sex == 'M':
            use_table = cdc_pctl_boys
        elif sex == 'F':
            use_table = cdc_pctl_girls
        else:
            return np.nan
        #raise(ValueError)
    else:
        table_type = 'who'
        if sex == 'M':
            use_table = who_pctl_boys
        elif sex == 'F':
            use_table = who_pctl_girls
        else:
            return np.nan
            #raise(ValueError)
    age_table = use_table.Agemos.values
    L_arr = use_table.L.values
    M_arr = use_table.M.values
    S_arr = use_table.S.values
    
    Z_arr = calc_z_score(bmi, L_arr, M_arr, S_arr)
    z_score = np.interp(age_months, age_table, Z_arr)
    percentile = zptile(z_score)

    if table_type == 'cdc':
        out = [percentile, np.nan]
    elif table_type == 'who':
        out = [np.nan, percentile]
    else:
        raise ValueError("table_type must be cdc or who")

    return out
 
def calc_95th_percentile(row):
    ''' 
    Calculate the 95th percentile bmi for a given visit
    '''
    age_days = row.visit_agedays
    sex = row.gender
    
    age_months = (age_days/365.25) * 12.
    if age_months >= 24:
        if sex == 'M':
            use_table = cdc_pctl_boys
        elif sex == 'F':
            use_table = cdc_pctl_girls
        else:
            return np.nan
        #raise(ValueError)
    else:
        if sex == 'M':
            use_table = who_pctl_boys
        elif sex == 'F':
            use_table = who_pctl_girls
        else:
            return np.nan
            #raise(ValueError)
    age_table = use_table.Agemos.values
    p95_table = use_table.P95.values

    return np.interp(age_months, age_table, p95_table)

    
def categorize_indiv_bmi(row):

    if row.outlier:
        bmi_pctl_max = np.nan
    else:
        bmi_pctl_max = np.nanmax(np.array([row.bmi_pctl_cdc, row.bmi_pctl_who]))
    
    if bmi_pctl_max == np.nan:
        return [0,0,0,0,0]
    elif row.bmi >= (1.4 * row.P95):
        return [1,1,1,1,1]
    elif row.bmi >= (1.2 * row.P95):
        return [1,1,1,1,0]
    elif bmi_pctl_max >= 0.98:
        return [1,1,1,0,0]
    elif bmi_pctl_max >= 0.97:
        return [1,1,0,0,0]
    elif bmi_pctl_max >= 0.95:
        return [1,0,0,0,0]
    else:
        return [0,0,0,0,0]


def categorize_indiv_patient(group):
    '''
    Categorize patients by the criteria of 2 bmis at 97th or above
    '''

    if 'outlier' in group.columns:
        counts_arr = group.loc[~group.outlier,['N95','N97','N98','N120','N140']].values
    else:
        counts_arr = group.loc[:,['N95','N97','N98','N120','N140']].values

    counts = counts_arr.sum(axis=0)

    if counts[0] < 2:
        return "none"
    elif counts[1] < 2:
        return "none"
    elif counts[2] == 0:
        return "97"
    elif counts[3] == 0:
        return "98"
    elif counts[4] > 0:
        return "120_95"
    elif counts[4] > 0:
        return "140_95"
    else:
        print("WARNING: Category not determined correctly!")
        return ""


def categorize_indiv_patient2(group):
    '''
    Categorize patients by the criteria of 1 95th percentile measure
    and 1 of 97th or above
    '''

    if 'outlier' in group.columns:
        counts_arr = group.loc[~group.outlier,['N95','N97','N98','N120','N140']].values
    else:
        counts_arr = group.loc[:,['N95','N97','N98','N120','N140']].values

    counts = counts_arr.sum(axis=0)

    if counts[0] < 2:
        return "none"
    elif counts[1] == 0:
        return "none"
    elif counts[2] == 0:
        return "97"
    elif counts[3] == 0:
        return "98"
    elif counts[4] == 0:
        return "120_95"
    elif counts[4] > 0:
        return "140_95"
    else:
        print("WARNING: Category not determined correctly!")
        return ""

if __name__ == "__main__":

    debug = True
    categorization = '95'
    calc_pctl = True
    ###FIXME: hardcode desired input file
    bmi_data = pd.read_csv('/'.join([data_path,'epic_refresh_20200918','epic_refresh_bmi_cleaned.csv']),header=0)
    #bmi_data = pd.read_csv('/'.join([out_path, 'new_patients_20200211_chunks', 'bmi_cleaned_cag.csv']),header=0)
    bmi_data['P95'] = bmi_data.apply(calc_95th_percentile, axis=1)
    
    if calc_pctl:
        bmi_data[['bmi_pctl_cdc','bmi_pctl_who']] = bmi_data.apply(calc_percentile, axis=1, result_type='expand')
    
    bmi_data[['N95','N97','N98','N120','N140']] = bmi_data.apply(categorize_indiv_bmi2, axis=1, result_type='expand')

    bmi_data_groups = bmi_data.groupby('subject_id')
    if categorization == '95':
        bmi_cat_df = bmi_data_groups.apply(categorize_indiv_patient2)
    elif categorization == '97':    
        bmi_cat_df = bmi_data_groups.apply(categorize_indiv_patient)
    else:
        raise ValueError("categorization must be 95 or 97")

    bmi_data['cat'] = bmi_data.subject_id.map(bmi_cat_df)    
    
    #bmi_cat_df.to_csv('/'.join([out_path, 'new_patient_cat.csv']))
    #bmi_data.to_csv('/'.join([out_path, 'new_patient_data_w_cat.csv']))
    if debug:
        pdb.set_trace()
    else:
        print(bmi_cat_df.value_counts())
        #bmi_cat_df.to_csv('/'.join([out_path, 'epic_refresh_bmi_cat2.csv']))
        #bmi_data.to_csv('/'.join([out_path, 'epic_refresh_data_w_cat2.csv']))
        #bmi_cat_df.to_csv('/'.join([out_path, 'new_patients_bmi_cat_frank_pctl_97_cat.csv']))
        #bmi_data.to_csv('/'.join([out_path, 'new_patients_data_frank_pctl_97_cat.csv']))
        ### FIXME: hardcode output files
        bmi_cat_df.to_csv('/'.join([out_path, 'epic_refresh_bmi_cat_steve_pctl_95_cat.csv']))
        bmi_data.to_csv('/'.join([out_path, 'epic_refresh_data_steve_pctl_95_cat.csv']))
        #
