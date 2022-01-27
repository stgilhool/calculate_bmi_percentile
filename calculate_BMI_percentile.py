import pandas as pd
import numpy as np
import os
import argparse
import math
import sys

bmi_table_cdc_filename = 'bmi_table_cdc.csv'
bmi_table_boys_who_filename = 'bmi_table_boys_who.txt'
bmi_table_girls_who_filename = 'bmi_table_girls_who.txt'

#Read in the CDC percentile data
pctl_table_cdc = pd.read_csv(bmi_table_cdc_filename, header=0)
cdc_pctl_boys = pctl_table_cdc.loc[pctl_table_cdc.Sex == 1]
cdc_pctl_girls = pctl_table_cdc.loc[pctl_table_cdc.Sex == 2]

who_pctl_boys = pd.read_csv(bmi_table_boys_who_filename,
                            sep='\t',
                            header=0)
who_pctl_boys.rename({'Month':'Agemos'},axis=1,inplace=True)
who_pctl_girls = pd.read_csv(bmi_table_girls_who_filename,
                            sep='\t',
                            header=0)
who_pctl_girls.rename({'Month':'Agemos'},axis=1,inplace=True)


def zptile(z_score):
    '''
    Convert z-score to percentile
    '''
    return .5 * (math.erf(z_score / 2. ** .5) + 1)

def calc_z_score(bmi, L, M, S):
    '''
    Calculate z-score given bmi and L, M, S parameters (scalar or vector)
    '''
    return (((bmi/M)**L) - 1.)/(L*S)

def calc_percentile(row):
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
        out = [z_score, np.nan, percentile, np.nan]
    elif table_type == 'who':
        out = [np.nan, z_score, np.nan, percentile]
    else:
        raise ValueError("table_type must be cdc or who")

    return out

if __name__ == "__main__":

    '''
    Take a table containing bmi data, add columns containing bmi z-score and percentile data, and write to file

    INPUTS:
    infile - str - Full path to csv file containing bmi data. Must contain 'visit_agedays', 'gender', 'bmi'
    outfile - str (optional, default = 'output.csv') -  Path and filename to write the table with z-score/percentile info
    ''' 

    args = sys.argv
    if len(args) < 2:
        raise ValueError("must supply file name of input csv file")
    elif len(args) == 2:
        infile = args[1]
        outfile = 'output.csv'
    elif len(args) == 3:
        infile, outfile = args[1:3]
    else:
        print("WARNING: more than two command line arguments supplied. Ignoring additional args.")
        
    bmi_data = pd.read_csv(infile,header=0)
    
    bmi_data[['bmi_z_cdc', 'bmi_z_who', 'bmi_pctl_cdc','bmi_pctl_who']] = bmi_data.apply(calc_percentile, axis=1, result_type='expand')

    bmi_data.to_csv(outfile, index=False)
    

