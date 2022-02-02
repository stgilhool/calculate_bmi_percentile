import pandas as pd
import numpy as np
import argparse
import math

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


def zptile(z_score: float) -> float:
    '''
    Convert z-score to percentile
    '''
    return .5 * (math.erf(z_score / 2. ** .5) + 1)

def calc_z_score(bmi: float, L_arr: np.ndarray, M_arr: np.ndarray, S_arr: np.ndarray, adjusted=True) -> np.ndarray:
    '''
    Calculate z-score given single bmi and L, M, S parameter vectors
    '''
    zscore_arr = (((bmi/M_arr)**L_arr) - 1.)/(L_arr*S_arr)
    if not adjusted:
        return zscore_arr
    else:
        zscore_final = np.zeros_like(zscore_arr)

        for idx, (L, M, S, zscr) in enumerate(zip(L_arr, M_arr, S_arr, zscore_arr)):
            if abs(zscr) <= 3.0:
                zscr_final = zscr
            else:
                # Calculated adjusted z-score
                def sdlevel(level: float) -> float:
                    return M * (1. + (level * L * S)) ** (1./L)
                if zscr < -3:
                    SD3neg = sdlevel(-3)
                    SD23neg = sdlevel(-2) - SD3neg
                    zscr_final = -3 + ((bmi - SD3neg)/SD23neg)
                elif zscr > 3:
                    SD3pos = sdlevel(3)
                    SD23pos = SD3pos - sdlevel(2)
                    zscr_final = 3 + ((bmi - SD3pos)/SD23pos)
            zscore_final[idx] = zscr_final

        return zscore_final

def calc_percentile(row: pd.Series, pctl=True, **kwargs):
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
    
    Z_arr = calc_z_score(bmi, L_arr, M_arr, S_arr, kwargs)
    z_score = np.interp(age_months, age_table, Z_arr)
    if pctl:
        output = zptile(z_score)
    else:
        output = z_score

    if table_type == 'cdc':
        out = [output, np.nan]
    elif table_type == 'who':
        out = [np.nan, output]
    else:
        raise ValueError("table_type must be cdc or who")

    return out

def main(infile):
    # Read in the data
    bmi_data = pd.read_csv(infile,header=0)
    
    # Fill in the zscore columns
    bmi_data[['bmi_pctl_cdc','bmi_pctl_who']] = bmi_data.apply(calc_percentile, axis=1, result_type='expand')
    bmi_data[['bmi_z_cdc','bmi_z_who']] = bmi_data.apply(calc_percentile, pctl=False, axis=1, result_type='expand')

    bmi_data[['bmi_pctl_cdc_raw','bmi_pctl_who_raw']] = bmi_data.apply(calc_percentile, adjusted=False, axis=1, result_type='expand')
    bmi_data[['bmi_z_cdc_raw','bmi_z_who_raw']] = bmi_data.apply(calc_percentile, pctl=False, adjusted=False, axis=1, result_type='expand')

    # Make summary pctl_raw, pctl, zscore_raw and zscore columns (ie take either CDC or WHO values)
    bmi_data['bmi_pctl'] = bmi_data.apply(lambda row: np.nanmax(np.array([row.bmi_pctl_cdc, row.bmi_pctl_who])), axis=1)
    bmi_data['bmi_zscore'] = bmi_data.apply(lambda row: np.nanmax(np.array([row.bmi_z_cdc, row.bmi_z_who])), axis=1)

    bmi_data['bmi_pctl_raw'] = bmi_data.apply(lambda row: np.nanmax(np.array([row.bmi_pctl_cdc_raw, row.bmi_pctl_who_raw])), axis=1)
    bmi_data['bmi_zscore_raw'] = bmi_data.apply(lambda row: np.nanmax(np.array([row.bmi_z_cdc_raw, row.bmi_z_who_raw])), axis=1)

    return bmi_data
    

if __name__ == "__main__":

    '''
    Take a table containing bmi data, add columns containing bmi z-score and percentile data, and write to file

    INPUTS:
    infile - str - Full path to csv file containing bmi data. Must contain 'visit_agedays', 'gender', 'bmi'
    outfile - str (optional, default = 'output.csv') -  Path and filename to write the table with z-score/percentile info
    ''' 
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate bmi zscores and percentiles and add them to table')
    parser.add_argument('infile',
                        type=str,
			nargs=1,
                        help='Path to file containing bmi, age and gender data')
    parser.add_argument('outfile',
                        type=str,
                        nargs='?',
                        default='output.csv',
                        help='Filename (path) of output table including bmi zscores/percentiles')
    
    args = parser.parse_args()
    
    infile = args.infile[0]
    outfile = args.outfile
    print(infile)
    print(outfile)

    bmi_data = main(infile)

    print(bmi_data)
    
    bmi_data.to_csv(outfile, index=False)
    

