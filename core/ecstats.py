from .eclogging import load_logger
from scipy.stats import chi2_contingency
import itertools
import pandas as pd
import numpy as np

logger = load_logger()


###
def chisquare_test_independence(dataframe, target=None, test_all_comb=True):
    """
    Parameters
    ----------
    dataframe = e.g. md_x[categorical_feature_columns]
    target = e.g. md_y
    test_all_comb = 

    """
    global result_table
    cdf=dataframe.copy()
    cat=cdf
    result_table=pd.DataFrame(index=cdf.columns, columns=['chi2', 'p_value', 'dof', 'expected'])
    
    if test_all_comb is True: # 모든 조합에서 검정을 진행
        cat_uniqs=cat.columns.unique() # Get unique values
        cat_comb=list(itertools.combinations(cat_uniqs, 2)) # Get all combinations of cat_uniqs.
        table_index=list(map(str, cat_comb)) # Make table index, convert tuple values into string.
        result_table=pd.DataFrame(columns=['chi2', 'p_value', 'dof', 'expected'])

        for i in cat_comb: # Chi square test
            print('')
            print('i: ', i)
            con_table=pd.crosstab(cat[i[0]], cat[i[1]])
            
            try:
                con_table=con_table.drop(-1.0, axis=0)
            except Exception as e:
                print(f'Dropping -1.0 indices was not progressed. \n{e}')
                
            try:
                con_table=con_table.drop(-1.0, axis=1)
            except Exception as e:
                print(f'Dropping -1.0 columns was not progressed. \n{e}')
            
            print("con_table: ", con_table, sep='\n')
            
            try:
                chi2, p, dof, expected = chi2_contingency(observed=con_table) # observed - input
            except ValueError as e:
                print(f'ValueError: {e}')
            
            print("Expected: ", expected, sep='\n')
            if ((expected < 5).sum() / len(expected.flatten())) > .2 : 
                print(r">> The proportion of expected values less than 5 is more than or equal to 20%. This function will pass current column, {}.".format(i))
                continue
                
            # Save result to result table.
            ti=str(i)
            result_table.loc[ti, 'chi2'] = chi2
            result_table.loc[ti, 'p_value'] = p
            result_table.loc[ti, 'dof'] = dof
            result_table.loc[ti, 'expected'] = expected

    if test_all_comb is False: # 모든 조합에서 검정을 진행하지 않음.
        if target is None: raise ValueError(f"Target argument should be given.")
        for c in cdf.columns:
            print(">> Current column:", c)
            con_table=pd.crosstab(cdf[c], target) # Make contingency table

            print(con_table)
            
            chi2, p, dof, expected = chi2_contingency(observed=con_table) # observed - input

            print("Expected: ", expected, sep='\n')
            if ((expected < 5).sum() / len(expected.flatten())) >= .2 : 
                print(r">> The proportion of expected values less than 5 is more than or equal to 80%. This function will pass current column, {}.".format(c))
                continue

            # Save result to result table.
            result_table.loc[c, 'chi2'] = chi2
            result_table.loc[c, 'p_value'] = p
            result_table.loc[c, 'dof'] = dof
            result_table.loc[c, 'expected'] = expected

            print('')

    result_table=result_table.sort_values(by='p_value')

    return result_table
