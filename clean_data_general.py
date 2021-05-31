'''
Cleans data files of multiple countries
'''

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

FILE_DICT = {'Cambodia_2014.DTA': {'country': 'Cambodia', 'year': '2014'},
    'Maldives_2016.DTA': {'country': 'Maldives', 'year': '2016'},
    'Nepal_2016.DTA': {'country': 'Nepal', 'year': '2016'},
    'Pakistan_2017.DTA': {'country': 'Pakistan', 'year': '2017'},
    'Philippines_2017.DTA': {'country': 'Philippines', 'year': '2017'},
}

VARIABLE_CODE = ["caseid","v101","v025", "d104", "d106", "d107", "d108",
              "v133","v190", "v501", "v502","v731", "v741", "v012", 
              "v745a","v745b", "v746", "v715", "v136", "v201", "v151", "v536", "v613", "v621", "v739"]

KEPT_VARIABLE = ['caseid', 'country', 'year',
                 'province','age','education', 'if_union', 'if_urban',
                 'wealth_index','wealth_index_code', 'house_ownership', 'land_owenership', 'if_own_house', 'if_own_land',
                 'if_employment', 'if_employment_current','employment_pay_method','if_earn_more',
                 'partner_edu', 
                 'num_household', 'num_child','sex_head_household', 'sexual_activity', 'ideal_num_child', 'partner_ideal_child', 'money_decide_person',
                'if_emo_vio', 'if_phy_vio', 'if_phy_vio_severe', 'if_sex_vio', 'if_vio', 'num_vio']

def process_files():
    '''
    Processes data files of multiple countries
    '''
    for file, file_info in FILE_DICT.items():
        path = "rawdata/" + file
        df = pd.read_stata(path, columns=VARIABLE_CODE)
        # df = df[VARIABLE_CODE]
        country = file_info['country']
        year = file_info['year']
        return_df = clean_data(df, country, year)
        filename = file.split('.')[0]
        output_path = filename + "_cleaned.csv"
        return_df.to_csv(output_path)


def clean_data(df, country, year):
    '''
    Cleans data
    '''
    # Filter: only include data with women who has been in marriage/union
    df['if_union'] = None
    df.loc[(df['v502'] == 1) | (df['v501'] == 'married') | (df['v502'].str.contains("currently")),
    'if_union'] = 1

    df_filtered = df[df['if_union'] == 1]
    df_filtered['country'] = country
    df_filtered['year'] = year

    rename_col = {}
    rename_col = {'v101': 'province',
        'v012': 'age',
        'v025': 'if_urban',
        'v133': 'education',
        'v190': 'wealth_index',
        'v715': 'partner_edu',
        'v745a': 'house_ownership',
        'v745b': 'land_owenership',
        'v731': 'if_employment',
        'v741': 'employment_pay_method',
        'v746': 'if_earn_more',
        "v136": 'num_household', 
        "v201": 'num_child',
        "v151": 'sex_head_household', 
        "v536": 'sexual_activity', 
        "v613": 'ideal_num_child', 
        "v621": 'partner_ideal_child', 
        "v739": 'money_decide_person'}

    df_filtered.rename(columns=rename_col, inplace=True)

    # Check NA values
    print(df_filtered.isna().sum())

    # Target: if_emo_vio, if has emotional violence
    df_filtered['if_emo_vio'] = None
    df_filtered.loc[(df_filtered['d104'] == 'yes'),'if_emo_vio'] = 1
    df_filtered.loc[(df_filtered['d104'] == 'no'),'if_emo_vio'] = 0
    df_filtered.groupby('if_emo_vio').caseid.nunique()

    # Target: if_phy_vio, if has physical violence
    df_filtered['if_phy_vio'] = None
    df_filtered.loc[(df_filtered['d106'] == 'yes') | (df_filtered['d107'] == 'yes'),'if_phy_vio'] = 1
    df_filtered.loc[(df_filtered['d106'] == 'no') & (df_filtered['d107'] == 'no'),'if_phy_vio'] = 0

    # Target: if_phy_vio_severe, if has severe physical violence
    df_filtered['if_phy_vio_severe'] = None
    df_filtered.loc[(df_filtered['d107'] == 'yes'),'if_phy_vio_severe'] = 1
    df_filtered.loc[(df_filtered['d107'] == 'no'),'if_phy_vio_severe'] = 0

    # Target: if_sex_vio, if has sexual violence
    df_filtered['if_sex_vio'] = None
    df_filtered.loc[(df_filtered['d108'] == 'yes'),'if_sex_vio'] = 1
    df_filtered.loc[(df_filtered['d108'] == 'no'),'if_sex_vio'] = 0

    # Target: num_vio, number of violence kinds the woman has
    df_filtered['num_vio'] = None
    df_filtered['num_vio'] = df_filtered['if_emo_vio'] + df_filtered['if_phy_vio'] + df_filtered['if_sex_vio']

    # Target: if_vio, if has any of the three kinds of violence
    df_filtered['if_vio'] = None
    df_filtered.loc[(df_filtered['num_vio'] > 0),'if_vio'] = 1
    df_filtered.loc[(df_filtered['num_vio'] == 0),'if_vio'] = 0

    # Features: Wealth_index_code
    wealth_index_dict = {'poorest': 0,
                        'poorer': 1,
                        'middle': 2,
                        'richer': 3,
                        'richest': 4}
    df_filtered['wealth_index_code'] = df_filtered['wealth_index'].replace(wealth_index_dict, inplace=False)

    # Features: if_own_house
    df_filtered['if_own_house'] = 1
    df_filtered.loc[(df_filtered['house_ownership'] == 'does not own'),'if_own_house'] = 0

    # Features: if_own_land
    df_filtered['if_own_land'] = 1
    df_filtered.loc[(df_filtered['land_owenership'] == 'does not own'),'if_own_land'] = 0

    # Features: if_employment_current
    df_filtered['if_employment_current'] = 0
    df_filtered.loc[
        (df_filtered['if_employment'] == 'have a job, but on leave last 7 days') | (
            df_filtered['if_employment'] == 'currently working'), 
        'if_employment_current'] = 1

    output_df = df_filtered[KEPT_VARIABLE]
    return output_df

if __name__ == "__main__":
    process_files()
