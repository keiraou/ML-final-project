import pandas as pd

def read_dta_file():
    filepath = '~/Harris_Capp/ml/project/data/'

    file_dic = {
                # 'Nepal_2016/NPHR7HDT/NPHR7HFL.DTA': "Nepal_2016_household_record", 
                # 'Nepal_2016/NPIR7HDT/NPIR7HFL.DTA': 'Nepal_2016_individual_record', 
                # 'Nepal_2016/NPPR7HDT/NPPR7HFL.DTA': 'Nepal_2016_member_record', 
                # 'Pakistan_2017_18/PKHR71DT/PKHR71FL.DTA': 'Pakistan_1718_household_record',
                # 'Pakistan_2017_18/PKIR71DT/PKIR71FL.DTA': 'Pakistan_1718_individual_record',
                # 'Pakistan_2017_18/PKPR71DT/PKPR71FL.DTA': 'Pakistan_1718_member_record',
                # 'Philippines_2017/PHHR71DT/PHHR71FL.DTA': 'Philippines_2017_household_record',
                # 'Philippines_2017/PHIR71DT/PHIR71FL.DTA': 'Philippines_2017_individual_record',
                # 'Philippines_2017/PHPR71DT/PHPR71FL.DTA': 'Philippines_2017_member_record',
                # 'Maldives_2016_17/MVHR71DT/MVHR71FL.DTA': 'Maldives_1617_household_record',
                # 'Maldives_2016_17/MVIR71DT/MVIR71FL.DTA': 'Maldives_1617_individual_record',
                # 'Maldives_2016_17/MVPR71DT/MVPR71FL.DTA': 'Maldives_1617_member_record',
                # 'Cambodia_2014/KHHR73DT/KHHR73FL.DTA': 'Cambodia_2014_household_record',
                # 'Cambodia_2014/KHIR73DT/KHIR73FL.DTA': 'Cambodia_2014_individual_record',
                # 'Cambodia_2014/KHPR73DT/KHPR73FL.DTA': 'Cambodia_2014_member_record',
                'India_2015_16/IAHR74DT/IAHR74FL.DTA': 'India_1516_household_record',
                'India_2015_16/IAIR74DT/IAIR74FL.DTA': 'India_1516_individual_record',
                'India_2015_16/IAPR74DT/IAPR74FL.DTA': 'India_1516_member_record'}
        
    for path, filename in file_dic.items():
        df = pd.read_stata(filepath + path, convert_categoricals=False)
        dataframe_to_csv(df, filename)


def dataframe_to_csv(dataframe, filename):
    '''
    Converts Pandas Dataframe to csv file

    Input:
        dataframe: pandas DataFrame object
        filename: str
    Output:
        a CSV file
    '''
    path = "~/Harris_Capp/ml/project/rawdata/" + filename + ".csv"
    dataframe.to_csv(path)
