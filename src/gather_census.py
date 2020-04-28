import os
import pandas as pd
import numpy as np
import csv
import requests

pd.set_option('mode.chained_assignment', None)

project_dir = os.path.abspath('')

url = 'https://www2.census.gov/programs-surveys/popest/datasets/2010-2018/counties/asrh/cc-est2018-alldata.csv'
response = requests.get(url)
with open('census_data_2018.csv', 'w') as f:
    writer = csv.writer(f)
    for line in response.iter_lines():
        writer.writerow(line.decode('ISO-8859-1').split(','))

filename = os.path.join(project_dir,'census_data_2018.csv')

# columns we care about
cols = ['TOT_POP', 'COUNTY','AGEGRP','YEAR','STATE']

df1 = pd.read_csv(filename, usecols=cols)

# add new `fips` column by 2-digit state code and concatenating it with 3-digit county code
df1['fips'] = df1['STATE'].map('{:02d}'.format) + df1['COUNTY'].map('{:03d}'.format)

df1 = df1[(df1['YEAR'] == 11) & (df1['AGEGRP']==0)][['fips','TOT_POP']]

# dict of nyc counties and fip codes
nyc_counties = {
    'Bronx'    : '36005',
    'Kings'    : '36047',
    'New York' : '36061',
    'Queens'   : '36081',
    'Richmond' : '36085',
}

# dict of kansas city counties and fip codes
mo_counties = {
    'Cass'    : '29037',
    'Clay'    : '29047',
    'Jackson' : '29095',
    'Platte'  : '29165',
}

# dummy dataframe. New fips are `nycty` and `kscty` just like with COVID datafram
df2 = pd.DataFrame({'fips': ['nycty','kscty'],'TOT_POP' : [0,0]})

# add NYC counties together
for fips in nyc_counties.values():
     df2['TOT_POP'][0] += df1[df1['fips']==fips]['TOT_POP']

# add Kansas City counties together
for fips in mo_counties.values():
    df2['TOT_POP'][1] += df1[df1['fips']==fips]['TOT_POP']

# add new NYC and Kansas City rows
census_df = pd.concat([df1, df2])

census_df['population'] = census_df['TOT_POP'].astype('Int64')
census_df = census_df.drop(columns=['TOT_POP'])

census_df.to_json('census_data_2018.json')
