import os
import subprocess
import pandas as pd
import numpy as np

pd.set_option('mode.chained_assignment', None)

# project home directory
project_dir = os.path.abspath('')

# run git pull on nyt's repository
msg = subprocess.run(['git','pull'],
                     cwd=os.path.join(project_dir,'nyt'),
                     stdout=subprocess.PIPE)

# print git output for good measure
print(msg.stdout.decode('UTF-8'))

filename = os.path.join(project_dir,'nyt','us-counties.csv')

# will need fips as string for later when combining with Census data
col_dtypes = {'fips' : str}

# also, parse dates as dates!
df = pd.read_csv(filename, dtype=col_dtypes, parse_dates=[0])

# helper function to combine Kansas city with four surrounding counties
def combine_kansas_city_covid(df,mo_dfs):

    # initialize dataframe with Kansas City data
    mo_df = df[df['county']=='Kansas City']

    # set fips to custom 'kscty'
    mo_df.loc[:,'fips'] = 'kscty'

    # loop over each of the counties and each of the halves of the merged dataframe
    for d in mo_dfs:

        # outer join on 'date' and 'state'
        mo_df = pd.merge(mo_df, d, on=['date','state'], how='outer')

        # set N/A cases and deaths from join to zero
        for cases in ('cases_x','cases_y'):
            mo_df[cases] = mo_df[cases].astype('Int64')
            mo_df.loc[np.isnan(mo_df[cases]),[cases]]=0
        for deaths in ('deaths_x','deaths_y'):
            mo_df[deaths] = mo_df[deaths].astype('Int64')
            mo_df.loc[np.isnan(mo_df[deaths]),[deaths]]=0

        # total cases
        mo_df['cases'] = (mo_df['cases_x'] + mo_df['cases_y']).map(int)

        # total deaths
        mo_df['deaths'] = (mo_df['deaths_x'] + mo_df['deaths_y']).map(int)

        # set fips, county, state
        mo_df['fips'] = 'kscty'
        mo_df['county'] = 'Kansas City'
        mo_df['state'] = 'Missouri'

        # keep only columns we need
        mo_df = mo_df[['date','county','state','fips','deaths','cases']]

    # eliminate original 'Kansas City' data, since it's now in mo_df
    df = df[df['county']!='Kansas City']

    # add new kansas city+ rows
    df = pd.concat([df,mo_df])

    # remove each of 'Cass', 'Clay', etc
    for county in counties:
       df = df[df['county']!=county]

    return df

# these are the four counties in which Kansas City lies
counties = ['Cass','Clay','Jackson','Platte']

# closure of dataframes for each county surrounding Kansas City
mo_dfs = [df[(df['county']==county) & (df['state']=='Missouri')] for county in counties]

# new dataframe with Kansas City fixed
covid_df = combine_kansas_city_covid(df,mo_dfs)

# use custom fips for NYC b/c NYT data bins NYC too
covid_df.loc[covid_df['county']=='New York City','fips'] = 'nycty'

# remove unknown counties
covid_df = covid_df[covid_df['county']!='Unknown']

covid_df.reset_index(inplace=True)

covid_df.to_json('covid_data_2020.json')
