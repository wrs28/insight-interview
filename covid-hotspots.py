import os
import subprocess
import pandas as pd
import numpy as np
import datetime
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor

# specify date to look at, defaults to today
end_date = np.datetime64(datetime.datetime.today())
# end_date = np.datetime64('2020-04-03')

# number of days to average case and death numbers over
n_days_window = 3

# number of days after which an infected individual can be assumed to be not infectious
# anectodally, about 3 weeks
n_days_of_infection = 26

# number of days between exposure and showing symptoms/becoming infectious
n_days_for_incubation = 5

# number of days from exposure to death (post-selected for those who die)
# anecdotally about 10 days after showing symptoms
n_days_to_death = 15

# minimum number of cases a county must have over entire window to be included in the dataset
min_num_cases = 100


feature_list = []

# measure of R_0, i.e. rate of infection per capita infected (at time of exposure)
def feature1(new_cases,new_deaths,
             new_case_baseline,
             new_death_baseline,
             population,
             cases,
             deaths,):
    return np.mean(new_cases/new_case_baseline)
feature_list.append(feature1)

# rate of death per capita infected (at time of exposure)
def feature2(new_cases,new_deaths,
             new_case_baseline,
             new_death_baseline,
             population,
             cases,
             deaths,):
    return np.mean(new_deaths/new_death_baseline)
feature_list.append(feature2)

# population of county (log, to make smoother and compress)
def feature3(new_cases,new_deaths,
             new_case_baseline,
             new_death_baseline,
             population,
             cases,
             deaths,):
    return np.mean(np.log(population))
feature_list.append(feature3)

# proxy for fraction of population that is infected (should be somewhat robust to poor testing)
def feature4(new_cases,new_deaths,
             new_case_baseline,
             new_death_baseline,
             population,
             cases,
             deaths,):
    return np.mean(new_case_baseline/population)
feature_list.append(feature4)

# proxy for fraction of population that has died (should be somewhat robust to poor testing)
def feature5(new_cases,new_deaths,
             new_case_baseline,
             new_death_baseline,
             population,
             cases,
             deaths,):
    return np.mean(deaths/population)
feature_list.append(feature5)





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

filename = os.path.join(project_dir,'cc-est2018-alldata.csv')

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

# left join
df = pd.merge(covid_df, census_df, on='fips', how='left')

# rename 'TOT_POP' column to `population`
df['population'] = df['TOT_POP'].astype('Int64')
df = df.drop(columns=['TOT_POP'])




# helper function finds last True expression in a boolean array
def findlast(bool_array):
    idx = len(bool_array)-1
    bool = bool_array[idx]
    while not bool:
        idx -= 1
        if idx <= 0:
            break
        bool = bool_array[idx]
    return idx



    # get unique fips and associated county and state names
FIPS, INDS = np.unique(np.array(df['fips']), return_index=True)
COUNTIES = np.array(df['county'])[INDS]
STATES = np.array(df['state'])[INDS]

# initialize sample data
X = []
Cases = []
Deaths = []
Population = []
fips = []
counties = []
states = []

for ind in range(len(FIPS)):

    fip = FIPS[ind]

    # select only one county
    x = df[df['fips']==fip]

    # get dates for that county, and start and end indices for time-window
    dates = np.array(x['date'])
    start_date = end_date - np.timedelta64(n_days_window,'D')
    start_idx = findlast(dates <= start_date)
    end_idx = findlast(dates <= end_date)

    # throw away county if time series is not long enough to support features
    if start_idx < max(n_days_of_infection, n_days_for_incubation, n_days_to_death):
        continue

    # helper functions for getting new_cases and deaths
    def differential(array, window):
        return array[start_idx:end_idx] - array[start_idx-window:end_idx-window]

    # helper functions for get baslines
    def shift(array, shift):
        return array[start_idx-shift:end_idx-shift]

    population = x['population'].mean()
    Population.append(population)
    cases = np.array(x['cases'])
    deaths = np.array(x['deaths'])

    # if number of cases too small (set by min_num_cases), throw it away
    if cases[start_idx] < min_num_cases:
        continue

    # build list of counties that meet all criteria for keeping
    fips.append(fip)
    counties.append(COUNTIES[ind])
    states.append(STATES[ind])

    # extract relevant dates
    dates = np.array(x['date'])[start_idx:end_idx]

    # build feature inputs
    currently_infected = differential(cases,n_days_of_infection)
    new_cases = differential(cases,1)
    new_deaths = differential(deaths,1)
    new_case_baseline = shift(cases, n_days_for_incubation)
    new_death_baseline = shift(cases, n_days_to_death)
    cases = shift(cases, 0)
    deaths = shift(deaths, 0)
    Cases.append(cases[-1])
    Deaths.append(deaths[-1])

    # compute features
    features = []
    for feature in feature_list:
        features.append(feature(new_cases,new_deaths,
                                new_case_baseline,
                                new_death_baseline,
                                population,
                                cases,
                                deaths,
                               )
                       )

    X.append(np.array(features))

# turn samples into ndarray, and list of counties and states, for boolean indexing later
X = np.asarray(X)
counties = np.asarray(counties)
states = np.asarray(states)

# mean-center and normalize data, using 'robust' b/c outlier problem
Xsc = preprocessing.robust_scale(X)

# run in parallel, auto-detect anomaly fraction
lof = LocalOutlierFactor(n_jobs=-1, contamination="auto")

# detect outliers
y_pred = lof.fit_predict(Xsc)

# get outlier scores
scores = lof.negative_outlier_factor_
perm = np.argsort(scores)
scores_sorted = scores[perm]
counties_sorted = counties[perm]
states_sorted = states[perm]
y_pred_sorted = y_pred[perm]



# print results
print('%i anomalies detected, %2.2f%% counties that meet case threhold:\n' % (sum(y_pred==-1),100*sum(y_pred==-1)/len(counties)))

print('%5s %20s %20s %20s %20s %20s %20s %20s\n' % ('SCORE','STATE','COUNTY','CASES','DEATHS','POPULATION','INFECTION RATE','DEATH RATE'))
for p in range(len(y_pred_sorted)):
    if y_pred_sorted[p]==-1:
        print('%5.2f %20s %20s %20i %20i %20i %20.4f %20.4f' % (-scores_sorted[p], states_sorted[p], counties_sorted[p], Cases[perm[p]], Deaths[perm[p]], Population[perm[p]], X[perm[p],0], X[perm[p],1]))
print()

Cases
