import numpy as np
import datetime

# specify date to look at, defaults to today
end_date = np.datetime64(datetime.datetime.today())
# end_date = np.datetime64('2020-04-03')

# number of days to average case and death numbers over
n_days_window = 3

# number of days between exposure and showing symptoms/becoming infectious
n_days_for_incubation = 5

# number of days after which an infected individual can be assumed to be not infectious
# anectodally, about 3 weeks
n_days_of_infection = n_days_for_incubation + 3*7

# number of days from exposure to death (post-selected for those who die)
# anecdotally about 10 days after showing symptoms
n_days_to_death = n_days_for_incubation + 11

# minimum number of cases a county must have over entire window to be included in the dataset
min_num_cases = 75


import os

if not os.path.exists('census_data_2018.json'):
    import gather_census

import gather_covid

import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor

from define_features import feature_list

covid_df = pd.read_json('covid_data_2020.json')
census_df = pd.read_json('census_data_2018.json')

# left join
df = pd.merge(covid_df, census_df, on='fips', how='left')

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
