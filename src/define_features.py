import numpy as np

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

# proxy for fraction of population that is infected (should be somewhat robust to poor testing)
def feature6(new_cases,new_deaths,
             new_case_baseline,
             new_death_baseline,
             population,
             cases,
             deaths,):
    return np(cases[-1]/population)
feature_list.append(feature6)
