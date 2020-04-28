# Insight Interview Project: Anomaly Detection on COVID-19 County-Level Data

***Goal***: of the > 3000 counties in the US, want to identify those most likely to become the next 'hotspots'. By this I mean places showing anomalous  the infection rates and/or death rates. I do not mean places that simply have 'a lot' of cases, such as NYC, which was *bound* to have high numbers, but also has high resources. This could identify a subset of places for investigators to focus their attention and for disaster relief managers to be vigilent about sending support.

***Solution***: combine [COVID-19 dataset](https://github.com/nytimes/covid-19-data) from the New York Times's public repository, with US Census Bureau's most recent [(2018) estimate of population by county](https://www2.census.gov/programs-surveys/popest/datasets/2010-2018/counties/asrh/cc-est2018-alldata.csv). Build features, run Local Outlier Factor analysis.

## Cleaning the Datasets

### NYT COVID-19 data

There are some [geographic exceptions](https://github.com/nytimes/covid-19-data#geographic-exceptions) which we will have. In particular, Kansas City and New York City are counted separately from the counties. Also, the `fips` identifiers are concatenated state + county codes.

We create new `fips` `'nycty'` and `'kscty'`, and also remove `'Unkown'` cases.

### US Census data

Have to create 5-digit concatenated `fips` codes from state and county `fips`. Also, have to extract relevant data, which is total population by county (columns `YEAR=11`, `AGEGRP=0`), can discard the rest.

Have to aggregate the counties surrounding NYC and Kansas City to make consistent with NYT's dataset.

### Features

Features defined in `src/define_features.py`.

They are build from: `n_days_window`-length series of `new_cases`, `new_deaths`, `new_case_baseline` (how many cases were there `n_days_for_incubation` ago?), `new_death_baseline` (how many cases were there `n_days_to_death` ago?), and `population`.

Features are parameterized by `n_days_window`, `n_days_for_incubation`, `n_days_to_death`, `n_days_of_infection`. `end_date` determines which day we are doing the analysis for, defaults to today.

Any counties whose case record does not exceed `min_num_cases` over the window defined by `n_days_window` and `end_date` are ignored.

### Outlier Detection

[Local Outlier Factor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor) estimation. Computes the local density of points using a nearest-neighbor algorithm, and identifies outliers based on how the local density at each point compares to the average density across all sample points.
