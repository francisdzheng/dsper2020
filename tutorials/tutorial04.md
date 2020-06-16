---
layout: page
title: Subset Selection Methods
parent: Tutorials
nav_exclude: true
latex: true
---

# Subset Selection Methods
{:.no_toc}

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

In this tutorial, we will write several functions to select and evaluate subsets of features of baseball players used to predict salary. 

Much of this code was provided by Professor Kucheryavyy; I have broken the code down into a few smaller pieces and added some comments and explanations that should help your understanding. You can view the code for this tutorial [here](https://colab.research.google.com/drive/1ciGROXpqnYZSDdy3pRXt7D0zIYXrzQzM).

## Importing Libraries

We will be using several libaries to aid our modeling and graphing:

```python
import itertools
import pandas as pd
import numpy as np
import copy

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display

# Reset all styles to the default:
plt.rcParams.update(plt.rcParamsDefault)
# Then make graphs inline:
%matplotlib inline

# Set custom style settings:
# NB: We need to separate "matplotlib inline" call and these settings into different
# cells, otherwise the parameters are not set. This is a bug somewhere in Jupyter
plt.rcParams['figure.figsize'] = (7, 6)
plt.rcParams['font.size'] = 24
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['figure.titlesize'] = 'large'
plt.rcParams['lines.markersize'] = 10
```

We won't necessarily use all of these in this specific tutorial, but please import the above so that you have a wide array of tools at your disposal. You can feel free to use this libraries in your homework as well! 

## Our Dataset

In this tutorial, we will be taking a look at [Hitters.csv](Hitters.csv), a dataset that describes several attributes of major league players, such as salary, number of homeruns, etc. 

First, let's load our data with the following:

```python
hittersDF = pd.read_csv('Hitters.csv', na_values=[""])
```

If you look at the first few rows of the data (`hittersDF.head()`), you may notice that the first column has no name. We'll give it a name using the following:

```python
hittersDF.rename(columns={hittersDF.columns[0] : "Name"}, inplace=True, copy=False)
hittersDF.set_index('Name', inplace=True)
```

We will also [remove any missing values](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html) with the following: 

```python
hittersDF.dropna(inplace=True)
```

> Note: 
> `inplace=True` means that we are modifying the original dataframe itself. We are not creating a copy with our new changes. 

We want to convert categorial variables into dummy variables. Conversion into dummy variables allows us to perform regression for quanlitative variables. Let's run the following: 

```python
dummies = pd.get_dummies(hittersDF[['League', 'Division', 'NewLeague']])
X = hittersDF.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1)
cat_features = ['League_N', 'Division_W', 'NewLeague_N']
num_features = list(X.columns)
```

Now, we can better define our X and y variables like so:

```python
X = pd.concat([X, dummies[cat_features]], axis=1).astype('float64')
y = hittersDF.Salary
```

Let's check out the first ten rows for our features:

```python
display(X[0:10])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Name</th>
      <th>AtBat</th>
      <th>Hits</th>
      <th>HmRun</th>
      <th>Runs</th>
      <th>RBI</th>
      <th>Walks</th>
      <th>Years</th>
      <th>CAtBat</th>
      <th>CHits</th>
      <th>CHmRun</th>
      <th>CRuns</th>
      <th>CRBI</th>
      <th>CWalks</th>
      <th>PutOuts</th>
      <th>Assists</th>
      <th>Errors</th>
      <th>League_N</th>
      <th>Division_W</th>
      <th>NewLeague_N</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>-Alan Ashby</th>
      <td>315.0</td>
      <td>81.0</td>
      <td>7.0</td>
      <td>24.0</td>
      <td>38.0</td>
      <td>39.0</td>
      <td>14.0</td>
      <td>3449.0</td>
      <td>835.0</td>
      <td>69.0</td>
      <td>321.0</td>
      <td>414.0</td>
      <td>375.0</td>
      <td>632.0</td>
      <td>43.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>-Alvin Davis</th>
      <td>479.0</td>
      <td>130.0</td>
      <td>18.0</td>
      <td>66.0</td>
      <td>72.0</td>
      <td>76.0</td>
      <td>3.0</td>
      <td>1624.0</td>
      <td>457.0</td>
      <td>63.0</td>
      <td>224.0</td>
      <td>266.0</td>
      <td>263.0</td>
      <td>880.0</td>
      <td>82.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>-Andre Dawson</th>
      <td>496.0</td>
      <td>141.0</td>
      <td>20.0</td>
      <td>65.0</td>
      <td>78.0</td>
      <td>37.0</td>
      <td>11.0</td>
      <td>5628.0</td>
      <td>1575.0</td>
      <td>225.0</td>
      <td>828.0</td>
      <td>838.0</td>
      <td>354.0</td>
      <td>200.0</td>
      <td>11.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>-Andres Galarraga</th>
      <td>321.0</td>
      <td>87.0</td>
      <td>10.0</td>
      <td>39.0</td>
      <td>42.0</td>
      <td>30.0</td>
      <td>2.0</td>
      <td>396.0</td>
      <td>101.0</td>
      <td>12.0</td>
      <td>48.0</td>
      <td>46.0</td>
      <td>33.0</td>
      <td>805.0</td>
      <td>40.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>-Alfredo Griffin</th>
      <td>594.0</td>
      <td>169.0</td>
      <td>4.0</td>
      <td>74.0</td>
      <td>51.0</td>
      <td>35.0</td>
      <td>11.0</td>
      <td>4408.0</td>
      <td>1133.0</td>
      <td>19.0</td>
      <td>501.0</td>
      <td>336.0</td>
      <td>194.0</td>
      <td>282.0</td>
      <td>421.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>-Al Newman</th>
      <td>185.0</td>
      <td>37.0</td>
      <td>1.0</td>
      <td>23.0</td>
      <td>8.0</td>
      <td>21.0</td>
      <td>2.0</td>
      <td>214.0</td>
      <td>42.0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>9.0</td>
      <td>24.0</td>
      <td>76.0</td>
      <td>127.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>-Argenis Salazar</th>
      <td>298.0</td>
      <td>73.0</td>
      <td>0.0</td>
      <td>24.0</td>
      <td>24.0</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>509.0</td>
      <td>108.0</td>
      <td>0.0</td>
      <td>41.0</td>
      <td>37.0</td>
      <td>12.0</td>
      <td>121.0</td>
      <td>283.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>-Andres Thomas</th>
      <td>323.0</td>
      <td>81.0</td>
      <td>6.0</td>
      <td>26.0</td>
      <td>32.0</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>341.0</td>
      <td>86.0</td>
      <td>6.0</td>
      <td>32.0</td>
      <td>34.0</td>
      <td>8.0</td>
      <td>143.0</td>
      <td>290.0</td>
      <td>19.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>-Andre Thornton</th>
      <td>401.0</td>
      <td>92.0</td>
      <td>17.0</td>
      <td>49.0</td>
      <td>66.0</td>
      <td>65.0</td>
      <td>13.0</td>
      <td>5206.0</td>
      <td>1332.0</td>
      <td>253.0</td>
      <td>784.0</td>
      <td>890.0</td>
      <td>866.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>-Alan Trammell</th>
      <td>574.0</td>
      <td>159.0</td>
      <td>21.0</td>
      <td>107.0</td>
      <td>75.0</td>
      <td>59.0</td>
      <td>10.0</td>
      <td>4631.0</td>
      <td>1300.0</td>
      <td>90.0</td>
      <td>702.0</td>
      <td>504.0</td>
      <td>488.0</td>
      <td>238.0</td>
      <td>445.0</td>
      <td>22.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>

<!-- In total, it contains 263 rows and 20 columns (remember that you can do this with something like `df.shape`!) -->

## Best Subset Selection

### Best Subset Selection Functions

Now, let's write a couple functions that will help us select the best subset of our features. 

We will first write a function that allows us to input our set of features (`X`), the variable we want to predict (`Y`, or salary in this case), and define a specific size for our subset (`subset_size`). To do this, we will loop through all possible combinations of our features from `X` that are of size `subset_size`. In each iteration of our loop, we fit a linear regression model and calculate a mean squared error. We keep track of the best mean squared error through compariosn, which allows us to select the best subset in the end:  

```python
def findBestSubsetFixedSize(X, y, subset_size):
    features_nmb = X.shape[1]
    best_subset = []
    best_mse = -1
    for idx_set in itertools.combinations(range(features_nmb), subset_size):
        X_subset = X.iloc[:, list(idx_set)]
        lin_reg = LinearRegression(fit_intercept=True, normalize=False)
        lin_reg.fit(X_subset, y)
        yhat = lin_reg.predict(X_subset)
        mse_resid = mean_squared_error(y, yhat)
        if best_mse < 0 or mse_resid < best_mse:
            best_subset = list(idx_set)
            best_mse = mse_resid
    return([best_subset, best_mse])
```

We also want to write a function that simply allows us to set an upper limit on the size of the subset. To do this, we will write a loop that tests all possible values for our subset size and inputs them into the function we just wrote, `findBestSubsetFixedSize()`:

```python
def findBestSubset(X, y, max_subset_size):
    best_subsets = [None] * max_subset_size
    best_mses = [None] * max_subset_size
    for subset_size in range(1, max_subset_size + 1):
        best_subsets[subset_size-1], best_mses[subset_size-1] =\
            findBestSubsetFixedSize(X, y, subset_size)

    return([best_subsets, best_mses])
```

### Executing Best Subset Selection

We'll now use this function to perform our subset selection. At the very end, we will produce some metrics that will help us evaluate our selections:

- `adjr2s` = Adjusted R-Squared
- `bics` = Bayesian information Criteria
- `aics` = Akaike’s Information Criteria

```python
# Since the exhaustive search takes a really long time to complete,
# I recorded the result of this procedure
run_exhaustive_search = False
if run_exhaustive_search:
    # This procedure takes really long time to complete!
    best_subsets, best_mses = findBestSubset(X, y, 3)
else:
    best_subsets = [[11],
                    [1, 11],
                    [1, 11, 13],
                    [1, 11, 13, 17],
                    [0, 1, 11, 13, 17],
                    [0, 1, 5, 11, 13, 17],
                    [1, 5, 7, 8, 9, 13, 17],
                    [0, 1, 5, 9, 10, 12, 13, 17],
                    [0, 1, 5, 7, 10, 11, 12, 13, 17],
                    [0, 1, 5, 7, 10, 11, 12, 13, 14, 17],
                    [0, 1, 5, 7, 10, 11, 12, 13, 14, 16, 17],
                    [0, 1, 3, 5, 7, 10, 11, 12, 13, 14, 16, 17],
                    [0, 1, 3, 5, 7, 10, 11, 12, 13, 14, 15, 16, 17],
                    [0, 1, 2, 3, 5, 7, 10, 11, 12, 13, 14, 15, 16, 17],
                    [0, 1, 2, 3, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17],
                    [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17],
                    [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    best_mses = [137565.32036137575, 116526.84368963055, 111214.05648618752, 106353.04872933947,
                 103231.5567757093,   99600.39516195898,  98503.98289210549,  95577.68037627422,
                  94350.00527219362,  93157.42029558783,  92727.54772410596,  92521.79611890586,
                  92354.1742898915,   92200.22963038784,  92148.96332783563,  92088.88772977113,
                  92051.12835223945,  92022.19527998423,  92017.86901772919]
    
adjr2s = [None] * len(best_subsets)
bics = [None] * len(best_subsets)
aics = [None] * len(best_subsets)
for idx_set in range(len(best_subsets)):
    X_subset = X.iloc[:, best_subsets[idx_set]].values
    X_subset = sm.tools.tools.add_constant(X_subset)
    result = sm.OLS(y, X_subset).fit()
    adjr2s[idx_set] = result.rsquared_adj
    bics[idx_set] = result.bic
    aics[idx_set] = result.aic
```

### Plotting

As always, we want to create some plots so that we can better visaulize and understand what's going on. To do this, we'll first write a function that takes in our metrics: 

```python
def makePlotsForBestSubsets(adjr2s, bics, aics, mses):
    markerSize = 8
    titlesFontSize = 18
    axisLabelFontSize = 16
    
    subsetsNmb = len(adjr2s)
    xvals = range(1, subsetsNmb + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(wspace=0.45, hspace=0.35) 

    ax1.plot(xvals, adjr2s, '-o', markersize=markerSize)
    ax1.set_ylabel('Adjusted R2', fontsize=titlesFontSize)

    ax2.plot(xvals, bics, '-o', markersize=markerSize)
    ax2.set_ylabel('BIC', fontsize=titlesFontSize)

    ax3.plot(xvals, aics, '-o', markersize=markerSize)
    ax3.set_ylabel('AIC', fontsize=titlesFontSize)

    ax4.plot(xvals, mses, '-o', markersize=markerSize)
    ax4.set_ylabel('MSE', fontsize=titlesFontSize)

    for ax in fig.axes:
        ax.set_xlabel('Number of variables', fontsize=titlesFontSize)
        ax.set_xlim(0.5, subsetsNmb + 1)
        ax.set_xticks(range(2, subsetsNmb + 1, 2));
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(axisLabelFontSize) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(axisLabelFontSize)
            
makePlotsForBestSubsets(adjr2s, bics, aics, best_mses)
```

![bt](bt.svg)

## Forward and Backward Stepwise Selection

Here, we'll write functions for forward and backward stepwise selection. Forward and backward stepwise selection follows a very similar logic to our best subsect selection, but looks at a more restrictive set of models. 

In forward stepwise selection, we start with no predictors. In backward stepwise selection, we start with all the predictors. 

### Backward Selection

```python
def doBwdSelectStep(X, y, cur_subset):
    best_subset = []
    best_mse = -1
    for feature_idx in cur_subset:
        reduced_subset = list(set(cur_subset) - {feature_idx})
        X_subset = X.iloc[:, reduced_subset]
        lin_reg = LinearRegression(fit_intercept=True, normalize=False)
        lin_reg.fit(X_subset, y)
        yhat = lin_reg.predict(X_subset)
        mse_resid = mean_squared_error(y, yhat)
        if best_mse < 0 or mse_resid < best_mse:
            best_subset = reduced_subset
            best_mse = mse_resid
    return([best_subset, best_mse])

def doBwdStepwiseSelect(X, y, starting_set):
    steps_nmb = len(starting_set)
    best_subsets = [None] * steps_nmb
    best_mses = [None] * steps_nmb

    X_subset = X.iloc[:, starting_set]
    lin_reg = LinearRegression(fit_intercept=True, normalize=False)
    lin_reg.fit(X_subset, y)
    yhat = lin_reg.predict(X_subset)
    mse_resid = mean_squared_error(y, yhat)

    best_subsets[0] = starting_set
    best_mses[0] = mse_resid

    for step in range(steps_nmb - 1):
        best_subsets[step+1], best_mses[step+1] = doBwdSelectStep(X, y, best_subsets[step])
    return([best_subsets, best_mses])
```

### Forward Selection

```python
def doFwdSelectStep(X, y, cur_subset):
    features_nmb = X.shape[1]
    new_features = set(range(features_nmb)) - set(cur_subset)

    best_subset = []
    best_mse = -1
    for feature_idx in new_features:
        increased_subset = cur_subset + [feature_idx]
        X_subset = X.iloc[:, increased_subset]
        lin_reg = LinearRegression(fit_intercept=True, normalize=False)
        lin_reg.fit(X_subset, y)
        yhat = lin_reg.predict(X_subset)
        mse_resid = mean_squared_error(y, yhat)
        if best_mse < 0 or mse_resid < best_mse:
            best_subset = increased_subset
            best_mse = mse_resid
    return([best_subset, best_mse])

def doFwdStepwiseSelect(X, y, starting_set):
    features_nmb = X.shape[1]
    steps_nmb = features_nmb - len(starting_set)
    best_subsets = [None] * steps_nmb
    best_mses = [None] * steps_nmb
    prev_subset = starting_set
    for step in range(steps_nmb):
        best_subsets[step], best_mses[step] = doFwdSelectStep(X, y, prev_subset)
        prev_subset = best_subsets[step]
    return([best_subsets, best_mses])
```


### Executing Forward and Backward Stepwise Selection

```python
best_bwd_subsets, best_bwd_mses = doBwdStepwiseSelect(X, y, list(range(19)))
# Reverse the lists:
best_bwd_subsets = best_bwd_subsets[::-1]
best_bwd_mses = best_bwd_mses[::-1]
bwd_sets_nmb = len(best_bwd_subsets)
bwd_adjr2s = [None] * bwd_sets_nmb
bwd_bics = [None] * bwd_sets_nmb
bwd_aics = [None] * bwd_sets_nmb
for idx_set in range(bwd_sets_nmb):
    X_subset = X.iloc[:, best_bwd_subsets[idx_set]].values
    X_subset = sm.tools.tools.add_constant(X_subset)
    result = sm.OLS(y, X_subset).fit()
    bwd_adjr2s[idx_set] = result.rsquared_adj
    bwd_bics[idx_set] = result.bic
    bwd_aics[idx_set] = result.aic

best_fwd_subsets, best_fwd_mses = doFwdStepwiseSelect(X, y, [])
fwd_sets_nmb = len(best_fwd_subsets)
fwd_adjr2s = [None] * fwd_sets_nmb
fwd_bics = [None] * fwd_sets_nmb
fwd_aics = [None] * fwd_sets_nmb
for idx_set in range(fwd_sets_nmb):
    X_subset = X.iloc[:, best_fwd_subsets[idx_set]].values
    X_subset = sm.tools.tools.add_constant(X_subset)
    result = sm.OLS(y, X_subset).fit()
    fwd_adjr2s[idx_set] = result.rsquared_adj
    fwd_bics[idx_set] = result.bic
    fwd_aics[idx_set] = result.aic
```

### Plotting

Again, let's make plots for our subsets produced here, using the function we wrote earlier. 

#### Backward Subsets

```python
makePlotsForBestSubsets(bwd_adjr2s, bwd_bics, bwd_aics, best_bwd_mses)
```
![back](back.svg)

#### Forward Subsets

```python
makePlotsForBestSubsets(fwd_adjr2s, fwd_bics, fwd_aics, best_fwd_mses)
```

![for](for.svg)

