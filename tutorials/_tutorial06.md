---
layout: page
title: Classification
parent: Tutorials
nav_exclude: true
latex: true
---

# Classification
{:.no_toc}

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

In this tutorial, we will be exploring several classification techniques.

Much of this code was provided by Professor Kucheryavyy; I have broken the code down into a few smaller pieces and added some comments and explanations that should help your understanding. You can view the code for this tutorial [here]().


## Getting Started

### Importing Libraries

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
```

### Plot and Output Settings
We'll also introduce a few extra settings just to make the output of each of our cells a bit nicer:

```python
# Reset all styles to the default:
plt.rcParams.update(plt.rcParamsDefault)
# Then make graphs inline:
%matplotlib inline

# Useful function for Jupyter to display text in bold:
def displaybd(text):
    display(Markdown("**" + text + "**"))
```

If you would like your plots to be a bit larger, please use the following code:
```python
plt.rcParams['figure.figsize'] = (7, 6)
plt.rcParams['font.size'] = 24
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['figure.titlesize'] = 'large'
plt.rcParams['lines.markersize'] = 10
```

### Our Dataset
In this tutorial, we we will be using a dataset on the stock market, which can be downloaded [here](Smarket.csv).

```python

```