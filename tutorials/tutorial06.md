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

Much of this code was provided by Professor Kucheryavyy; I have broken the code down into a few smaller pieces and added some comments and explanations that should help your understanding. You can view the code for this tutorial [here](https://colab.research.google.com/drive/1hYevhj_tTUBQCcj8OGSyGduoOL8c8m4K).


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

In this tutorial, we we will be using a dataset on the stock market, which can be downloaded [here](Smarket.csv). This dataset is from *An Introduction to Statistical Learning, with applications in R* (Springer, 2013).

As usual, we can use [`read_csv`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) to create a pandas dataframe:

```python
smarket = pd.read_csv('Smarket.csv', parse_dates=False)
```

Note that this dataset contains a column `Direction`, which takes on two different values, either `Up` or `Down`. To make this column easier to work with in our regressions, we want to represent these values numerically. Let's have `Up` be `1` and `Down` be `0`. To do this, we can use [`np.where`](https://numpy.org/doc/stable/reference/generated/numpy.where.html):

```python
smarket["DirectionCode"] = np.where(smarket["Direction"].str.contains("Up"), 1, 0)
```

Now, let's get a bit more familiar with our data:

```python
display(smarket[1:10])
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Lag1</th>
      <th>Lag2</th>
      <th>Lag3</th>
      <th>Lag4</th>
      <th>Lag5</th>
      <th>Volume</th>
      <th>Today</th>
      <th>Direction</th>
      <th>DirectionCode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>0.959</td>
      <td>0.381</td>
      <td>-0.192</td>
      <td>-2.624</td>
      <td>-1.055</td>
      <td>1.2965</td>
      <td>1.032</td>
      <td>Up</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001</td>
      <td>1.032</td>
      <td>0.959</td>
      <td>0.381</td>
      <td>-0.192</td>
      <td>-2.624</td>
      <td>1.4112</td>
      <td>-0.623</td>
      <td>Down</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>-0.623</td>
      <td>1.032</td>
      <td>0.959</td>
      <td>0.381</td>
      <td>-0.192</td>
      <td>1.2760</td>
      <td>0.614</td>
      <td>Up</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001</td>
      <td>0.614</td>
      <td>-0.623</td>
      <td>1.032</td>
      <td>0.959</td>
      <td>0.381</td>
      <td>1.2057</td>
      <td>0.213</td>
      <td>Up</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2001</td>
      <td>0.213</td>
      <td>0.614</td>
      <td>-0.623</td>
      <td>1.032</td>
      <td>0.959</td>
      <td>1.3491</td>
      <td>1.392</td>
      <td>Up</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2001</td>
      <td>1.392</td>
      <td>0.213</td>
      <td>0.614</td>
      <td>-0.623</td>
      <td>1.032</td>
      <td>1.4450</td>
      <td>-0.403</td>
      <td>Down</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2001</td>
      <td>-0.403</td>
      <td>1.392</td>
      <td>0.213</td>
      <td>0.614</td>
      <td>-0.623</td>
      <td>1.4078</td>
      <td>0.027</td>
      <td>Up</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2001</td>
      <td>0.027</td>
      <td>-0.403</td>
      <td>1.392</td>
      <td>0.213</td>
      <td>0.614</td>
      <td>1.1640</td>
      <td>1.303</td>
      <td>Up</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2001</td>
      <td>1.303</td>
      <td>0.027</td>
      <td>-0.403</td>
      <td>1.392</td>
      <td>0.213</td>
      <td>1.2326</td>
      <td>0.287</td>
      <td>Up</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


```python
display(smarket.describe())
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Lag1</th>
      <th>Lag2</th>
      <th>Lag3</th>
      <th>Lag4</th>
      <th>Lag5</th>
      <th>Volume</th>
      <th>Today</th>
      <th>DirectionCode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1250.000000</td>
      <td>1250.000000</td>
      <td>1250.000000</td>
      <td>1250.000000</td>
      <td>1250.000000</td>
      <td>1250.00000</td>
      <td>1250.000000</td>
      <td>1250.000000</td>
      <td>1250.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2003.016000</td>
      <td>0.003834</td>
      <td>0.003919</td>
      <td>0.001716</td>
      <td>0.001636</td>
      <td>0.00561</td>
      <td>1.478305</td>
      <td>0.003138</td>
      <td>0.518400</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.409018</td>
      <td>1.136299</td>
      <td>1.136280</td>
      <td>1.138703</td>
      <td>1.138774</td>
      <td>1.14755</td>
      <td>0.360357</td>
      <td>1.136334</td>
      <td>0.499861</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2001.000000</td>
      <td>-4.922000</td>
      <td>-4.922000</td>
      <td>-4.922000</td>
      <td>-4.922000</td>
      <td>-4.92200</td>
      <td>0.356070</td>
      <td>-4.922000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2002.000000</td>
      <td>-0.639500</td>
      <td>-0.639500</td>
      <td>-0.640000</td>
      <td>-0.640000</td>
      <td>-0.64000</td>
      <td>1.257400</td>
      <td>-0.639500</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2003.000000</td>
      <td>0.039000</td>
      <td>0.039000</td>
      <td>0.038500</td>
      <td>0.038500</td>
      <td>0.03850</td>
      <td>1.422950</td>
      <td>0.038500</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2004.000000</td>
      <td>0.596750</td>
      <td>0.596750</td>
      <td>0.596750</td>
      <td>0.596750</td>
      <td>0.59700</td>
      <td>1.641675</td>
      <td>0.596750</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2005.000000</td>
      <td>5.733000</td>
      <td>5.733000</td>
      <td>5.733000</td>
      <td>5.733000</td>
      <td>5.73300</td>
      <td>3.152470</td>
      <td>5.733000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


```python
displaybd("Correlations matrix:")
display(smarket.corr())
```

**Correlations matrix:**

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Lag1</th>
      <th>Lag2</th>
      <th>Lag3</th>
      <th>Lag4</th>
      <th>Lag5</th>
      <th>Volume</th>
      <th>Today</th>
      <th>DirectionCode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Year</th>
      <td>1.000000</td>
      <td>0.029700</td>
      <td>0.030596</td>
      <td>0.033195</td>
      <td>0.035689</td>
      <td>0.029788</td>
      <td>0.539006</td>
      <td>0.030095</td>
      <td>0.074608</td>
    </tr>
    <tr>
      <th>Lag1</th>
      <td>0.029700</td>
      <td>1.000000</td>
      <td>-0.026294</td>
      <td>-0.010803</td>
      <td>-0.002986</td>
      <td>-0.005675</td>
      <td>0.040910</td>
      <td>-0.026155</td>
      <td>-0.039757</td>
    </tr>
    <tr>
      <th>Lag2</th>
      <td>0.030596</td>
      <td>-0.026294</td>
      <td>1.000000</td>
      <td>-0.025897</td>
      <td>-0.010854</td>
      <td>-0.003558</td>
      <td>-0.043383</td>
      <td>-0.010250</td>
      <td>-0.024081</td>
    </tr>
    <tr>
      <th>Lag3</th>
      <td>0.033195</td>
      <td>-0.010803</td>
      <td>-0.025897</td>
      <td>1.000000</td>
      <td>-0.024051</td>
      <td>-0.018808</td>
      <td>-0.041824</td>
      <td>-0.002448</td>
      <td>0.006132</td>
    </tr>
    <tr>
      <th>Lag4</th>
      <td>0.035689</td>
      <td>-0.002986</td>
      <td>-0.010854</td>
      <td>-0.024051</td>
      <td>1.000000</td>
      <td>-0.027084</td>
      <td>-0.048414</td>
      <td>-0.006900</td>
      <td>0.004215</td>
    </tr>
    <tr>
      <th>Lag5</th>
      <td>0.029788</td>
      <td>-0.005675</td>
      <td>-0.003558</td>
      <td>-0.018808</td>
      <td>-0.027084</td>
      <td>1.000000</td>
      <td>-0.022002</td>
      <td>-0.034860</td>
      <td>0.005423</td>
    </tr>
    <tr>
      <th>Volume</th>
      <td>0.539006</td>
      <td>0.040910</td>
      <td>-0.043383</td>
      <td>-0.041824</td>
      <td>-0.048414</td>
      <td>-0.022002</td>
      <td>1.000000</td>
      <td>0.014592</td>
      <td>0.022951</td>
    </tr>
    <tr>
      <th>Today</th>
      <td>0.030095</td>
      <td>-0.026155</td>
      <td>-0.010250</td>
      <td>-0.002448</td>
      <td>-0.006900</td>
      <td>-0.034860</td>
      <td>0.014592</td>
      <td>1.000000</td>
      <td>0.730563</td>
    </tr>
    <tr>
      <th>DirectionCode</th>
      <td>0.074608</td>
      <td>-0.039757</td>
      <td>-0.024081</td>
      <td>0.006132</td>
      <td>0.004215</td>
      <td>0.005423</td>
      <td>0.022951</td>
      <td>0.730563</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

***

```python
smarket["Volume"].plot()
plt.xlabel("Day");
plt.ylabel("Volume");
```

![smarket](smarket.svg)


## Logit

### Running Logit via GLM

 A generalized linear model usually refers to a model in which the dependent variable \\(y\\) follows some non-normal distribution with a mean \\(\mu\\) that is assumed to be some (often nonlinear) function of the independent variable \\(x\\). Note that [*generalized linear models*](https://en.wikipedia.org/wiki/Generalized_linear_model#:~:text=In%20statistics%2C%20the%20generalized%20linear,other%20than%20a%20normal%20distribution.) are ***different*** from [*general linear models*](https://en.wikipedia.org/wiki/General_linear_model). We will use the [generalized linear models from the statsmodels package](https://www.statsmodels.org/stable/glm.html) to run [logit](https://en.wikipedia.org/wiki/Logit):


```python
model = smf.glm("DirectionCode~Lag1+Lag2+Lag3+Lag4+Lag5+Volume", data=smarket,
                family=sm.families.Binomial())
res = model.fit()
display(res.summary())
```


<table class="simpletable">
<caption>Generalized Linear Model Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>DirectionCode</td>  <th>  No. Observations:  </th>  <td>  1250</td> 
</tr>
<tr>
  <th>Model:</th>                  <td>GLM</td>       <th>  Df Residuals:      </th>  <td>  1243</td> 
</tr>
<tr>
  <th>Model Family:</th>        <td>Binomial</td>     <th>  Df Model:          </th>  <td>     6</td> 
</tr>
<tr>
  <th>Link Function:</th>         <td>logit</td>      <th>  Scale:             </th> <td>  1.0000</td>
</tr>
<tr>
  <th>Method:</th>                <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -863.79</td>
</tr>
<tr>
  <th>Date:</th>            <td>Sat, 27 Jun 2020</td> <th>  Deviance:          </th> <td>  1727.6</td>
</tr>
<tr>
  <th>Time:</th>                <td>20:46:14</td>     <th>  Pearson chi2:      </th> <td>1.25e+03</td>
</tr>
<tr>
  <th>No. Iterations:</th>          <td>4</td>        <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   -0.1260</td> <td>    0.241</td> <td>   -0.523</td> <td> 0.601</td> <td>   -0.598</td> <td>    0.346</td>
</tr>
<tr>
  <th>Lag1</th>      <td>   -0.0731</td> <td>    0.050</td> <td>   -1.457</td> <td> 0.145</td> <td>   -0.171</td> <td>    0.025</td>
</tr>
<tr>
  <th>Lag2</th>      <td>   -0.0423</td> <td>    0.050</td> <td>   -0.845</td> <td> 0.398</td> <td>   -0.140</td> <td>    0.056</td>
</tr>
<tr>
  <th>Lag3</th>      <td>    0.0111</td> <td>    0.050</td> <td>    0.222</td> <td> 0.824</td> <td>   -0.087</td> <td>    0.109</td>
</tr>
<tr>
  <th>Lag4</th>      <td>    0.0094</td> <td>    0.050</td> <td>    0.187</td> <td> 0.851</td> <td>   -0.089</td> <td>    0.107</td>
</tr>
<tr>
  <th>Lag5</th>      <td>    0.0103</td> <td>    0.050</td> <td>    0.208</td> <td> 0.835</td> <td>   -0.087</td> <td>    0.107</td>
</tr>
<tr>
  <th>Volume</th>    <td>    0.1354</td> <td>    0.158</td> <td>    0.855</td> <td> 0.392</td> <td>   -0.175</td> <td>    0.446</td>
</tr>
</table>


### Predicted Probabilities and Confusion Matrix


```python
displaybd("Predicted probabilities for the first observations:")
DirectionProbs = res.predict()
print(DirectionProbs[0:10])

DirectionHat = np.where(DirectionProbs > 0.5, "Up", "Down")
confusionDF = pd.crosstab(DirectionHat, smarket["Direction"],
                          rownames=['Predicted'], colnames=['Actual'],
                          margins=True)
display(Markdown("***"))
displaybd("Confusion matrix:")
display(confusionDF)

displaybd("Share of correctly predicted market movements:")
print(np.mean(smarket['Direction'] == DirectionHat))
```


**Predicted probabilities for the first observations:**


    [0.50708413 0.48146788 0.48113883 0.51522236 0.51078116 0.50695646
     0.49265087 0.50922916 0.51761353 0.48883778]



***



**Confusion matrix:**



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Actual</th>
      <th>Down</th>
      <th>Up</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Predicted</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Down</th>
      <td>145</td>
      <td>141</td>
      <td>286</td>
    </tr>
    <tr>
      <th>Up</th>
      <td>457</td>
      <td>507</td>
      <td>964</td>
    </tr>
    <tr>
      <th>All</th>
      <td>602</td>
      <td>648</td>
      <td>1250</td>
    </tr>
  </tbody>
</table>
</div>



**Share of correctly predicted market movements:**


    0.5216


### Estimation of Test Error

Here, we'll first train a model on the data from before 2005, and then test it on the data from after 2005. 

```python
train = (smarket['Year'] < 2005)
smarket2005 = smarket[~train]
displaybd("Dimensions of the validation set:")
print(smarket2005.shape)

model = smf.glm("DirectionCode~Lag1+Lag2+Lag3+Lag4+Lag5+Volume", data=smarket,
                family=sm.families.Binomial(), subset=train)
res = model.fit()

DirectionProbsTets = res.predict(smarket2005)
DirectionTestHat = np.where(DirectionProbsTets > 0.5, "Up", "Down")
displaybd("Share of correctly predicted market movements in 2005:")
print(np.mean(smarket2005['Direction'] == DirectionTestHat))
```


**Dimensions of the validation set:**


    (252, 10)



**Share of correctly predicted market movements in 2005:**


    0.4801587301587302


## Linear Discriminant Analysis

[Linear discriminant analysis](https://en.wikipedia.org/wiki/Linear_discriminant_analysis) is a robust classification method that relies on the following assumptions:
- the class conditional distributions are Gaussian
- these Gaussians have the same covariance matrix (assume homoskedasticity)

<!-- Another way of 
- the independent variable \\(x\\) comes from a Gaussian distribution
- the dependent variable \\(y\\) is discrete -->

Without these assupmtions, linear discriminant analysis is a form of dimenstionality reduction, so it is especially well-suited for high-dimensional data. Thus, we would want to use linear discriminant analysis when we want to reduce the number of features (reduce the dimensionality) while preserving the distinction between our classes.

### Custom Output Functions
Before getting started with linear discriminat analysis, we'll write a couple of our own functions that'll help display some of our calculations nicely: 

```python
def printPriorProbabilities(ldaClasses, ldaPriors):
    priorsDF = pd.DataFrame()
    for cIdx, cName in enumerate(ldaClasses):
        priorsDF[cName] = [ldaPriors[cIdx]];
    displaybd('Prior probablities of groups:')
    display(Markdown(priorsDF.to_html(index=False)))
    
def printGroupMeans(ldaClasses, featuresNames, ldaGroupMeans):
    displaybd("Group means:")
    groupMeansDF = pd.DataFrame(index=ldaClasses)
    for fIdx, fName in enumerate(featuresNames):
         groupMeansDF[fName] = ldaGroupMeans[:, fIdx]
    display(groupMeansDF)

def printLDACoeffs(featuresNames, ldaCoeffs):
    coeffDF = pd.DataFrame(index=featuresNames)
    for cIdx in range(ldaCoeffs.shape[0]):
        colName = "LDA" + str(cIdx + 1)
        coeffDF[colName] = ldaCoeffs[cIdx]
    displaybd("Coefficients of linear discriminants:")
    display(coeffDF)
```

### Fitting an LDA Model

Here, we'll be using [scikit-learn's `Linear Discriminant Analysis` class](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html) to fit our model:

```python
outcomeName = 'Direction'
featuresNames = ['Lag1', 'Lag2'];

X_train = smarket.loc[train, featuresNames]
y_train = smarket.loc[train, outcomeName]

lda = LinearDiscriminantAnalysis()
ldaFit = lda.fit(X_train, y_train);

printPriorProbabilities(ldaFit.classes_, ldaFit.priors_)
printGroupMeans(ldaFit.classes_, featuresNames, ldaFit.means_)
printLDACoeffs(featuresNames, ldaFit.coef_)
# Coefficients calcualted by Python's LDA are different from R's LDA
# But they are proportional:
printLDACoeffs(featuresNames, 11.580267503964166 * ldaFit.coef_)
# See this: https://stats.stackexchange.com/questions/87479/what-are-coefficients-of-linear-discriminants-in-lda
```


**Prior probablities of groups:**



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Down</th>
      <th>Up</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0.491984</td>
      <td>0.508016</td>
    </tr>
  </tbody>
</table>



**Group means:**



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Lag1</th>
      <th>Lag2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Down</th>
      <td>0.042790</td>
      <td>0.033894</td>
    </tr>
    <tr>
      <th>Up</th>
      <td>-0.039546</td>
      <td>-0.031325</td>
    </tr>
  </tbody>
</table>
</div>



**Coefficients of linear discriminants:**



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LDA1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Lag1</th>
      <td>-0.055441</td>
    </tr>
    <tr>
      <th>Lag2</th>
      <td>-0.044345</td>
    </tr>
  </tbody>
</table>
</div>



**Coefficients of linear discriminants:**



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LDA1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Lag1</th>
      <td>-0.642019</td>
    </tr>
    <tr>
      <th>Lag2</th>
      <td>-0.513529</td>
    </tr>
  </tbody>
</table>
</div>


### LDA Predictions


```python
X_test = smarket2005.loc[~train, featuresNames]
y_test = smarket.loc[~train, outcomeName]
y_hat = ldaFit.predict(X_test)

confusionDF = pd.crosstab(y_hat, y_test,
                          rownames=['Predicted'], colnames=['Actual'],
                          margins=True)
displaybd("Confusion matrix:")
display(confusionDF)

displaybd("Share of correctly predicted market movements:")
print(np.mean(y_test == y_hat))
```


**Confusion matrix:**



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Actual</th>
      <th>Down</th>
      <th>Up</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Predicted</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Down</th>
      <td>35</td>
      <td>35</td>
      <td>70</td>
    </tr>
    <tr>
      <th>Up</th>
      <td>76</td>
      <td>106</td>
      <td>182</td>
    </tr>
    <tr>
      <th>All</th>
      <td>111</td>
      <td>141</td>
      <td>252</td>
    </tr>
  </tbody>
</table>
</div>



**Share of correctly predicted market movements:**


    0.5595238095238095


### Posterior Probabilities
Here, we'll estimate [posterior propbabilities](https://en.wikipedia.org/wiki/Posterior_probability), using [scikit-learn's `predict_proba` function](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis.predict_proba):

```python
pred_p = lda.predict_proba(X_test)
# pred_p is an array of shape (number of observations) x (number of classes)

upNmb = np.sum(pred_p[:, 1] > 0.5)
displaybd("Number of upward movements with threshold 0.5: " + str(upNmb))

upNmb = np.sum(pred_p[:, 1] > 0.9)
displaybd("Number of upward movements with threshold 0.9: " + str(upNmb))
```


**Number of upward movements with threshold 0.5: 182**



**Number of upward movements with threshold 0.9: 0**


## Quadratic Discriminant Analysis
[Quadratic discriminant analysis](https://en.wikipedia.org/wiki/Quadratic_classifier#Quadratic_discriminant_analysis) is a generalization of linear discriminant analysis as a classifier, but it does not make the same covariance assumption. 

### Fitting a QDA Model

Here, we'll be using [scikit-learn's `Quadratic Discriminant Analysis` class](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html) to fit our model:

```python
qda = QuadraticDiscriminantAnalysis()
qdaFit = qda.fit(X_train, y_train);
printPriorProbabilities(qdaFit.classes_, qdaFit.priors_)
printGroupMeans(qdaFit.classes_, featuresNames, qdaFit.means_)
```


**Prior probablities of groups:**



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Down</th>
      <th>Up</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0.491984</td>
      <td>0.508016</td>
    </tr>
  </tbody>
</table>



**Group means:**



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Lag1</th>
      <th>Lag2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Down</th>
      <td>0.042790</td>
      <td>0.033894</td>
    </tr>
    <tr>
      <th>Up</th>
      <td>-0.039546</td>
      <td>-0.031325</td>
    </tr>
  </tbody>
</table>
</div>


### QDA Predictions

```python
y_hat = qdaFit.predict(X_test)
confusionDF = pd.crosstab(y_hat, y_test,
                          rownames=['Predicted'], colnames=['Actual'],
                          margins=True)
displaybd("Confusion matrix:")
display(confusionDF)
displaybd("Share of correctly predicted market movements:")
print(np.mean(y_test == y_hat))
```


**Confusion matrix:**



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Actual</th>
      <th>Down</th>
      <th>Up</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Predicted</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Down</th>
      <td>30</td>
      <td>20</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Up</th>
      <td>81</td>
      <td>121</td>
      <td>202</td>
    </tr>
    <tr>
      <th>All</th>
      <td>111</td>
      <td>141</td>
      <td>252</td>
    </tr>
  </tbody>
</table>
</div>



**Share of correctly predicted market movements:**


    0.5992063492063492


## k-Nearest Neighbors

Here, we'll be looking at k-nearest neighbors, which we talked about in [lecture 02 of this course](https://piazza.com/class_profile/get_resource/k8pcxfiwkxf2ec/k8zhpvig7ko5hs). [Tutorial 02](../tutorial02) was also on k-nearest neighbors classification, so please refer to that tutorial for an additional examples and explanations. 

### One Neighbor

```python
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
y_hat = knn.fit(X_train, y_train).predict(X_test)
confusionDF = pd.crosstab(y_hat, y_test,
                          rownames=['Predicted'], colnames=['Actual'],
                          margins=True)
displaybd("Confusion matrix:")
display(confusionDF)
displaybd("Share of correctly predicted market movements:")
print(np.mean(y_test == y_hat))
```


**Confusion matrix:**

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Actual</th>
      <th>Down</th>
      <th>Up</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Predicted</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Down</th>
      <td>43</td>
      <td>58</td>
      <td>101</td>
    </tr>
    <tr>
      <th>Up</th>
      <td>68</td>
      <td>83</td>
      <td>151</td>
    </tr>
    <tr>
      <th>All</th>
      <td>111</td>
      <td>141</td>
      <td>252</td>
    </tr>
  </tbody>
</table>
</div>



**Share of correctly predicted market movements:**


    0.5


### Three Neighbors

```python
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
y_hat = knn.fit(X_train, y_train).predict(X_test)
confusionDF = pd.crosstab(y_hat, y_test,
                          rownames=['Predicted'], colnames=['Actual'],
                          margins=True)
displaybd("Confusion matrix:")
display(confusionDF)
displaybd("Share of correctly predicted market movements:")
print(np.mean(y_test == y_hat))
```


**Confusion matrix:**



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Actual</th>
      <th>Down</th>
      <th>Up</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Predicted</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Down</th>
      <td>48</td>
      <td>55</td>
      <td>103</td>
    </tr>
    <tr>
      <th>Up</th>
      <td>63</td>
      <td>86</td>
      <td>149</td>
    </tr>
    <tr>
      <th>All</th>
      <td>111</td>
      <td>141</td>
      <td>252</td>
    </tr>
  </tbody>
</table>
</div>



**Share of correctly predicted market movements:**


    0.5317460317460317


## An Application to Caravan Insurance Data

This section will demonstrate the use of two techniques we learned above, [KNN](#k-nearest-neighbors) and [logit](#logit).

### A New Dataset
We'll be using a [new dataset](Caravan.csv) that contains information on customers of an insurance company. You can see a detailed description of this dataset [here](https://www.kaggle.com/uciml/caravan-insurance-challenge).

#### Loading Our Dataset


```python
caravan = pd.read_csv('Caravan.csv', index_col=0)

display(caravan.describe())
display(caravan.describe(include=[np.object]))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MOSTYPE</th>
      <th>MAANTHUI</th>
      <th>MGEMOMV</th>
      <th>MGEMLEEF</th>
      <th>MOSHOOFD</th>
      <th>MGODRK</th>
      <th>MGODPR</th>
      <th>MGODOV</th>
      <th>MGODGE</th>
      <th>MRELGE</th>
      <th>...</th>
      <th>ALEVEN</th>
      <th>APERSONG</th>
      <th>AGEZONG</th>
      <th>AWAOREG</th>
      <th>ABRAND</th>
      <th>AZEILPL</th>
      <th>APLEZIER</th>
      <th>AFIETS</th>
      <th>AINBOED</th>
      <th>ABYSTAND</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5822.000000</td>
      <td>5822.000000</td>
      <td>5822.000000</td>
      <td>5822.000000</td>
      <td>5822.000000</td>
      <td>5822.000000</td>
      <td>5822.000000</td>
      <td>5822.000000</td>
      <td>5822.000000</td>
      <td>5822.000000</td>
      <td>...</td>
      <td>5822.000000</td>
      <td>5822.000000</td>
      <td>5822.000000</td>
      <td>5822.000000</td>
      <td>5822.000000</td>
      <td>5822.000000</td>
      <td>5822.000000</td>
      <td>5822.000000</td>
      <td>5822.000000</td>
      <td>5822.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>24.253349</td>
      <td>1.110615</td>
      <td>2.678805</td>
      <td>2.991240</td>
      <td>5.773617</td>
      <td>0.696496</td>
      <td>4.626932</td>
      <td>1.069907</td>
      <td>3.258502</td>
      <td>6.183442</td>
      <td>...</td>
      <td>0.076606</td>
      <td>0.005325</td>
      <td>0.006527</td>
      <td>0.004638</td>
      <td>0.570079</td>
      <td>0.000515</td>
      <td>0.006012</td>
      <td>0.031776</td>
      <td>0.007901</td>
      <td>0.014256</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12.846706</td>
      <td>0.405842</td>
      <td>0.789835</td>
      <td>0.814589</td>
      <td>2.856760</td>
      <td>1.003234</td>
      <td>1.715843</td>
      <td>1.017503</td>
      <td>1.597647</td>
      <td>1.909482</td>
      <td>...</td>
      <td>0.377569</td>
      <td>0.072782</td>
      <td>0.080532</td>
      <td>0.077403</td>
      <td>0.562058</td>
      <td>0.022696</td>
      <td>0.081632</td>
      <td>0.210986</td>
      <td>0.090463</td>
      <td>0.119996</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>10.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>5.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>35.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>7.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>41.000000</td>
      <td>10.000000</td>
      <td>5.000000</td>
      <td>6.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>9.000000</td>
      <td>5.000000</td>
      <td>9.000000</td>
      <td>9.000000</td>
      <td>...</td>
      <td>8.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 85 columns</p>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Purchase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5822</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>No</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>5474</td>
    </tr>
  </tbody>
</table>
</div>


#### Standardizing Our Data


```python
y = caravan.Purchase
X = caravan.drop('Purchase', axis=1).astype('float64')
X_scaled = preprocessing.scale(X)
```

#### Splitting Data into Train and Test Data


```python
X_train = X_scaled[1000:,:]
y_train = y[1000:]
X_test = X_scaled[:1000,:]
y_test = y[:1000]
```

### Using KNN for Prediction


```python
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
y_hat = knn.fit(X_train, y_train).predict(X_test)
confusionDF = pd.crosstab(y_hat, y_test,
                          rownames=['Predicted'], colnames=['Actual'],
                          margins=True)
displaybd("Confusion matrix:")
display(confusionDF)
displaybd("Share of correctly predicted purchases:")
print(np.mean(y_test == y_hat))
```


**Confusion matrix:**



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Actual</th>
      <th>No</th>
      <th>Yes</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Predicted</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No</th>
      <td>873</td>
      <td>50</td>
      <td>923</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>68</td>
      <td>9</td>
      <td>77</td>
    </tr>
    <tr>
      <th>All</th>
      <td>941</td>
      <td>59</td>
      <td>1000</td>
    </tr>
  </tbody>
</table>
</div>



**Share of correctly predicted purchases:**


    0.882


### Logit

```python
X_train_w_constant = sm.add_constant(X_train)
X_test_w_constant = sm.add_constant(X_test, has_constant='add')

y_train_code = np.where(y_train == "No", 0, 1)

res = sm.GLM(y_train_code, X_train_w_constant, family=sm.families.Binomial()).fit()
y_hat_code = res.predict(X_test_w_constant)
PurchaseHat = np.where(y_hat_code > 0.25, "Yes", "No")

confusionDF = pd.crosstab(PurchaseHat, y_test,
                          rownames=['Predicted'], colnames=['Actual'],
                          margins=True)
displaybd("Confusion matrix:")
display(confusionDF)
```


**Confusion matrix:**



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Actual</th>
      <th>No</th>
      <th>Yes</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Predicted</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No</th>
      <td>919</td>
      <td>48</td>
      <td>967</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>22</td>
      <td>11</td>
      <td>33</td>
    </tr>
    <tr>
      <th>All</th>
      <td>941</td>
      <td>59</td>
      <td>1000</td>
    </tr>
  </tbody>
</table>
</div>

## More Iris Classification
Here, we will apply some of the new techniques we learned above to the iris classification problem we explored using k-nearest neighbors in [Tutorial 02](../tutorial02). 

### Our Dataset
I've included some of the important descriptions from Tutorial 02 in this tutorial as well, but please review tutorial 02 for more details on how we initially set up and process our dataset.   
As a reminder, we are using the [iris data set](https://archive.ics.uci.edu/ml/datasets/Iris) from the University of California, Irvine and are attempting to classify types of irises using the following four attributes:

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

There are three types of irises: 
1. Iris Setosa
2. Iris Versicolor
3. Iris Virginica

#### Importing Our Dataset

Let's import the data set as a `pandas` dataframe:

```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'type']
iris_df = pd.read_csv(url, names=names)
```

#### Splitting Data into Train and Test Data

Let's split our data into 80% training data and 20% testing data. We can do this using [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) and its `train_size` parameter:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)
```

#### Feature Scaling
Now, we want to perform some [feature scaling](https://en.wikipedia.org/wiki/Feature_scaling) to normalize the range of our independent variables. 

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

### Logit
Let's first take a look at how we might apply logit to our iris classification problem. You may apply logit using the techniques we learned above (using GLM), but I will show you one other method we can employ using [scikit-learn's `Logistic Regression` class](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), as we can consider logit and logistic regression to be [the same thing](https://stats.idre.ucla.edu/r/dae/logit-regression/)

#### Fitting our Model

Let's import the `Logistic Regression` class and fit our model as follows:

```python
from sklearn.linear_model import LogisticRegression
logit_model = LogisticRegression()
logit_model.fit(X_train, y_train)
```

#### Making Predictions
Then, we'll make some predictions and store them in a variable called `y_pred`:

```python
y_pred = logit_model.predict(X_test)
```

#### Evaluating our Predictions
Like we did in [Tutorial 02](../tutorial02), let's make a classification report and confusion matrix. 

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

|                     | **precision** | **recall** | **f1-score** | **support** |
|---------------------|:-------------:|:----------:|:------------:|:-----------:|
|     **Iris-setosa** |      1.00     |    1.00    |     1.00     |     9     |
| **Iris-versicolor** |      1.00    |    0.70   |     0.82    |      10   |
|  **Iris-virginica** |      0.79     |    1.00    |     0.88     |      11      |


```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm,
                     index = ['setosa','versicolor','virginica'], 
                     columns = ['setosa','versicolor','virginica'])

sns.heatmap(cm_df, annot=True)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

![logitheat](logitheat.svg)


Using this heat map, we can make the following observations:

1. All setosa flowers were correctly classified by our model. 
2. Seven versicolor flowers were correctly classified, and three versicolor flowers were incorrectly classified as virginica flowers.
3. All virginica flowers were correctly classified by our model. 

Again, you may not get the same exact classification report or confusion matrix, but this is normal, as your results will vary each time you run your model.


### Linear Discriminant Analysis

Let's now try using linear discriminant analysis for our classification. 

#### Fitting our Model

Again, let's use the `Linear Discriminant Analysis` class to fit our model:

```python
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)
```

#### Making Predictions
Then, we'll make some predictions and store them in a variable called `y_pred`:

```python
y_pred = lda_model.predict(X_test)
```

#### Evaluating our Predictions
Like we did in [Tutorial 02](../tutorial02), let's make a classification report and confusion matrix. If you want, you can also use the functions `printPriorProbabilities()`,  `printGroupMeans()`, and `printLDACoeffs()` that we wrote earlier, but here I'll keep it simple and just look at our classification report and heatmap like we did just earlier. 

```python
print(classification_report(y_test, y_pred))
```

|                     | **precision** | **recall** | **f1-score** | **support** |
|---------------------|:-------------:|:----------:|:------------:|:-----------:|
|     **Iris-setosa** |      1.00     |    1.00    |     1.00     |     9     |
| **Iris-versicolor** |      1.00    |    1.00   |     1.00    |      10   |
|  **Iris-virginica** |      1.00     |    1.00    |     1.00     |      11      |


In thise case, we can see our model did very well. Let's also take a look at the heatmap to see that a little bit more easily: 

```python
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm,
                     index = ['setosa','versicolor','virginica'], 
                     columns = ['setosa','versicolor','virginica'])

sns.heatmap(cm_df, annot=True)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

![ldaheat](ldaheat.svg)


Using this heat map, we can make the following observations:

1. All setosa flowers were correctly classified by our model. 
2. All versicolors were correctly classified by our model.
3. All virginica flowers were correctly classified by our model. 

Again, you may not get the same exact classification report or confusion matrix, but this is normal, as your results will vary each time you run your model.


### Quadratic Discriminant Analysis

Let's now try using quadratic discriminant analysis for our classification. 

#### Fitting our Model

Again, let's use the `Linear Discriminant Analysis` class to fit our model:

```python
qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(X_train, y_train)
```

#### Making Predictions
Then, we'll make some predictions and store them in a variable called `y_pred`:

```python
y_pred = qda_model.predict(X_test)
```

#### Evaluating our Predictions
Like we did in [Tutorial 02](../tutorial02), let's make a classification report and confusion matrix. If you want, you can also use the functions `printPriorProbabilities()` and `printGroupMeans()` that we wrote earlier, but here I'll keep it simple and just look at our classification report and heatmap like we did just earlier. 

```python
print(classification_report(y_test, y_pred))
```

|                     | **precision** | **recall** | **f1-score** | **support** |
|---------------------|:-------------:|:----------:|:------------:|:-----------:|
|     **Iris-setosa** |      1.00     |    1.00    |     1.00     |     9     |
| **Iris-versicolor** |      1.00    |    0.80   |     0.89    |      10   |
|  **Iris-virginica** |      0.85     |    1.00    |     0.92    |      11      |


```python
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm,
                     index = ['setosa','versicolor','virginica'], 
                     columns = ['setosa','versicolor','virginica'])

sns.heatmap(cm_df, annot=True)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

![qdaheat](qdaheat.svg)


Using this heat map, we can make the following observations:

1. All setosa flowers were correctly classified by our model. 
2. Eight versicolor flowers were correctly classified, and two versicolor flowers were incorrectly classified as virginica flowers.
3. All virginica flowers were correctly classified by our model. 

Again, you may not get the same exact classification report or confusion matrix, but this is normal, as your results will vary each time you run your model.









