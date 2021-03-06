---
layout: page
title: Bootstrap
parent: Tutorials
nav_exclude: true
latex: true
---

# Bootstrapping for Linear Regression
{:.no_toc}

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

You can view the code for this tutorial [here](https://colab.research.google.com/drive/1Nfz2tunq7jZjQVLptUHAT2izQ7WXmN70).

In Homework 02, you fitted a linear regression model for your data, which may have looked something like this:

\\[f_{\hat{\theta}}(x) = \hat{\theta_0} + \hat{\theta_1}x_1 + \cdots + \hat{\theta_p}x_p\\]

Now, we would like to construct conﬁdence intervals for the estimated coefﬁcients \\(\hat{\theta_0}, \hat{\theta_1}, \cdots \hat{\theta_p} \\) and perhaps infer the true coefficients of our model. Bootstrap is a computational method that allows us to calculate standard errors and confidence intervals for our parameters.

## Importing Libraries

As usual, we will want to use `numpy`, `pandas`, `matplot`, and `seaborn` to help us manipulate and visualize our data. 

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

## Our Data Set

In this example, I'll be estimating the price of houses in [King County, Washington](https://en.wikipedia.org/wiki/King_County,_Washington). 

You can obtain the dataset [here](https://www.kaggle.com/harlfoxem/housesalesprediction).

There are 21 columns in this dataset, but for the purposes of this tutorial, we will just look at a couple. Let's look at the following parameters to estimate price:
- `sqft_living`: the size, in square feet, of the living area of the house
- `yr_built`: the year in which the house was built

After downloading, read our data using something like the following:

```python
df = pd.read_csv("kc_house_data.csv", usecols=["price", "sqft_living", "yr_built"])
```

## Visualizing Our Data

Before doing anything, let's get a little bit more familiar with our data. As we learned in Tutorial 02, first use `head()` to see the first five rows of our dataframe: 

```python
df.head()
```


|       | price | sqft_living | yr_built |
|:-----:|:------------:|:-----------:|:------------:|
| **0** |      221900.0    |     1180     |      1955     |
| **1** |      538000.0     |     2570     |      1951    |
| **2** |      180000.0    |     770     |      1933    |
| **3** |      604000.0     |     1960     |      1965     |
| **4** |      510000.0     |     1680     |      1987    | 

Instead of looking at the year built, we'll look at the age of the house:

```python
df["age of house"] = 2015 - df["yr_built"]
```

|       | price | sqft_living | yr_built | age of house |
|:-----:|:------------:|:-----------:|:------------:|:------------:|
| **0** |      221900.0    |     1180     |      1955     | 60 |
| **1** |      538000.0     |     2570     |      1951    | 64| 
| **2** |      180000.0    |     770     |      1933    | 82 | 
| **3** |      604000.0     |     1960     |      1965     | 50 |
| **4** |      510000.0     |     1680     |      1987    | 28 |


In reality, there's also a column `yr_renovated` in the original dataset, but for simplicity, we will ignore this in this tutorial. 

Let's also create scatter plots to visualize the relationships between our variables:

```python
plt.xlabel("sqft_living")
plt.ylabel("price")
plt.scatter(x=df["sqft_living"], y=df["price"])
```

![sqftbyprice](sqftbyprice.svg)


```python
plt.xlabel("age of house")
plt.ylabel("price")
plt.scatter(x=df["age of house"], y=df["price"])
```

![ageprice](ageprice.svg)


## Fitting the Model

First, let's organize our data into X and y, what we're trying to predict (price, in our case).

```python
X = df.loc[:, ["sqft_living", "age of house"]]
y = df.loc[:, "price"]
```

> Note: 
> Visually, we can see that the age of the home does not seem to be strongly correlated with the price of the home. In a real data analysis, we may want to investigate further and/or remove this from our regression. 


Now, let's use `scikit-learn` to perform linear regression (there's no need to write the code for this on our own):

```python
import sklearn.linear_model as lm

linear_model = lm.LinearRegression()
linear_model.fit(X, y)

print("""
intercept: %.2f
sqft_living: %.2f
age of house: %.2f
""" % (tuple([linear_model.intercept_]) + tuple(linear_model.coef_)))
```

> intercept: -196929.07

> sqft_living:    304.57

> age of house:  2353.73

Above are the (estimates of) coefficients we get from our linear regression model. Now, we want to bootstrap our observations. 

## Using Bootstrap

For the purposes of this tutorial, we can write a simple resampling function using [random integers generated by NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html?highlight=randint#numpy.random.randint) (note that these are not true random numbers): 
```python
def simple_resample(n): 
    return(np.random.randint(low = 0, high = n, size = n))
```

Now, let's write out a very general bootstrap function that takes in a bootstrap population (not the true population), some statistic, a resampling function (set default to our `simple_resample`), and the amount of replicates (set default to 10000):

```python
def bootstrap(boot_pop, statistic, resample = simple_resample, replicates = 10000):
    n = len(boot_pop)
    resample_estimates = np.array([statistic(boot_pop[resample(n)]) for _ in range(replicates)])
    return resample_estimates
```

Let's just focus on the coefficient for `sqft_living`. 
We will sample with replacement from our bootstrap sample (our original data), and fit a new linear regression model to the sampled data. The coefficient for `sqft_living` will be used as our bootstrap statistic: 


```python
def sqft_coeff(data_array):
    X = data_array[:, 1:]
    y = data_array[:, 0]
    
    linear_model = lm.LinearRegression()
    model = linear_model.fit(X, y)
    theta_sqft = model.coef_[1]

    return theta_sqft

data_array = df.loc[:,  ["price", "age of house", "sqft_living"]].values

theta_hat_sampling = bootstrap(data_array, sqft_coeff)
```

## Constructing a Confidence Interval 

First, let's construct a histogram of our sampling distribution to get a visual approximation: 

```python
plt.hist(theta_hat_sampling, bins = 30, density = True)
plt.xlabel("$\\tilde{\\theta}_{sqft}$ Values")
plt.ylabel("Proportion per Unit")
plt.title("Bootstrap Sampling Distribution of $\\tilde{\\theta}_{sqft}$ (Nonparametric)");
plt.show()
```
![bt](bt.svg)


Though we cannot direclty measure the *true* coefficient, we can construct a confidence interval to account for variability in the *sample* coefficient. If we construct a 95% confidence interval, we need to look at the 2.5th percentile and 97.5th percentile as our endpoints: 


```python
np.percentile(theta_hat_sampling, 2.5), np.percentile(theta_hat_sampling, 97.5)
```

> (293.0643343501463, 316.5597894314288)

<!-- Columns:

- `id`
- `date`
- `price`
- `bedrooms`
- `bathrooms`
- `sqft_living`
- `sqft_lot`
- `floors`
- `waterfront`
- `view`
- `condition`
- `grade`
- `sqft_above`
- `sqft_basement`
- `yr_built`
- `yr_renovated`
- `zipcode`
- `lat`
- `long`
- `sqft_living15`
- `sqft_lot15` -->


<!-- ## Additional Resources -->
<!-- 
You can read more about bootstrapping [on Wikipedia](https://en.wikipedia.org/wiki/Bootstrapping_(statistics).) -->


<!-- ## References -->