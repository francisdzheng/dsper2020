---
layout: page
title: K-Nearest Neighbors
parent: Tutorials
nav_exclude: true
---

# K-Nearest Neighbors
{:.no_toc}

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Importing Libraries

As usual, we will want to use `numpy`, `pandas`, `matplot`, `seaborn` to help us manipulate and visualize our data. 

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

Additionally, we will want to use `scikit-learn` to apply the KNN algorithm on our data set. This is the beauty of libraries such as `scikit-learn`; we do not have to worry about the details of the algorithm's implementation and can simply use the functions the libraries provide. Later in this tutorial, we'll import functions from `scikit-learn` as we need. 

We'll be able to use `scikit-learn` to help us with both our classification problems and our regression problems. 

## K-Nearest Neighbors Classification
You can view the code for this tutorial [here](https://colab.research.google.com/drive/10WytCrsm7kWcIjTuowVVeoI5WGjtMPqj).

### Importing Our Data Set
For examples of KNN, the [iris data set](https://archive.ics.uci.edu/ml/datasets/Iris) from the University of California, Irvine is often used. 

In this tutorial, we will be attempting to classify types of irises using the following four attributes:

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

There are three types of irises: 
1. Iris Setosa
2. Iris Versicolor
3. Iris Virginica

Let's import the data set as a `pandas` dataframe:

```python
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
```

Let's take a small look at what our data set looks like, using `df.head()`, which shows us the first five rows of our data set:

|   | 5.1 | 3.5 | 1.4 | 0.2 | Iris-setosa |
|:-:|:---:|:---:|:---:|:---:|:-----------:|
| **0** | 4.9 | 3.0 | 1.4 | 0.2 | Iris-setosa |
| **1** | 4.7 | 3.2 | 1.3 | 0.2 | Iris-setosa |
| **2** | 4.6 | 3.1 | 1.5 | 0.2 | Iris-setosa |
| **3** | 5.0 | 3.6 | 1.4 | 0.2 | Iris-setosa |
| **4** | 5.4 | 3.9 | 1.7 | 0.4 | Iris-setosa |

### Preprocessing 

Notice that our data set does not have proper column names. Thus, we actually need to add the columm names on our own so that our dataframe is easier to read and work with. Let's try importing our data set one more time, using `names` from [`read_csv`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html):

```python
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'type'])
```

To make our code a bit easier to read, let's write the following instead:

```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'type']
df = pd.read_csv(url, names=names)
```

Now, let's take a look at our dataframe using `df.head()` again:

|       | sepal-length | sepal-width | petal-length | petal-width |    type    |
|:-----:|:------------:|:-----------:|:------------:|:-----------:|:-----------:|
| **0** |      5.1     |     3.5     |      1.4     |     0.2     | Iris-setosa |
| **1** |      4.9     |     3.0     |      1.4     |     0.2     | Iris-setosa |
| **2** |      4.7     |     3.2     |      1.3     |     0.2     | Iris-setosa |
| **3** |      4.6     |     3.1     |      1.5     |     0.2     | Iris-setosa |
| **4** |      5.0     |     3.6     |      1.4     |     0.2     | Iris-setosa |

#### Defining Attributes and Labels

Now, we need to split our data set into its attributes and labels, using [`iloc`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html):

```python
X = df.iloc[:, :-1] #attributes, iloc[:, :-1] means until the last column
y = df['type'] #labels
```
#### Splitting Training Data and Testing Data

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

### Fitting Data and Predicting Data

Now, we're finally ready to import the [KNN classifier algorithm from `scikit_learn`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html):

```python
from sklearn.neighbors import KNeighborsClassifier
```

Now, we need to choose a `K` value for our classifer. As we learned in class, there are pros and cons to choosing a higher or lower `K` value. For now, let's start out with 5, as this is a common initial value to work with:

```python
classifier = KNeighborsClassifier(n_neighbors=5)
```

Finally, we can fit our model using our training data, and then make our first predictions using this model: 

```python
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
```

Now, our model can take attributes (sepal-length, sepal-width, petal-length, and petal-width) and predict which type of iris it is. We are doing this using our test data, `X_test`.

### Evaluating our Algorithm

We do not have too many data points, so we can first just compare our predictions with our test data visually:

```python
print("predicted: ", y_pred)
print("actual: ", y_test)
```

|    Predicted    |      Actual     |
|:---------------:|:---------------:|
|  Iris-virgnica  |  Iris-virginica |
| Iris-versicolor | Iris-versicolor |
|  Iris-virginica |  Iris-virginica|
|   **Iris-versicolor**  | **Iris-virginica**|
| Iris-versicolor | Iris-versicolor |
| Iris-versicolor | Iris-versicolor |
| Iris-setosa | Iris-setosa|
|  Iris-versicolor |  Iris-versicolor |
| Iris-versicolor | Iris-versicolor |
| Iris-virginica | Iris-virginica |
| Iris-versicolor | Iris-versicolor |
| Iris-versicolor | Iris-versicolor |
|   Iris-veriscolor  |   Iris-versicolor   |
|  Iris-setosa | Iris-setosa |
|   Iris-setosa   |   Iris-setosa   |
| Iris-setosa| Iris-setosa |
| Iris-versicolor | Iris-versicolor |
|  Iris-virginica |  Iris-virginica |
|   Iris-setosa   |   Iris-setosa   |
|  Iris-setosa |  Iris-setosa |
| Iris-setosa | Iris-setosa |
| Iris-virginica | Iris-virginica|
|   Iris-versicolor   |   Iris-versicolor  |
| Iris-setosa| Iris-setosa |
| **Iris-virginica** | **Iris-versicolor** |
|  Iris-virginica |  Iris-virginica |
|  Iris-versicolor|  Iris-versicolor |
|   Iris-versicolor  |   Iris-versicolor   |
| Iris-virginica| Iris-virginica |
| Iris-virginica | Iris-virginica |

Because the way we split our data will be different each time, you may get different results, but the above table is shown just to give you an idea of how well our classification algorithm works. 

#### Generating a Classification Report

We still would like to evaluate our algorithm numerically, rather than just visually. Especially for larger data sets, looking at the type of table above becomes impossible. Let's take a look at the confusion matrix and classification report using the following code:

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

We then get the following classification report:

|                     | **precision** | **recall** | **f1-score** | **support** |
|---------------------|:-------------:|:----------:|:------------:|:-----------:|
|     **Iris-setosa** |      1.00     |    1.00    |     1.00     |      8     |
| **Iris-versicolor** |      0.92    |    0.92   |     0.92     |      13    |
|  **Iris-virginica** |      0.89     |    0.89    |     0.89     |      9      |

#### Visualizing our Predictions

I won't go through the details of the following code, but you should understand that it produces a [heat map](https://en.wikipedia.org/wiki/Heat_map). 

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

clf = SVC(kernel = 'linear').fit(X_train, y_train)
clf.predict(X_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm,
                     index = ['setosa','versicolor','virginica'], 
                     columns = ['setosa','versicolor','virginica'])

sns.heatmap(cm_df, annot=True)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

Using the above code, we get the following heat map:

![heatmap](heat.png)


Using this heat map, we can make the following observations:

1. All setosa flowers are correctly classified by our model. 
2. 12 versicolor flowers are correclty classified, and one versicolor flower is incorrectly classified as a virginica flower.
3. 8 virginica flowers are correctly classified, and one virginica flower is incorrectly classified as a versicolor flower.

Again, your results will be slightly depending on how you split your training and test data.  

## K-Nearest Neighbors Regression

This section is under construction.
