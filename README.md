# Iris-flower-classification
It is a classification based machine learning model to predict the species of an iris flower.

This program applies basic machine learning (classification) concept using KNN(k-nearest neighbors) on Fisher's Iris Data to predict the species of a new sample of Iris flower.

### Tools and Languages Used:
<img align="left" alt="VS Code" width="26px" src="vscode.png" />
<img align="left" alt="Python" width="26px" src="python.png" />
<img align="left" alt="colab" width="26px" height="34px" src="colab.png" />
<img align="left" alt="numpy"  height="34px" src="numpy.png" />
<img align="left" alt="scikit learn"  height="34px" src="Scikit_learn.png" />
<img align="left" alt="pandas"  height="34px" src="pandas.png" />
<br>

## Introduction
The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis.

The dataset consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor).
Four features were measured from each sample (in centimetres):
* Length of the sepals
* Width of the sepals
* Length of the petals
* Width of the petals

## Working of the model

* We took data from the load_iris() function available in sklearn module already. 
* The program then divides the dataset into training and testing samples in 80:20 ratio .
* Accuracy score is then calculated by comparing with the correct results of the training dataset.

## Steps to follow
-Download the code from the given github repository<br>
-open it in colab platform <br>
-Run the code<br>

# Breaking the code


##Importing libraries:

###Code snippet 1
```python
import numpy as np
import pandas as pd
```

##Exploring the data 
###Code snippet 2
```python
from sklearn.datasets import load_iris
iris_dataset = load_iris()
```


###Code snippet 3
```python
print("Keys of iris dataset: \n{}".format(iris_dataset.keys()))

val = iris_dataset['DESCR']
start_val = val[:200]     # slicing the string
print(start_val + "\n...")

print("Target names: {}".format(iris_dataset['target_names']))
```
printing keys of the dataset.The value of the key DESCR is a short description of the dataset.<br>
Target names are the targets to which the data is to be classified.<br>


 
###Code snippet 4
```python
print("Feature names: {}".format(iris_dataset['feature_names']))

print("Type of data: {}".format(type(iris_dataset['data'])))
```
Features are the parameters of classification (these data are taken) <br>
Type of data is how the data is stored. In this case it is stored as numpy ndarray.<br>



###Code snippet 5
```python
print("Shape of data: {}".format(iris_dataset['data'].shape))

print("First five columns of data: \n{}".format(iris_dataset['data'][:5]))
```
Shape of data gives an idea about data points and features(no of rows and columns the data have).<br>
Features for first five samples(data points) Here we are just slicing the 2d numpy array.<br>



###Code snippet 6
```python
print("Type of target : {}".format(type(iris_dataset['target'])))

print("Shape of target : {}".format(iris_dataset['target'].shape))
```
Here we are using train_test_split to split the entire data. X_train contains 75% of the rows and X_test contains the remaining. Here y denotes labels.



###Code snippet 7
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state = 0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))
```
The dataset after dividing into training and testing dataset.



###Code snippet 8
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
```
* This is the k nearest neighbours machine learning model. <br>
* Here we are using k nearest neighbour classification algorithm in the KNeighborsClassifier class in the neighbours module. Before we can use model, we need to instantiate the class into an object.



###Code snippet 9
```python
knn.fit(X_train, y_train)
```
Training the entire model using training data<br>
* Fit the entire k nearest neighbour classification algorithm to the X_train, y_train values



##Making Predictions:

###Code snippet 10
```python
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

prediction = knn.predict(X_new)

print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
    iris_dataset['target_names'][prediction]))
```
It predicts <br>
Prediction: [0] <br>
Predicted target name: ['setosa']



##Evaluating the model
###Code snippet 11
```python
y_pred = knn.predict(X_test)
print("Test set predictions: \n {}".format(y_pred))

print("Test set score: {}".format(np.mean(y_pred == y_test)))
```
Evaluation of the model on the basis of the prediction to calculate the accuracy.



##Checking the accuracy of the model
###Code snippet 11
```python
print("Test set score: {}".format(np.mean(y_pred == y_test)))
```
We got an accuracy of 0.9736842105263158


