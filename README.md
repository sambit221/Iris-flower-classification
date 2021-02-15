# Iris-flower-classification
It is a classification based machine learning model to predict the species of an iris flower.

This program applies basic machine learning (classification) concept using KNN(k-nearest neighbors) on Fisher's Iris Data to predict the species of a new sample of Iris flower.

### Tools and Languages Used:
<img align="left" alt="VS Code" width="26px" src="vscode.png" />
<img align="left" alt="Python" width="26px" src="python.png" />
<img align="left" alt="colab" width="26px" height="34px" src="colab.png" />
<img align="left" alt="numpy" width="26px" src="numpy.png" />
<img align="left" alt="scikit learn" width="26px" src="Scikit_learn.png" />
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

### Steps to follow
-Download the code from the given github repository<br>
-open it in colab platform <br>
-Run the code<br>

## Breaking the code
-Code snippet 1
Importing libraries

```cmd
import numpy as np
import pandas as pd

```