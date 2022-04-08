
DIAMONDS PRICE PREDICT 
====================================


This project is based on the practice of machine learning
and its models just like sql, trying to predict the price
of a diamond with a previously trained model, and some data
organized thanks to the sql alchemy library.



ENVIRONMENT AND IMPORTS
========================

We created a new environment called: SCIKIT-LEARN In this 
environment we had to install the next libraries from python:

* Python (conda install python == 3.7 / pip install python == 3.7)
* Pandas (conda install pandas / pip install pandas)
* Matplotlib (conda install -c conda-forge matplotlib / pip install matplotlib)
* SQLAlchemy (conda install sqlalchemy / pip install sql alchemy)
* OS (conda install os / pip install os)
* Scikit-learn (pip install -U scikit-learn)
* Numpy (conda install numpy / pip install numpy)

IMPORTS:

* import pandas as pd
* import numpy as np
* import os
* import sqlalchemy
* from sklearn.linear_model import LinearRegression
* import matplotlib.pyplot as plt
* from sklearn import datasets
* from sklearn.datasets import make_regression
* from sklearn.model_selection import train_test_split
* import pickle
* import joblib
* from sklearn.linear_model import LinearRegression
* from sklearn.ensemble import RandomForestRegressor
* from sklearn.metrics import mean_squared_error
* from sklearn.metrics import r2_score
* from sklearn.preprocessing import StandardScaler
* from sklearn.preprocessing import RobustScaler
* from sklearn.preprocessing import MinMaxScaler
* from sklearn.preprocessing import OrdinalEncoder



CONSTRUCTION AND USE
======================

As the first part of the project, we perform a query
to join the data from the diamonds database
with which later making certain modifications
We will train the chosen machine learning model.

Once the machine learning model has been trained and knowing which
is the error (RMSE) that we have in our test, we load the dataset
of diamonds_test with which we will later make a submission
to kaggle as part of the competition.

On the test dataset we make the same modifications as
we did in previous steps to the train dataset, with the peculiarity that
on this dataset we cannot eliminate outliers, since what
we want is the prediction of the price of diamonds independently
of the data we have.

The last step is to perform the submission on the test dataset
to check what is our score (RMSE), in the competition
by kaggle.


FOLDERS
========
Regarding the structure of the project, it is organized in four folders
defining the most important parts of the project.

   * The first one is the "correlation" folder, in which you can find
     an analysis of the correlation between the different features that affect
     to the prediction of the target("price").

   * Secondly, there is a folder "kaggle_tests_and_data" in which you can
     find all the tests performed for the prediction of our model,
     also including the necessary data for those predictions, as well as the queries
     necessary to be able to obtain the train dataset through the database
     db_train.

   * As for the third folder we find a selection of the best
     predictions for the kaggle competition.

   * Finally we find a folder "plots" in which are the
     visualizations about the features used for our model, since it is
     a simpler and more intuitive way to obtain conclusions, both for the
     creation of the model as for the cleaning of outliers.





