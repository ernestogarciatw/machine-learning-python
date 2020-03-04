# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
ds = pd.read_csv('Data.csv')
X = ds.iloc[:, :-1].values
y = ds.iloc[:, 3].values

from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
#First fill out the NaN fields like in the missing data example
X[:, 1:3] = Imputer(axis = 0, strategy = 'mean').fit(X[:, 1:3]).transform(X[:, 1:3])
#Then transform data in column 0 to numerical values [0,1,2]
X[: , 0] = LabelEncoder().fit_transform(X[: , 0])
#Then transform one column into three with numerical values [0,1]
myOneHotEncoder = OneHotEncoder(categorical_features = [0])
X = myOneHotEncoder.fit_transform(X).toarray()
#Then transform data in y to numerical values [0,1]
y = LabelEncoder().fit_transform(y)
#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
##3 Feature Scaling
standardScaler = StandardScaler()
X_train = standardScaler.fit_transform(X_train)
X_test = standardScaler.transform(X_test)