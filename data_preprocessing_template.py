# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
ds = pd.read_csv('Data.csv')
X = ds.iloc[:, :-1].values
y = ds.iloc[:, 3].values

##2 Categorical Data example
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#First fill out the NaN fields like in the missing data example
X[:, 1:3] = Imputer(axis = 0, strategy = 'mean').fit(X[:, 1:3]).transform(X[:, 1:3])
#Then transform data in column 0 to numerical values [0,1,2]
X[: , 0] = LabelEncoder().fit_transform(X[: , 0])
#Then transform one column into three with numerical values [0,1]
myOneHotEncoder = OneHotEncoder(categorical_features = [0])
X = myOneHotEncoder.fit_transform(X).toarray()
#Then transform data in y to numerical values [0,1]
y = LabelEncoder().fit_transform(y)