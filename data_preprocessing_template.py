# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
ds = pd.read_csv('Data.csv')
X = ds.iloc[:, :-1].values
y = ds.iloc[:, 3].values

##1 Missing data example
from sklearn.preprocessing import Imputer
myImputer = Imputer(axis = 0, strategy = 'mean')
X[:, 1:3] = myImputer.fit(X[:, 1:3]).transform(X[:, 1:3])