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


##2 Categorical Data example
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#leX = LabelEncoder()
#X[: , 0] = leX.fit_transform(X[: , 0])
#oneHotEncoder = OneHotEncoder(categorical_features = [0])
#X = oneHotEncoder.fit_transform(X).toarray()