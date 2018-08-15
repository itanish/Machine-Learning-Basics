#Data Preprocessing

import numpy as nm
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN",strategy="most_frequent",axis=0)

imputer = imputer.fit(X[:,1:3])

X[:,1:3] = imputer.transform(X[:,1:3])

#Categorizing

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le_X = LabelEncoder()

X[:,0] = le_X.fit_transform(X[:,0])

ohe = OneHotEncoder(categorical_features= [0])

X = ohe.fit_transform(X).toarray()

X = X.toarray()

le_Y = LabelEncoder();

Y = le_Y.fit_transform(Y);

## Splitting the dataset into the Training set and Test set


from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)












