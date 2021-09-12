import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


#------data gathering
#------importing dataset
dataset = pd.read_csv('example_data.csv')

#------data exploring
#------selecting column (features, target, etc)
X = dataset.iloc[:, :-1].values #select all data except -1 = the last data or column
Y = dataset.iloc[:, -1]

#------split the dataese into train dataset and test dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,Y) #import another file, if the testing data in different file

#------Data cleansing: handling the missing data, suspicious data and errorness data
#------missing data? using SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean') #for all nan data will be replaced by the mean
imputer.fit(X[:, 1:3]) #will look all missing values only in the int column
X[:, 1:3] = imputer.transform(X[:, 1:3])
a=1

if type(a) is str:
    print('str')
else:
    print('eror')


