import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


#------data gathering
#------importing dataset
filename = 'example_data.csv'
dataset = pd.read_csv(filename)
print(dataset)
#------data exploring
#------selecting column (features, target, etc)
X = dataset.iloc[:, :-1].values #select all data except -1 = the last data or column
Y = dataset.iloc[:, -1].values #target
#------column name
columns = dataset.columns.tolist()
features = columns[:-1]
target = columns[-1:]

#------split the dataese into train dataset and test dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,Y) #import another file, if the testing data in different file

#------Data cleansing: handling the missing data, suspicious data and errorness data
#------missing data? using SimpleImputer
#imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean') #for all nan data will be replaced by the mean
#imputer.fit(X[:, 1:3]) #will look all missing values only in the int column
#X[:, 1:3] = imputer.transform(X[:, 1:3])
missing_values = ['n/a', 'na', '--']
dataset = pd.read_csv(filename, na_values = missing_values)
#for i in range(dataset.shape[1]):
    #for j in range(dataset.shape[0]):







#print(dataset.isnull().sum())


