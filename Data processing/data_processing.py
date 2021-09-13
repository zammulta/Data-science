import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


#------data gathering
#------importing dataset
filename = 'example_data.csv'
dataset = pd.read_csv(filename)


#------data exploring
#------column name
columns = dataset.columns.tolist()
features = columns[:-1]
target = columns[-1:]
#------handling missing data
missing_values = ['n/a', 'na', '--']
for i in range(dataset.shape[1]):
    for j in range(np.shape(missing_values)[0]):
        dataset[columns[i]] = dataset[columns[i]].replace(missing_values[j], np.NaN)
print(dataset.isnull().sum())
print(dataset)

#------Data cleansing: handling the missing data, suspicious data and errorness data
#------missing data? using SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'most_frequent') #most_requent = for all types data, mean = all except str
imputer.fit(dataset) #will look all missing values only in the int column
dataset = imputer.transform(dataset)

#------selecting column (features, target, etc)
X = dataset[:, :-1] #select all data except -1 = the last data or column
Y = dataset[:, -1:] #target
#------split the dataese into train dataset and test dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,Y) #import another file, if the testing data in different file





