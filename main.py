import sys
sys.path.append('D:\Github\Data-science\Python')
from Pearsons_correlation import drop_feature
import numpy as np
import pandas as pd



#input data sets
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
df = pd.read_csv(url, names=['Age', 'Workclass', 'Fnlwgt', 'Education', 'Education-num', 'Marital-status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss', 'hours-per-week', 'Native-country', 'Target'])
#features = ['Age', 'Workclass', 'Fnlwgt', 'Education', 'Education-num', 'Marital-status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss', 'hours-per-week', 'Native-country']
features=['Age','Capital-gain', 'Capital-loss', 'hours-per-week' ]
target=['Target']

X=pd.DataFrame(df, columns=features)
Y=pd.DataFrame(df, columns=target)
print(drop_feature(X,Y))


