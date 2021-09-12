import sys
sys.path.append('D:\Github\Data-science\Python')
from Pearsons_correlation import drop_feature
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pandas as pd
le = LabelEncoder()

#data for pearson's analysis
#all data should be in numerical
#input data sets
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
df = pd.read_csv(url, names=['Age', 'Workclass', 'Fnlwgt', 'Education', 'Education-num', 'Marital-status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss', 'hours-per-week', 'Native-country', 'Target'])
#features = ['Age', 'Workclass', 'Fnlwgt', 'Education', 'Education-num', 'Marital-status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss', 'hours-per-week', 'Native-country']
#features=['Age','Education','Capital-gain']
#target=['Target']


