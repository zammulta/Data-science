import sys
sys.path.append('D:\Github\Data-science\Python')
from Pearsons_correlation import drop_feature
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
le = LabelEncoder()
from sklearn.datasets import load_iris

df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],'C': [1, 2, 3]})
print(df)
columns = df.columns
features = columns.tolist()
features1=['A', 'B', 'C']
print(features[0])
print(features1)
