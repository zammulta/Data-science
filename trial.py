import sys
sys.path.append('D:\Github\Data-science\Python')
from Pearsons_correlation import drop_feature
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
le = LabelEncoder()
from sklearn.datasets import load_iris
x1=([[1,7,3],
            [2,8,1],
            [3,9,5],
             [4,10,2]])
c=['x1','x2','x3']
x1=pd.DataFrame(x1, columns=c)

y1=np.array([1,2,3,4])
y2=np.array([[1],
             [2],
             [3],
             [4]])
y2=pd.DataFrame(y2, columns=['target'])
y3=np.transpose(y2)

print(drop_feature(x1,y2))


