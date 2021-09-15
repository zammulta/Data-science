import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
ct = ColumnTransformer(
     [("norm1", Normalizer(norm='l1'), [1]),
     ("norm2", Normalizer(norm='l1'), slice(4))])
X = np.array([[0., 1., 2., 2.],
              [1., 1., 0., 1.]])
print(X)
print(ct.fit_transform(X))
