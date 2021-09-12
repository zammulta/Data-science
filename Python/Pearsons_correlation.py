# example data = url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
from scipy.stats.stats import pearsonr
import numpy as np
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
def drop_feature(X,Y):
    checkY=Y.dtypes
    if checkY[0]=='object':
        Y['Target'] = le.fit_transform(Y['Target'])
    Y = Y.values
    X = X.values
    i1 = 0
    for i in range(X.shape[1]):
        i = i - i1
        corr = pearsonr(X[:, i], Y[:,0])
        PEr = .674 * (1 - corr[0] * corr[0]) / (len(X[:, i]) ** (1 / 2.0))
        print(corr[0])
        if abs(corr[0]) < PEr:
            X = np.delete(X, i, 1)
            i1 = i1 + 1
    return X