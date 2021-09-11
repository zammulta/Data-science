from scipy.stats.stats import pearsonr
import numpy as np
def drop_feature(X,Y):
    i1 = 0
    for i in range(X.shape[1]):
        i = i - i1
        corr = pearsonr(X[:, i], Y)
        PEr = .674 * (1 - corr[0] * corr[0]) / (len(X[:, i]) ** (1 / 2.0))
        print(corr[0])
        if abs(corr[0]) < PEr:
            X = np.delete(X, i, 1)
            i1 = i1 + 1
    return X