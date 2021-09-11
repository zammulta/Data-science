from Pearson import drop_feature
import numpy as np
import pandas as pd
from classification import d_tree
from classification import rfc
from classification import NB
from classification import vec_mach
from pca import princ_ana
from sklearn.datasets import load_iris


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, names=['sepal width','sepal length','petal length','petal width','target'])
features = ['sepal length', 'sepal width', 'petal length', 'petal width']

print(df.head())

##cassification
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Categorical.from_codes(iris.target, iris.target_names)
x1=np.array([[3,7,3],
            [4,8,1],
            [5,9,5],
             [6,10,2]])
y1=np.array([1,2,3,4])
#y= pd.get_dummies(y)
#print(y)



#classification-------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import pandas as pd


#input X and y
def d_tree(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    return y_pred

def rfc(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #print(y_pred)
    return

def NB(X,y):
    # Create a Gaussian Classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    model = GaussianNB()
    # Train the model using the training sets
    model.fit(X_train, y_train)
    # Predict Output
    y_pred=model.predict(X_test)
    #print(y_pred)
    return y_pred

def vec_mach(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    clf = svm.SVC(kernel='linear')  # Linear Kernel
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    print(y_pred)
    return y_pred
#-----------

#call---------
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
from pydot import graph_from_dot_data
import pandas as pd
import numpy as np

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Categorical.from_codes(iris.target, iris.target_names)
y = pd.get_dummies(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print(y_pred)

#------------


#mi----------
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
a = np.array([1, 12, 13, 20, 30, 16, 70, 20, 80, 1])
b = np.array([14, 15, 16, 50, 40, 41, 40, 30, 10, 12])

print(stats.entropy([0.5,0.5])) # entropy of 0.69, expressed in nats
print(mutual_info_classif(a.reshape(-1,1), b, discrete_features = True)) # mutual information of 0.69, expressed in nats
print(mutual_info_score(a,b)) # information gain of 0.69, expressed in nats
#-----------

#PCA----
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
#features = ['sepal length', 'sepal width', 'petal length', 'petal width']
def princ_ana(df,features):
    df1 = df.loc[:, features].values
    x = df.loc[:, features].values
    y = df.loc[:, ['target']].values
    x = StandardScaler().fit_transform(x)
    pca = PCA()
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents)
    finalDf = pd.concat([principalDf, df[['target']]], axis=1)
    print(pca.explained_variance_ratio_)
    return finalDf

#--------------