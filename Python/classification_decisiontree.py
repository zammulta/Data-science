from sklearn.tree import DecisionTreeClassifier
def d_tree(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    return y_pred