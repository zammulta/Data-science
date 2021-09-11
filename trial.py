from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x = ['Apple', 'Orange', 'Apple', 'Pear']
y = le.fit_transform(x)
print(x)
print(y)