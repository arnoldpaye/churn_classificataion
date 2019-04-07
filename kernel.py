import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

dataframe = pd.read_csv('data/churn.csv')
dataframe = dataframe.drop('phone number', axis=1)

cate = [key for key in dict(dataframe.dtypes) if dict(dataframe.dtypes)[key] in ['bool', 'object']]

le = LabelEncoder();
for i in cate:
    le.fit(dataframe[i])
    dataframe[i] = le.transform(dataframe[i])

X = dataframe.drop('churn', axis=1)
y = dataframe.churn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print confusion_matrix(y_test, y_pred)
print classification_report(y_test, y_pred)
