import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn_pandas import CategoricalImputer
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
# import pandas_profiling

data = pd.read_excel('data/dataset.xlsx')


predictors = data.iloc[:, 1:74]
predictors = predictors.drop(columns=['SARS-Cov-2 exam result', 'Mycoplasma pneumoniae', 'Urine - pH'])
predictors = predictors
classes = data.iloc[:,2].values

predictors.info()
predictors = predictors.values

simpleImputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
predictors[:,1:19] = simpleImputer.fit_transform(predictors[:,1:19])
predictors[:,36:42] = simpleImputer.fit_transform(predictors[:,36:42])
predictors[:,44:52] = simpleImputer.fit_transform(predictors[:,44:52])
predictors[:,53:68] = simpleImputer.fit_transform(predictors[:,53:68])
categoricalImputer = CategoricalImputer()
predictors[:,19:25] = categoricalImputer.fit_transform(predictors[:,19:25])
predictors[:,25:36] = categoricalImputer.fit_transform(predictors[:,25:36])
predictors[:,42:44] = categoricalImputer.fit_transform(predictors[:,42:44])
predictors[:,68:70] = categoricalImputer.fit_transform(predictors[:,68:70])
predictors[:,52] = categoricalImputer.fit_transform(predictors[:,52])

labelEncoder = LabelEncoder()
for x in [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 42, 43, 68, 69, 52]:
    predictors[:,x] = labelEncoder.fit_transform(predictors[:,x])
print(predictors)
# pandas_profiling.ProfileReport(classes)
train_x,test_x,train_y,test_y = train_test_split(predictors, classes)

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(train_x, train_y)

results = clf.predict(test_x)
print (accuracy_score(test_y, results))

export_graphviz(clf.estimators_[0],
    filled = True)