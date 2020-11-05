import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn_pandas import CategoricalImputer
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz,plot_tree
# import pandas_profiling

data = pd.read_excel('data/dataset.xlsx')

predictors = data.iloc[:, 1:74]
predictors = predictors.drop(columns=['Patient age quantile', 'SARS-Cov-2 exam result', 'Mycoplasma pneumoniae', 'Urine - pH'])
columns_predictors = predictors.columns
classes = data.iloc[:,2].values

predictors.info()
predictors = predictors.values

simpleImputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
predictors[:,0:18] = simpleImputer.fit_transform(predictors[:,0:18])
predictors[:,35:41] = simpleImputer.fit_transform(predictors[:,35:41])
predictors[:,43:51] = simpleImputer.fit_transform(predictors[:,43:51])
predictors[:,52:67] = simpleImputer.fit_transform(predictors[:,52:67])

categoricalImputer = CategoricalImputer()
predictors[:,18:24] = categoricalImputer.fit_transform(predictors[:,18:24])
predictors[:,24:35] = categoricalImputer.fit_transform(predictors[:,24:35])
predictors[:,41:43] = categoricalImputer.fit_transform(predictors[:,41:43])
predictors[:,67:69] = categoricalImputer.fit_transform(predictors[:,67:69])
predictors[:,51] = categoricalImputer.fit_transform(predictors[:,51])

labelEncoder = LabelEncoder()
for x in [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 67, 68, 51]:
    predictors[:,x] = labelEncoder.fit_transform(predictors[:,x])
print(predictors)
# pandas_profiling.ProfileReport(classes)
train_x,test_x,train_y,test_y = train_test_split(predictors, classes)

clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(train_x, train_y)

results = clf.predict(test_x)
print (accuracy_score(test_y, results))

""" fn = columns_predictors
cn = np.unique(classes)
fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=900)
for index in range(0, 5):
    plot_tree(clf.estimators_[index],
                   feature_names = fn, 
                   class_names=cn,
                   filled = True,
                   ax = axes[index]);

    axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
fig.savefig('rf_5trees.png') """

importance = clf.feature_importances_
for i,v in sorted(enumerate(importance), key=lambda i: i[1], reverse=True)[:10]:
	print('Feature: %s, Score: %.5f' % (columns_predictors[i],v))


columns_predictors = list(map(
    lambda i: columns_predictors[i[0]],
    sorted(enumerate(importance), key=lambda i: i[1], reverse=False)[:10]
))
plot_importances = list(map(
    lambda x: x[1],
    sorted(enumerate(importance), key=lambda i: i[1], reverse=False)[:10]
))
plt.rc('font', size=8)
plt.bar(columns_predictors, plot_importances)
plt.xticks(rotation='92')
plt.tight_layout()
plt.savefig("impostances.png", dpi=600)