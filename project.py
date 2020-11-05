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
predictors = predictors.drop(columns=['Patient addmited to regular ward (1=yes, 0=no)', 'Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)', 'Patient age quantile', 'SARS-Cov-2 exam result', 'Mycoplasma pneumoniae', 'Urine - pH'])
columns_predictors = predictors.columns
classes = data.iloc[:,5].values

predictors.info()
predictors = predictors.values

simpleImputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
predictors[:,0:15] = simpleImputer.fit_transform(predictors[:,0:15])
predictors[:,32:38] = simpleImputer.fit_transform(predictors[:,32:38])
predictors[:,40:48] = simpleImputer.fit_transform(predictors[:,40:48])
predictors[:,49:64] = simpleImputer.fit_transform(predictors[:,49:64])

categoricalImputer = CategoricalImputer()
predictors[:,15:21] = categoricalImputer.fit_transform(predictors[:,15:21])
predictors[:,21:32] = categoricalImputer.fit_transform(predictors[:,21:32])
predictors[:,38:40] = categoricalImputer.fit_transform(predictors[:,38:40])
predictors[:,64:66] = categoricalImputer.fit_transform(predictors[:,64:66])
predictors[:,48] = categoricalImputer.fit_transform(predictors[:,48])

labelEncoder = LabelEncoder()
for x in [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 38, 39, 64, 65, 48]:
    predictors[:,x] = labelEncoder.fit_transform(predictors[:,x])
print(predictors)
# pandas_profiling.ProfileReport(classes)
train_x,test_x,train_y,test_y = train_test_split(predictors, classes)

clf = RandomForestClassifier(max_depth=3, random_state=0)
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
    sorted(enumerate(importance), key=lambda i: i[1], reverse=True)[:10]
))
plot_importances = list(map(
    lambda x: x[1],
    sorted(enumerate(importance), key=lambda i: i[1], reverse=True)[:10]
))
plt.rc('font', size=8)
plt.bar(columns_predictors, plot_importances)
plt.xticks(rotation='92')
plt.tight_layout()
plt.savefig("impostances.png", dpi=600)