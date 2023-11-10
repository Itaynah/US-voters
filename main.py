import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics, __all__, tree
from sklearn import preprocessing
import scipy.stats as stats
from sklearn.tree import DecisionTreeClassifier, plot_tree
df = pd.read_csv('voters_hm4.csv')
# Q1
r_seed = 123

# Q2.a
Crosstab = pd.crosstab(index= df.vote, columns= df.sex)
Crosstab.plot.bar(stacked= True)
plt.show()
Crosstab = pd.crosstab(index= df.vote, columns= df.passtime)
Crosstab.plot.bar(stacked= True)
plt.show()
Crosstab = pd.crosstab(index= df.vote, columns= df.status)
Crosstab.plot.bar(stacked= True)
plt.show()
# Q2.b
df.boxplot(column=['volunteering'], by= 'vote', grid=False)
plt.show()
df.boxplot(column=['salary'], by= 'vote', grid=False)
plt.show()
df.boxplot(column=['age'], by= 'vote', grid=False)
plt.show()

# Q3
print(df.isnull().sum())
df.dropna(subset=['passtime'], inplace=True)
df['age'] = df['age'].replace(to_replace=np.nan,value=df.age.mean())
print('**********\n',df.isnull().sum())
df['salary'] = df['salary'].replace(to_replace=np.nan,value=df.salary.mean())
print('**********\n',df.isnull().sum())
# Q4
le_vote = preprocessing.LabelEncoder()
le_sex = preprocessing.LabelEncoder()
le_pass = preprocessing.LabelEncoder()
le_status = preprocessing.LabelEncoder()
le_vote.fit(df['vote'])
df['target'] = le_vote.transform(df['vote'])
df['sex_s'] = le_sex.fit_transform(df['sex'])
df['passtime_p'] = le_pass.fit_transform(df['passtime'])
df['status_s'] = le_status.fit_transform(df['status'])
x = df.drop(['vote', 'target', 'sex', 'passtime', 'status'], axis= 1)
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=r_seed)
# Q5
model = DecisionTreeClassifier(random_state=r_seed)
model.fit(x_train, y_train)
# Q6
# bulding a confusion matrix
y_pred_test = model.predict(x_test)
cm_test = pd.crosstab(y_test, y_pred_test, colnames = ["pred"], margins = True)
print(cm_test)
print("Test set result:")
print("Accuracy:", metrics.accuracy_score(y_test,y_pred_test))
print("Precision: ", metrics.precision_score(y_test, y_pred_test))
print("Recall: ", metrics.recall_score(y_test, y_pred_test))
# Q7
# matrix for train set
y_pred_train = model.predict(x_train)
cm_train = pd.crosstab(y_train, y_pred_train, colnames = ["pred"], margins = True)
print(cm_train)
print("Train set result:")
print("Accuracy:", metrics.accuracy_score(y_train,y_pred_train))
print("Precision: ", metrics.precision_score(y_train, y_pred_train))
print("Recall: ", metrics.recall_score(y_train, y_pred_train))
# The model is over fitted to the train set
# because the train results are almost perfect while the test set is providing less high results
# Q8
re_model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=40, random_state=r_seed)
re_model.fit(x_train, y_train)
fig = plt.figure(figsize=(14, 8))
tree.plot_tree(re_model, feature_names= x_train.columns, class_names= le_vote.classes_)
plt.show()
# Q8a
# Tree depth is 3
# Q8b
# There are 5 leaves on the tree
# Q8c
# Best splitting feature is "volunteering"
# Q8d
# Not all features were included in the tree: passtime and status
# Q8e
rr = x.iloc[[68]]
print("Sample number 68 predicted to be ", le_vote.inverse_transform(re_model.predict(rr)), "and in reality its 'Republican'")
# Sample number 68 was correctly classified as "republican" as it is in the original data frame
# Q9
# Matrix for test set
y_pred_test = re_model.predict(x_test)
cm_test = pd.crosstab(y_test, y_pred_test, colnames = ["pred"], margins = True)
print("*************************************\nREVISED MODEL")
print(cm_test)
print("Test set result:")
print("Accuracy:", metrics.accuracy_score(y_test,y_pred_test))
print("Precision: ", metrics.precision_score(y_test, y_pred_test))
print("Recall: ", metrics.recall_score(y_test, y_pred_test))
# Matrix for train set
y_pred_train = re_model.predict(x_train)
cm_train = pd.crosstab(y_train, y_pred_train, colnames = ["pred"], margins = True)
print(cm_train)
print("Train set result:")
print("Accuracy:", metrics.accuracy_score(y_train,y_pred_train))
print("Precision: ", metrics.precision_score(y_train, y_pred_train))
print("Recall: ", metrics.recall_score(y_train, y_pred_train))
# Q10
'''
Can conclude from those results that the model is not overfitted or underfitted 
and spot relatively well most of the sample that are positive and predict 
them as true(recall).
On the other hand, the model preform relatively weak results in accuracy and precision  
'''
# Q11
z = df.drop(['vote', 'sex', 'passtime', 'status', 'status_s'], axis= 1)
w = df['status_s']
z_train, z_test, w_train, w_test = train_test_split(z, w, test_size=0.3, random_state=r_seed)
st_model = DecisionTreeClassifier(random_state=r_seed)
st_model.fit(z_train, w_train)
w_pred_test = st_model.predict(z_test)
cm_test = pd.crosstab(w_test, w_pred_test, colnames = ["pred"], margins = True)
print("*************************************\nSTATUS MODEL")
print(cm_test)
print("Test set result:")
print("Accuracy:", metrics.accuracy_score(w_test,w_pred_test))
print("Precision: ", metrics.precision_score(w_test, w_pred_test, average=None))
print("Recall: ", metrics.recall_score(w_test, w_pred_test, average=None))
# Matrix for train set
w_pred_train = st_model.predict(z_train)
cm_train = pd.crosstab(w_train, w_pred_train, colnames=["pred"], margins=True)
print(cm_train)
print("Train set result:")
print("Accuracy:", metrics.accuracy_score(w_train, w_pred_train))
print("Precision: ", metrics.precision_score(w_train, w_pred_train, average=None))
print("Recall: ", metrics.recall_score(w_train, w_pred_train, average=None))
# Q11 Accuracy is: 0.82
# The model won't predict the status that well, we are looking for accuracy higher than what we got
# Q12
# Yes, the model is suspected to be ovrfitted
# when comparing test results and train results, we get perfect train results
# and relatively bad test results
# Q13
# Precision for the "single" category in Status: 0.91666667
