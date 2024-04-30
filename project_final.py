import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

import sklearn
import random

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import pickle

# Load the data
data = pd.read_csv('creditcard.csv')

# Display the distributions of 'Amount' and 'Time'
sns.histplot(data['Amount'], kde=True)
plt.show()

sns.histplot(data['Time'], kde=True)
plt.show()

# Display histograms of all features
data.hist(figsize=(20,20))
plt.show()

# Define 'd' before using it
d = data

# Create a joint plot between 'Time' and 'Amount'
sns.jointplot(x='Time', y='Amount', data=d)
plt.show()

# Separate classes
class0 = d[d['Class']==0]
class1 = d[d['Class']==1]

# Resample the data to balance the classes
temp = shuffle(class0)
d1 = temp.iloc[:2000,:]
df_temp = pd.concat([d1, class1])
df = shuffle(df_temp)
df.to_csv('creditcardsampling.csv')

# Display class distribution
# Display class distribution after oversampling
sns.countplot(x='Class', data=data)
plt.show()


# Oversample using SMOTE
oversample = SMOTE()
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]
X, Y = oversample.fit_resample(X, Y)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create DataFrame with standardized features
X = pd.DataFrame(X, columns=df.columns[:-1])

# Concatenate standardized features with target variable
data = pd.concat([X, pd.DataFrame(Y, columns=['Class'])], axis=1)

# Display class distribution after oversampling
sns.countplot(x='Class', data=data)
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data.drop('Class', axis=1), data['Class'], test_size=0.3, random_state=42)

# Perform PCA to reduce dimensionality
pca = PCA(n_components=7)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

# Support Vector Machine with hyperparameter tuning
svc = SVC(kernel='rbf', probability=True)
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 1, 0.01, 0.0001, 0.001]}]
grid_search = GridSearchCV(estimator=svc, param_grid=parameters, scoring='accuracy', n_jobs=-1)
grid_search = grid_search.fit(X_train_reduced, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

# Initialize SVC with best parameters
svc_param = SVC(kernel='rbf', gamma=best_parameters['gamma'], C=best_parameters['C'], probability=True)
svc_param.fit(X_train_reduced, y_train)

# Save the model
pickle.dump(svc_param, open('model.pkl', 'wb'))

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Predict probabilities for test data
y_prob = model.predict_proba(X_test_reduced)
