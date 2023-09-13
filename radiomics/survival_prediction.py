#%% Setup
import pandas as pd
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Lasso, LogisticRegression, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn import preprocessing

import warnings
warnings.filterwarnings(action = 'once')

mount = "MICCAI_BraTS2020_TrainingData"
print("Setup Complete")

#%% Load Data
name_mapping = pd.read_csv(mount + "/name_mapping.csv")
survival_info = pd.read_csv(mount + "/survival_info.csv", keep_default_na=False)
X = pd.read_csv(mount + "/radiomics.csv")

#%% Load Survival Prediction Data

ids = survival_info['Brats20ID']
y = survival_info['Survival_days'][survival_info['Brats20ID'].isin(ids)].reset_index(drop=True)

skip_index = []
for i in range(len(y)):
    if y[i].find('ALIVE') > -1:
        skip_index.append(i)

age_info = survival_info['Age'][survival_info['Brats20ID'].isin(ids)].reset_index(drop=True)
resection_info = survival_info['Extent_of_Resection'][survival_info['Brats20ID'].isin(ids)].reset_index(drop=True)

for i in range(len(resection_info)):
    if resection_info[i] == 'NA':
        resection_info[i] = 1
    if resection_info[i] == 'STR':
        resection_info[i] = 2
    if resection_info[i] == 'GTR':
        resection_info[i] = 3

X = X[X['BraTS_2020_subject_ID'].isin(ids)].reset_index(drop=True)
X = X.drop(['BraTS_2020_subject_ID'], axis = 1)
X = X.drop(X.columns[0], axis = 1).reset_index(drop=True)

X = X.join(age_info)
X = X.join(resection_info)

X = X.drop(skip_index, axis = 0).reset_index(drop=True).astype('float64')
y = y.drop(skip_index, axis = 0).reset_index(drop=True).astype('float64')

y_categorical = y.copy()
for i in range(len(y)):
    if y_categorical[i] < 300:
        y_categorical[i] = 1
    if y_categorical[i] > 300 and y_categorical[i] < 450:
        y_categorical[i] = 2
    if y_categorical[i] > 450:
        y_categorical[i] = 3
        
#%% Plot Survival Prediction Data Distribution

y_low = y[y < 300]

y_mid = y[y > 300]
y_mid = y_mid[y_mid < 450]

y_high = y[y > 450]

plt.figure(figsize=(12, 8))
sns.set(font_scale = 1.3)

sns.distplot(y_low, bins = 10)
sns.distplot(y_mid, bins = 5, color = 'green')
sns.distplot(y_high, bins = 15, color = 'red')

plt.legend(
        ('Short-Term','Mid-Term','Long-Term'),
        loc='upper right',
        prop={'size': 15}
    ) 
plt.title("Distributions of Survival Days", fontsize = 25)
plt.xlabel("Survival Days", fontsize = 20)
plt.ylabel("Density", fontsize = 20)

plt.savefig("survival.png")

#%% Survival Prediction Regressor

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

x_train = x_train.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train)

x_test = x_test.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_test = min_max_scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test)

parameters = [{'alpha' : [1.0, 0.75, 0.5, 0.25, 0.1, 0.01], 'tol' : [1, 0.1, 0.01, 0.001], 'max_iter' : [1000000]}] 
gcv = GridSearchCV(Lasso(), parameters)
gcv.fit(x_train, y_train)
best_parameters = gcv.best_params_
regressor = Lasso(alpha = best_parameters['alpha'], normalize = True, tol = best_parameters['tol'], max_iter = best_parameters['max_iter'])

regressor = Lasso(alpha = 1.0, tol = 0.01, max_iter = 100000)
regressor.fit(x_train, y_train)

score = regressor.score(x_test, y_test)
print(score)
filename = "SurvivalPredictionRegressor-{}.pkl".format(score)

with open(filename, 'wb') as file:
    pickle.dump(regressor, file)

#%% Survival Prediction Classifier

x_train, x_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=0)

classifier = RandomForestClassifier(n_estimators = 1000, 
                                    max_features = 'auto', 
                                    max_depth = 7)
classifier.fit(x_train, y_train)
score = classifier.score(x_test, y_test)

filename = "SurvivalPredictionClassifier-{}.pkl".format(score)
print(score)

with open(filename, 'wb') as file:
    pickle.dump(classifier, file)

#%% Plot Survival Prediction Model Importances

importance = classifier.feature_importances_
most_important = []
colnames = []

for i in range(len(importance)):
    
    if importance[i] > 0.005:
        print(X.columns[i] + " " + str(importance[i]))
        most_important.append(importance[i])
        colnames.append(X.columns[i])
        
colnames = ['Tumor Core 2D Diameter (X)',
            'Tumor Core 2D Diameter (Z)',
            'Tumor Core 3D Diameter',
            'Edema Sphericity',
            'Enhancing Tumor Major Axis Length',
            'Enhancing Tumor 2D Diameter (X)',
            'Enhancing Tumor 2D Diameter (Z)',
            'Enhancing Tumor 3D Diameter',
            'FLAIR MRI First Order - Minimum',
            'T1 MRI First Order - Minimum',
            'T1 MRI NGTDM - Contrast',
            'T1CE MRI First Order - Minimum',
            'T2 MRI GLCM - Imc1',
            'T2 MRI GLCM - MCC',
            'Age']

colnames = colnames[-1:] + colnames[:-1]  
most_important = most_important[-1:] + most_important[:-1]  

fig = plt.figure(figsize=(12, 8))
sns.set(font_scale = 2.0)
sns.barplot(colnames, most_important, color = 'darkcyan')
plt.xticks(ticks = list(range(15)), labels=colnames,rotation=90,fontsize=20)
plt.ylabel("Gini Importance")
plt.ylim(ymin=0.004)
plt.title("Most Important Features in Survival Prediction")
plt.show()

fig.savefig("survivalpredfeatures.png", bbox_inches = "tight")