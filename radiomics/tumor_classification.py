#%% Setup
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import preprocessing

import warnings
warnings.filterwarnings(action = 'once')

mount = "MICCAI_BraTS2020_TrainingData"
print("Setup Complete")

#%% Load Data
name_mapping = pd.read_csv(mount + "/name_mapping.csv")
X = pd.read_csv(mount + "/radiomics.csv")

#%% Load Tumor Classification Data
y = pd.DataFrame({'grade':name_mapping['Grade'].map({'HGG' : 1, 'LGG': 0})})
y = y['grade']

X = X.drop('BraTS_2020_subject_ID', axis=1).reset_index(drop=True)
X = X.drop(X.columns[0], axis = 1).reset_index(drop=True).astype('float64')

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%% Train Tumor Classification Model
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
score = rfc.score(x_test, y_test)

filename = "TumorClassification-{}.pkl".format(score)
with open(filename, 'wb') as file:
    pickle.dump(rfc, file)
   
#%% Plot Tumor Classification Model Importances
importance = rfc.feature_importances_
most_important = []
colnames = []

for i in range(len(importance)):
    if importance[i] > 0.01:
        most_important.append(importance[i])
        colnames.append(X.columns[i])

        
colnames = ['Tumor Core Minor Axis Length',
            'Tumor Core Mesh Volume', 
            'Tumor Core Surface Volume Ratio', 
            'Tumor Core Mesh Volume',
            'Enhancing Tumor Center of Mass (Y)', 
            'Enhancing Tumor Center of Mass (Z)',
            'Enhancing Tumor Flatness',
            'Enhancing Tumor Least Axis Length',
            'Enhancing Tumor Major Axis Length',
            'Enhancing Tumor 2D Diameter (Y)',
            'Enhancing Tumor 2D Diameter (X)',
            'Enhancing Tumor 2D Diameter (Z)',
            'Enhancing Tumor 3D Diameter',
            'Enhancing Tumor Mesh Volume',
            'Enhancing Tumor Minor Axis Length',
            'Enhancing Tumor Sphericity',
            'Enhancing Tumor Surface Area',
            'Enhancing Tumor Surface Volume Ratio',
            'Enhancing Tumor Voxel Volume',
            'T1CE MRI First Order - Skewness',
            'T1CE MRI GLCM - Cluster Shade']
        
fig = plt.figure(figsize=(12, 8))
sns.set(font_scale = 2.0)
sns.barplot(colnames, most_important, color = 'darkcyan')
plt.xticks(ticks = list(range(21)), labels=colnames,rotation=90,fontsize=20)
plt.ylabel("Gini Importance")
plt.title("Most Important Features in Classifying Tumor Progression")
plt.show()
fig.savefig("tumorclassfeatures.png", bbox_inches = "tight")