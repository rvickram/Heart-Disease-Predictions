# By Ryan Vickramasinghe

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV into a dataframe
df_data = pd.read_csv('heart.csv')
print(df_data.head())

# Examine the balance between two classes (target 1 = has heart disease, 
# target 0 = does not have heart disease)
sns.countplot(x = df_data['target'])
# Add labels to plot
plt.title('Countplot of Target')
plt.xlabel('target')
plt.ylabel('Patients')
plt.show()

# separate prediction data (x) from desired output (y)
x = df_data.drop(['target'], axis=1) # parameters/data used to make predictions
y = df_data['target'].values # what we are predicting

# do some scaling (since using KNN)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x = ss.fit_transform(x)

# split training and test data (70 : 30)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)