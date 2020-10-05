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
sns.countplot(x = 'target', data = df_data)
# Add labels to plot
plt.title('Countplot of Target')
plt.xlabel('target')
plt.ylabel('Patients')
print('\nClose plot to continue!')
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

# determine max test score and k value
train_score = []
test_score = []
k_vals = []

for k in range(1, 21):
    k_vals.append(k)
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    
    tr_score = knn.score(X_train, y_train)
    train_score.append(tr_score)
    
    te_score = knn.score(X_test, y_test)
    test_score.append(te_score)

max_test_score = max(test_score)
test_scores_ind = [i for i, v in enumerate(test_score) if v == max_test_score]
print('Max test score {} and k = {}'.format(max_test_score * 100, list(map(lambda x: x + 1, test_scores_ind))))

# Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(3)

knn.fit(X_train, y_train)
print('KNN score: ', knn.score(X_test, y_test))

# Make a prediction
y_pred = knn.predict(X_test)
confusion_matrix(y_test,y_pred)
print("Confusion matrix: \n", pd.crosstab(y_test, y_pred, rownames = ['Actual'], colnames =['Predicted'], margins = True))