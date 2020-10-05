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