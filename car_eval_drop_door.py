
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 10:47:52 2024

@author: SAIF
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("car_evaluation.csv", header=None)

col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df.columns = col_names

# Drop the 'doors' column
df = df.drop(columns=['doors'])

# Update the column names after dropping 'doors'
col_names = ['buying', 'maint', 'persons', 'lug_boot', 'safety', 'class']
df.columns = col_names
df.info()

# Value counts for each column
for col in col_names:
    print(df[col].value_counts())
    
# Check for missing values
df.isnull().sum()

# Split the data into features and target variable
X = df.drop(['class'], axis=1)
y = df['class']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Encode categorical variables
import category_encoders as ce
encoder = ce.OrdinalEncoder(cols=col_names[:-1],
                            mapping=[{'col':'buying', 'mapping':{None:0, 'low':1,'med':2, 'high':3,'vhigh':4}},
                                     {'col':'maint', 'mapping':{None:0, 'low':1,'med':2, 'high':3,'vhigh':4}},
                                     {'col':'persons', 'mapping':{None:0, '2':1,'3':2,'4':3,'more':4}},
                                     {'col':'lug_boot', 'mapping':{'small':1, 'med':2, 'big':3}},
                                     {'col':'safety','mapping':{'low':1, 'med':2, 'high':3}}])

X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# Train the RandomForest model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=0)
rfc.fit(X_train, y_train)

# Predict the test set results
y_pred = rfc.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score
print('Model Accuracy:', accuracy_score(y_test, y_pred))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=rfc.classes_, yticklabels=rfc.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Visualize one tree from the forest
from sklearn.tree import plot_tree
plt.figure(figsize=(160,80))
plot_tree(rfc.estimators_[5], feature_names=list(X_test.columns), class_names=rfc.classes_, filled=True, max_depth=2)
plt.show()

# Feature importances
feature_importances = pd.Series(rfc.feature_importances_, index=X_train.columns).sort_values()
print(feature_importances)

feature_scores = pd.Series(rfc.feature_importances_, index=X_train.columns).sort_values()


# Creating a seaborn bar plot
sns.barplot(x=feature_scores, y=feature_scores.index)

# Add labels to the graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
# Add title to the graph
plt.title("Visualizing Important Features")
# Visualize the graph
plt.show()