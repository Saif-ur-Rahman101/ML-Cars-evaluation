# -*- coding: utf-8 -*-
"""
Created on Sat May 18 10:47:52 2024

@author: SAIF
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("car_evaluation.csv", header = None)
df.head()
df.shape
df.info()

col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df.columns = col_names
df.info()

for col in col_names:
    print(df[col].value_counts())
    
df.isnull().sum()


X = df.drop(['class'], axis=1)
y = df['class']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)

import category_encoders as ce
encoder = ce.OrdinalEncoder(cols=col_names[:-1],
                            mapping=[{'col':'buying', 'mapping':{None:0, 'low':1,'med':2, 'high':3,'vhigh':4}},
                                     {'col':'maint', 'mapping':{None:0, 'low':1,'med':2, 'high':3,'vhigh':4}},
                                     {'col':'doors', 'mapping':{'2':1,'3':2, '4':3, '5more':4}},
                                     {'col':'persons', 'mapping':{None:0, '2':1,'3':2,'4':3,'more':4}},
                                     {'col':'lug_boot', 'mapping':{'small':1, 'med':2, 'big':3}},
                                     {'col':'safety','mapping':{'low':1, 'med':2, 'high':3}}])

X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100, random_state=0)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)


from sklearn.metrics import accuracy_score
print('Model Accuracy', accuracy_score(y_test, y_pred))

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


# # Import tools needed for visualization
# from sklearn.tree import export_graphviz
# import pydot
# # Pull out one tree from the forest
# tree = rfc.estimators_[5]
# # Export the image to a dot file
# export_graphviz(tree, out_file = 'tree.dot', feature_names = list(X_train.columns), class_names=['unacc','acc','good','vgood'],
#                 rounded = True, precision = 1, max_depth=3, filled=True)
# # Use dot file to create a graph
# (graph, ) = pydot.graph_from_dot_file('tree.dot')
# # Write graph to a png file
# graph.write_png('tree.png')

from sklearn.tree import plot_tree
plt.figure(figsize=(160,80))
plot_tree(rfc.estimators_[5], feature_names = list(X_test.columns),class_names=rfc.classes_,filled=True, max_depth=2);

pd.Series(rfc.feature_importances_, index = X_train.columns).sort_values()
