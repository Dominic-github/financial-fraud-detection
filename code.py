# importing the library
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report


# reading the dataset
df = pd.read_csv("./Synthetic_Financial_datasets_log.csv")


# Removing the columns that are not necessary for the data modeling
# the columns that are not necessary are oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, nameDest and nameOrig
df.drop(['oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest', 'nameOrig', 'nameDest'], axis = 1, inplace = True)

# checking the missing values in the data
print(df.isna().sum())


print(df['isFraud'].value_counts())

# # encoding the categorical column into numerical data
# le = LabelEncoder()
# df['type'] = le.fit_transform(df['type'])


# # separating feature variables and class variables
# X = df.drop('isFraud', axis = 1)
# y = df['isFraud']


# # standardizing the data
# sc = StandardScaler()
# X = sc.fit_transform(X)

# # splitting the data into training and testing set
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)



# # make an object of logistic regression
# sv = DecisionTreeClassifier(criterion='entropy', max_depth = 20, )

# #fitting the trainig data into lr model
# model = sv.fit(X_train, y_train)

# # testing the model on test data
# y_pred = sv.predict(X_test)

# #accuracy of the logistic regression
# accuracy_sv = accuracy_score(y_test, y_pred)

# # precision of the logistic regression
# precision_sv = precision_score(y_test, y_pred)

# # recall of the logistic regression
# recall_sv = recall_score(y_test, y_pred)

# # classification report
# classification_sv = classification_report(y_test, y_pred)

# # print the performance matrix
# print(f"Accuracy of Decision Tree {accuracy_sv}")
# print(f"Precision of Decision Tree {precision_sv}")
# print(f"Recall of Decision Tree {recall_sv}")
# print(f"Classification Report of Decision Tree\n {classification_sv}")


# performance_df = pd.DataFrame({
#     'models' : [ 'Decision Tree'],
#     'accuracy' : [accuracy_sv],
#     'precision' : [precision_sv],
#     'recall' : [recall_sv]
# })
# print(performance_df)



# import graphviz
# from sklearn import tree

# # Feature and class names
# feature_names = df.drop('isFraud', axis=1).columns
# class_names = ['Not Fraud', 'Fraud']

# # Export the decision tree
# dot_data = tree.export_graphviz(sv, out_file=None, 
#                                 feature_names=feature_names,  
#                                 class_names=class_names,  
#                                 filled=True, rounded=True,  
#                                 special_characters=True)

# # Render the tree
# graph = graphviz.Source(dot_data)
# # graph.format = 'pdf'
# graph.render("Decision_Tree")  # Save as a file (optional)
# # graph.view()                   # View the tree
