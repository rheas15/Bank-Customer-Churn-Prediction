
pip install mixed_naive_bayes

#Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import seaborn as sns
import scipy.stats as stats
from tabulate import tabulate
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score



#Importing model packages
from mixed_naive_bayes import MixedNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from google.colab import drive
drive.mount('/content/drive')

drive.mount('/content/gdrive/', force_remount = True)

#Reading the CSV file
df = pd.read_csv("/content/drive/MyDrive/Churn_Modelling.csv")

# setting index
df.set_index("RowNumber", inplace=True)

"""# Exploratory Data Analysis"""

# Correlation plot
sub_df =  df.drop(['CustomerId', 'Surname', 'Geography',  'Gender'], axis = 1)

corr = sub_df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
colormap = sns.diverging_palette(220, 10, as_cmap = True)
dropvals = np.zeros_like(corr)
dropvals[np.triu_indices_from(dropvals)] = True
sns.heatmap(corr, cmap = colormap, linewidths = .5, annot = True, fmt = ".2f", mask = dropvals)
plt.show()

# Set style of background of plot
sns.set(style="whitegrid")

# Select features of Geography, Balance, and hue as Gender
sns.boxplot(x="Geography", y="Balance",
                hue="Gender",
                data=df, palette="Set2",
                dodge=True)

plt.legend(loc='upper right')
plt.show()

# Check if there is null in any of the features
df.isnull().sum()

# Summary statistics in excel format
df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']].describe().to_excel('stats.xlsx')

# 1. Drop unnecessary columns, including CustomerId and Surname.
df = df.drop(['CustomerId','Surname'], axis = 1)

# 2. Get dummies for Geography, Gender.

df = pd.get_dummies(df, columns = ['Geography', 'Gender'], drop_first = True)

# Split the data into training, validation and testing sets.
X = df.drop(['Exited'], axis = 1)
y = df['Exited']

X_train_temp, X_test_unscaled, y_train, y_test = train_test_split(X, y, test_size = .5, random_state = 21)
X_val_temp, X_test_unscaled, y_val, y_test = train_test_split(X_test_unscaled, y_test, test_size = .5, random_state = 21)

# 3. Scaling by MinMaxScaler (keeping df for descriptive analysis)
scaler = preprocessing.MinMaxScaler()
X_train = scaler.fit_transform(X_train_temp)
X_val = scaler.transform(X_val_temp)
X_test = scaler.transform(X_test_unscaled)

"""## SVM

We perform a grid search in order to optimize the hyperparameters for SVM.
"""

# defining parameter range
param_grid = {'C': [2,4,6,10,100,1000], 
              'gamma': [0.01, 0.1, 1, 10],
              'kernel': ['rbf']} 
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
  
# fitting the model for grid search
grid.fit(X_train, y_train)

# print best parameter after tuning
print(grid.best_params_)
  
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

"""From the grid search, SVC with hyperparameters of (C=100, gamma=0.1) gives us the highest score.

"""

# Train the SVM model with the optimized hyperparaters
svc = SVC(kernel='rbf', C=100, gamma=0.1).fit(X_train, y_train)

# Accuracy score after fitting the model with the validation set
print("SVM Accuracy on validation set: {:.3f}".format(svc.score(X_val, y_val)))
print('\n')

# AUC 
y_pred_val = svc.predict(X_val)
auc= metrics.roc_auc_score(y_val, y_pred_val)
print('AUC: ', str(auc))
print('\n')

# False positive rate
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
FPR = FP/(FP+TN)
print('False Positive Rate: ', FPR)
print('\n')

# Classification report including precision and recall rates
print(classification_report(y_val, y_pred_val))
print('\n')

# Cofusion Matrix
cm = confusion_matrix(y_val, y_pred_val, labels=svc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=svc.classes_)
disp.plot(cmap=plt.cm.Blues)

# ROC curve
svc_roc = SVC(kernel='rbf', C=100, gamma=0.1, probability = True).fit(X_train, y_train)


decision_scores = svc_roc.decision_function(X_val)
fpr, tpr, thres = roc_curve(y_val, decision_scores)
auc= metrics.roc_auc_score(y_val, y_pred_val)

plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (AUC = {:.3f})'.format(auc))
plt.show()

"""## Naive Bayes"""

# Specify the categorical features
nb = MixedNB(categorical_features=[5, 6, 8, 9, 10])
nb.fit(X_train, y_train)

# Accuracy score
print("NB Accuracy on validation set: {:.3f}".format(nb.score(X_val, y_val)))
print('\n')

nb_pred_val = nb.predict(X_val)

# AUC 
print('AUC: ', metrics.roc_auc_score(y_val, nb_pred_val))
print('\n')

# False positive rate
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
FPR = FP/(FP+TN)
print('False Positive Rate: ', FPR)
print('\n')

# Classification report including precision and recall rates
print(classification_report(y_val, nb_pred_val))
print('\n')

# Confusion matrix
cm = confusion_matrix(y_val, nb_pred_val)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=svc.classes_)
disp.plot(cmap=plt.cm.Blues)

# Plot ROC curve
y_score = nb.predict_proba(X_val)
fpr, tpr, thresholds = roc_curve(y_true=y_val, y_score=y_score[:,1])
auc= metrics.roc_auc_score(y_val, nb_pred_val)

plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (AUC = {:.3f})'.format(auc))
plt.show()

"""## Random Forest Classifier"""

#First we get a benchmark model with default hyperparameter values

#Learning process
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

#Making predictions over validation set
preds = clf.predict(X_val)

#Performance evaluation
print(clf.score(X_train, y_train))
print(clf.score(X_val, y_val))

"""While the model performs well on the validation set, the perfect accuracy score over the training data suggests overfitting. We thus try to optimize hyperparameters. """

#Implementing GridSearchCV for hyperparameter optimization

param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 210, 230],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 3, 5]
}
  
grid = GridSearchCV(RandomForestClassifier(), param_grid, verbose = 3)

# fitting the model for grid search
grid.fit(X_train, y_train)

# print best parameters after tuning
print(grid.best_params_)
  
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

#Refitting the model with optimal values for hyperparameters
clf_tuned = RandomForestClassifier(max_depth = 230, min_samples_leaf= 1, 
                                   min_samples_split = 3, n_estimators = 100)

clf_tuned.fit(X_train, y_train)

#Making predictions over validation set
preds = clf_tuned.predict(X_val)

#Performance evaluation
print("Training Accuracy:", clf_tuned.score(X_train, y_train))
print("Validation Accuracy:", clf_tuned.score(X_val, y_val))

"""Training accuracy slightly decreases which means reduced overfitting while validation accuracy slightly increases. """

#Checking feature importance

feature_imp = pd.DataFrame(clf_tuned.feature_importances_, index = X.columns).sort_values(by=0, ascending=False)
feature_imp.columns = ["Importance"]
feature_imp

#Visualizing feature importance
feature_imp.plot.bar(figsize = (12,8))

"""We see that "Age" has the highest feature importance in predicting bank customer churn, followed by "Balance" and "Estimated Salary" and so on."""

#More Evaluation Metrics apart from Validation Accuracy

#1. Create the confusion matrix
cm = confusion_matrix(y_val, preds)

ConfusionMatrixDisplay(confusion_matrix = cm).plot(cmap=plt.cm.Blues);

#2. Printing classification report
from sklearn.metrics import classification_report

print(classification_report(y_val, y_pred_val))

#3. ROC/AUC 
y_score = clf_tuned.predict_proba(X_val)

fpr, tpr, thresholds = roc_curve(y_true=y_val, y_score=y_score[:,1])

auc= metrics.roc_auc_score(y_val, y_pred_val)


# Plot ROC curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (AUC = {:.3f})'.format(auc))
plt.show()

"""## K-Nearest Neighbours"""

#Baseline model using randomly selected value of K = 3

knn = KNeighborsClassifier(n_neighbors = 3)
knn_base = knn.fit(X_train, y_train)


print("Training set score: {}".format(knn_base.score(X_train, y_train)))
print("Validation set score: {}".format(knn_base.score(X_val, y_val)))

#Hyperparameter Optimization

####Approach 1: Tuning using GridSearchCV

k_range = list(range(1, 31))
param_grid = dict(n_neighbors = k_range)
  
# defining parameter range
grid = GridSearchCV(KNeighborsClassifier(), param_grid, verbose=3)
  
# fitting the model for grid search
grid.fit(X_train, y_train)

# print best parameter after tuning
print(grid.best_params_)
  
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

#Refitting the model with optimal values for hyperparameter 'K'
knn_tuned = KNeighborsClassifier(n_neighbors=15)

knn_tuned.fit(X_train, y_train)

#Performance evaluation
print("Training Accuracy:", knn_tuned.score(X_train, y_train))
print("Validation Accuracy:", knn_tuned.score(X_val, y_val))

####Approach 2: Using Elbow Method to Optimize K

k_range = list(range(1, 31))

#Storing scores in a list
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
 
    scores.append(round(knn.score(X_val, y_val),4))

print(scores)

#Plotting Scores against Values of K 

plt.figure(figsize= (10,6))
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Accuracy Score')
plt.title("KNN Accuracy Scores for different Ks")

"""After K = 9 we don't see much improvement in accuracy. We can try this value."""

#Refitting the model with optimal value for hyperparameter 'K': Elbow Method
knn_tuned2 = KNeighborsClassifier(n_neighbors=9)

knn_tuned2.fit(X_train, y_train)

#Performance evaluation
print("Training Accuracy:", knn_tuned2.score(X_train, y_train))
print("Validation Accuracy:", knn_tuned2.score(X_val, y_val))

"""We see that this value of K yields better results than the GridSearchCV value."""

#More Evaluation Metrics

#1. Making predictions
y_pred_val = knn_tuned2.predict(X_val)

#Create the confusion matrix
cm = confusion_matrix(y_val, y_pred_val)

ConfusionMatrixDisplay(confusion_matrix = cm).plot(cmap=plt.cm.Blues);

#2. Printing classification report
from sklearn.metrics import classification_report

print(classification_report(y_val, y_pred_val))

#3. ROC/AUC 
y_score = clf_tuned.predict_proba(X_val)

fpr, tpr, thresholds = roc_curve(y_true=y_val, y_score=y_score[:,1])

auc= metrics.roc_auc_score(y_val, y_pred_val)


# Plot ROC curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (AUC = {:.3f})'.format(auc))
plt.show()

"""## Stacking"""

pip install mlens

from mlens.ensemble import SuperLearner

#Setting up base learners 
base_learners = [LogisticRegression(),
                RandomForestClassifier(),
                KNeighborsClassifier(),
                MLPClassifier(),
                SVC()]

super_learner = SuperLearner(folds = 10, random_state = 42)

super_learner.add(base_learners)


#Fit to training data
super_learner.fit(X_train, y_train)


#Get base predictions
base_predictions = super_learner.predict(X_train)

#Training the metalearner-- we choose logistic regression
log_reg = LogisticRegression(fit_intercept = False).fit(base_predictions, y_train)

#Printing coefficients for each base learner
print("Coefficients:")
log_reg.coef_

#Making predictions over training set
y_pred_train = log_reg.predict(super_learner.predict(X_train))

#Making predictions over validation set
y_pred_val = log_reg.predict(super_learner.predict(X_val))

from sklearn.metrics import accuracy_score

#Evaluation Metrics

#1. Validation Accuracy
print("Training Accuracy:", accuracy_score(y_train, y_pred_train))
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))

#2. Create the confusion matrix
cm = confusion_matrix(y_val, y_pred_val)

ConfusionMatrixDisplay(confusion_matrix = cm).plot(cmap=plt.cm.Blues);

#3. Printing classification report

print(classification_report(y_val, y_pred_val))

#4. ROC/AUC 
y_score = log_reg.predict_proba(super_learner.predict(X_val))

fpr, tpr, thresholds = roc_curve(y_true=y_val, y_score=y_score[:,1])

auc= metrics.roc_auc_score(y_val, y_pred_val)


# Plot ROC curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (AUC = {:.3f})'.format(auc))
plt.show()

"""# Logistic Regression

## Method: sklearn
"""

from sklearn.linear_model import LogisticRegression

# performing grid search over potential hyperparameters
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'penalty': ['l1', 'l2', 'elasticnet', 'none'],
              'solver': ['lbfgs', 'liblinear', 'sag', 'saga']}


# Create a GridSearchCV object
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)

# Fit the GridSearchCV object on training data
grid_search.fit(X_train, y_train)

# Extract the best hyperparameters
best_params = grid_search.best_params_

# Create a new classifier with the best hyperparameters
clf = LogisticRegression(**best_params)

# Train the new classifier on the training data
clf.fit(X_train, y_train)

# Evaluate performance on validation set
val_score = clf.score(X_val, y_val)
print("Validation set score: {:.4f}".format(val_score))

val_score

best_params

# use sklearn class
clf = LogisticRegression()

# call the function fit() to train the class instance
clf.fit(X_train,y_train)

# Evaluate performance on validation set
val_score = clf.score(X_val, y_val)
print("Validation set score: {:.4f}".format(val_score))

# Evaluate performance on test set
test_score = clf.score(X_test, y_test)
print("Test set score: {:.4f}".format(test_score))

"""An accuracy of 0.8188 on the validation set means that the model correctly predicted the target variable for about 81.88% of the samples in the validation set. Similarly, an accuracy of 0.7988 on the test set means that the model correctly predicted the target variable for about 79.88% of the samples in the test set.

## Evaluating the Model
"""

# fitting the model on the validation set
y_pred = clf.predict(X_val)

# evaluation metric I: confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class_labels = [0, 1]

# compute confusion matrix
cm = confusion_matrix(y_true=y_val, y_pred=y_pred, labels=class_labels)

# display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)

from sklearn.metrics import precision_score, recall_score, roc_curve, roc_auc_score

# evaluation metric II and III:
# compute precision and recall rates
precision = precision_score(y_true=y_val, y_pred=y_pred)
recall = recall_score(y_true=y_val, y_pred=y_pred)

print("Precision: {:.3f}".format(precision))
print("Recall: {:.3f}".format(recall))

# evaluation metric IV
# compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true=y_val, y_score=y_pred)
auc = roc_auc_score(y_true=y_val, y_score=y_pred)

# plot ROC curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (AUC = {:.3f})'.format(auc))
plt.show()

# since accuracy high, AUC low - studying the distribution of the data
import pandas as pd

# count the number of samples for each class in the training set
train_counts = pd.Series(y_train).value_counts()

# count the number of samples for each class in the validation set
val_counts = pd.Series(y_val).value_counts()

# print the counts
print("Training data:")
print(train_counts)
print("Validation data:")
print(val_counts)

"""## Feedforward Neural Networks"""

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import r2_score, accuracy_score, precision_recall_fscore_support

# performing grid search for optimal hyperparameters
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search over
param_grid = {
    'hidden_layer_sizes': [(5,), (10,), (5,5), (10,10)],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.001, 0.01, 0.1],
    'max_iter': [100, 500]
}

# Create a MLPClassifier object
mlp = MLPClassifier()

# Create a GridSearchCV object and fit it to the training data
grid_search = GridSearchCV(mlp, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and corresponding validation score
print("Best hyperparameters:", grid_search.best_params_)
print("Validation accuracy:", grid_search.best_score_)

# evaluation metric I: Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, roc_curve, roc_auc_score

# Predict labels for test set using the best model from grid search
y_pred = grid_search.predict(X_val)

# Compute confusion matrix
class_labels = [0, 1]
cm = confusion_matrix(y_true=y_val, y_pred=y_pred, labels=class_labels)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)

# evaluation metric II and III:
# Compute accuracy, precision, and recall rates
accuracy = grid_search.score(X_val, y_val)
precision = precision_score(y_true=y_val, y_pred=y_pred)
recall = recall_score(y_true=y_val, y_pred=y_pred)

print("Accuracy: {:.3f}".format(accuracy))
print("Precision: {:.3f}".format(precision))
print("Recall: {:.3f}".format(recall))

# evaluation metric IV:
# Compute ROC curve and AUC
y_score = grid_search.predict_proba(X_val)[:,1]
fpr, tpr, thresholds = roc_curve(y_true=y_val, y_score=y_score)
auc = roc_auc_score(y_true=y_val, y_score=y_score)

# Plot ROC curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (AUC = {:.3f})'.format(auc))
plt.show()

"""## Building a Keras with the architecture defined by GridSearchCV

Just for my knowledge trying to visualise the architecture of NN
"""

import keras.models
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

# Create Keras Sequential model with specified architecture and hyperparameters
model = Sequential()
model.add(Dense(5, input_dim=X_train.shape[1], activation='sigmoid', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(1, activation='linear'))

# Compile the model with specified optimizer and loss function
model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(X_train, y_train, epochs = 100)

y_pred = model.predict(X_val)

y_pred

# Saving the model
keras.models.save_model(model, "/folder/model.pb")

# Loading the model
mod = keras.models.load_model("/folder/model.pb")

model.summary()

from tensorflow.keras.utils import plot_model

plot_model(model, show_shapes = True)

!pip install graphviz ann_visualizer

from ann_visualizer.visualize import ann_viz

ann_viz(model, title="CLV NN Viz", filename="model.png")

from IPython.display import Image
Image(filename = "/content/gdrive/MyDrive/ML Final Project/image.png")

"""## Best-Performing Model

### Testing the model
"""

#Random Forest Classifier on Testing Set
clf_tuned = RandomForestClassifier(max_depth = 230, min_samples_leaf= 1, 
                                   min_samples_split = 4, n_estimators = 200)

clf_tuned.fit(X_train, y_train)

#Making predictions over testing set
preds = clf_tuned.predict(X_test)

#Performance evaluation
print("Training Accuracy:", clf_tuned.score(X_train, y_train))
print("Testing Accuracy:", clf_tuned.score(X_test, y_test))

#Evaluation Metrics

#Create the confusion matrix
cm = confusion_matrix(y_test, preds)

ConfusionMatrixDisplay(confusion_matrix = cm).plot(cmap=plt.cm.Blues);

#Printing classification report
from sklearn.metrics import classification_report

print(classification_report(y_test, preds))

#ROC/AUC 
y_score = clf_tuned.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score[:,1])

auc= metrics.roc_auc_score(y_test, preds)


# Plot ROC curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (AUC = {:.3f})'.format(auc))
plt.show()

"""### Evaluating success and failure samples"""

y_pred_df = pd.DataFrame(preds)
Y_test_df = pd.DataFrame(y_test)

y_df = pd.merge(Y_test_df, y_pred_df, left_index = True, right_index = True)
y_df.columns = "Actual", "Predicted"
y_df

# Failure samples: samples for which our model can not correctly predict their labels

failures = y_df[y_df["Actual"] != y_df["Predicted"]]
failures.head()

# Success samples: samples for which you model can correctly predict their labels

successes = y_df[y_df["Actual"] == y_df["Predicted"]]
successes.head()

from prettytable import PrettyTable

#Converting features dataframe to an array
X_test_unscaled = np.array(X_test_unscaled)
X_test_unscaled

#Features for 5 success samples
result_table_successes = PrettyTable()
result_table_successes.field_names = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'Geography_Germany',
       'Geography_Spain', 'Gender_Male', "Actual", "Predicted"]

#Pulling out feature values for 5 successful predictions and storing into a table
for j in successes[:5].index:
    result_table_successes.add_row([X_test_unscaled[j,0], X_test_unscaled[j,1], X_test_unscaled[j,2], 
                          X_test_unscaled[j,3], X_test_unscaled[j,4], X_test_unscaled[j,5], 
                          X_test_unscaled[j,6], X_test_unscaled[j,7], X_test_unscaled[j,8],
                          X_test_unscaled[j,9], X_test_unscaled[j,10],
                          successes.loc[j, "Predicted"], successes.loc[j, "Actual"]])

print(result_table_successes)

#Features for 5 failures 

result_table_failures = PrettyTable()
result_table_failures.field_names = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'Geography_Germany',
       'Geography_Spain', 'Gender_Male', "Actual", "Predicted"]

#Pulling out feature values for 5 failed predictions and storing into table
for i in failures[:5].index:
    result_table_failures.add_row([X_test_unscaled[i,0], X_test_unscaled[i,1], X_test_unscaled[i,2], 
                          X_test_unscaled[i,3], X_test_unscaled[i,4], X_test_unscaled[i,5], 
                          X_test_unscaled[i,6], X_test_unscaled[i,7], X_test_unscaled[i,8],
                          X_test_unscaled[i,9], X_test_unscaled[i,10],
                          failures.loc[i, "Predicted"], failures.loc[i, "Actual"]])

#result_table.add_row(["-"*10, "-"*10, "-"*10, "-"*10, "-"*10, "-"*10, "-"*10, "-"*10])
print(result_table_failures)

X_test_df = pd.DataFrame(X_test_unscaled)

test_df = pd.merge(X_test_df, y_df, left_index = True, right_index = True)
test_df.columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'Geography_Germany',
       'Geography_Spain', 'Gender_Male', "Actual", "Predicted"]
test_df.groupby("Actual").mean()

"""From the analysis above we try to dig into the reasons for the failure cases by comparing the values of their features to the mean value of the features for their actual classes. We see that the failure cases stem from the fact that these samples' features were more closely aligned with the other class' feature values than their actual class'.

For example, Class 0 has a slightly higher mean value for Age than Class 1. We see that most of the failures that were misclassified as negative when they were actually positive have higher values for "Age". On the other hand, the one sample misclassified as positive when it is actually negative has a low value for age, as do most of the samples under class 1. 
"""