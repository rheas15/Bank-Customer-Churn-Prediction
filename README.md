# **Bank Customer Churn Prediction with Machine Learning**

## **Overview**

This project aims to develop a predictive model to identify bank customers who are at risk of churning. By accurately predicting churn, banks can proactively implement retention strategies to retain customers and minimize revenue loss. The project utilizes a dataset sourced from Kaggle, comprising information from 10,000 bank customers, including various demographic and engagement features.

## **Methodology**

- Data Preprocessing: The dataset undergoes preprocessing steps to handle missing values, encode categorical variables, and normalize numerical features.
- Exploratory Data Analysis (EDA): EDA techniques are employed to gain insights into the distribution of features, identify correlations, and understand patterns in the data.
- Model Selection: Seven machine learning models are evaluated for predicting customer churn, including K-Nearest Neighbor, Logistic Regression, Neural Network, Naive Bayes, Support Vector Machine, Random Forest, and a stacked model.
- Model Evaluation: Performance metrics such as accuracy score, AUC, and recall rate are used to evaluate the effectiveness of each model in predicting churn.
- Best Performing Model: The Random Forest classifier emerges as the best-performing model, achieving high accuracy and recall rates on the testing set.
- Corner Case Analysis: Misclassifications in corner cases for outliers are analyzed to understand model limitations and the need for human oversight.

## **Conclusion**

The project concludes that predictive modeling, particularly using the Random Forest classifier, offers valuable insights into customer churn prediction for banks. While the model demonstrates high accuracy, challenges such as imbalanced datasets and model limitations in handling outliers are identified. Future research could focus on addressing these challenges and further improving churn prediction methodologies.

## **Implementation**

The project implementation includes the Python source code for data preprocessing, model training, and evaluation. 
The main report provides a detailed overview of the methodology, analysis, results, and discussion.
