# Module17
# Practical Application III: Classification

Link to jupyter notebook: https://github.com/amanvashishtgould/Module17/blob/main/prompt_III.ipynb

Overview

In this practical application, our goal is to compare the performance of the classifiers, namely, K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines, using a dataset related to marketing bank products over the telephone.


Understanding the Data

The dataset is from a Portuguese banking institution and is a collection of the results of 17 marketing campaigns conducted between May 2008 and November 2010. During these phone campaigns, the client was offered an attractive long-term deposit application, with good interest rates. Information related to client personal data, bank data, and information related to contact (including past contacts and campaigns) along with the target variable of whether or not the client subscribed to the long-term deposit are available in the dataset.

There are 41188 rows and 21 columns in this dataset, with a couple of numeric (int or float) type columns and several object type columns. There are no null values, but there are a couple of duplicated values, which are removed. The target variable has two values, with clients/contacts who subscribed to the deposit making up about 11% of the data, and non-subscribers making up the rest approximately 89% of the data.


*Business Objective*

The business objective is to predict whether a customer will or will not subscribe to the long-term deposit application. This is a classification problem, where the goal is to classify the target variable (yes or no for subscribed) using various attributes given.


Data Preparation: Feature Engineering

Using just the bank information features (first seven columns), the features are first encoded using Category encoders. LeaveOneOutEncoder is used for this purpose to minimize bias. A quick seaborn-based heatmap of correlation of different variables is plotted and it shows minimal multicollinearity. Since the target data is imbalanced, so Synthetic Minority Oversampling Technique (SMOTE) is utilized to balance out the target classes.


Data Preparation: Train and Test Split & Scaling/Standardization

The data is split into train test sets using test size of 20%, and after that it is standardized/ scaled/ normalized using StandardScaler().


Modeling: Baseline Dummy and Initial Classification Models

Before we build our first model, we establish a baseline using a DummyClassifer that our models should aim to beat. Then initial classification models are run using default parameters with Logistic Regression, K Nearest Neighbor (KNN), Decision Trees, and Support Vector Machines (SVM) classifier. The time to fit these models (in seconds), training accuracy and test accuracy of these are summarized below. Additionally, it was reasoned that F-1 score is a better score as it maximizes both recall and precision, as it would be preferable to minimize both false negatives (where an actual subscriber is predicted as non-subscribed--this scenario can lead to loss of potential important clients/subscribers) and false positives (where an actual non-subscriber is predicted as subscribed--this scenario can lead to a waste of time and resources). These F-1 scores are also shown below in parenthesis.

Dummy Classifier
Time to train: 0.003
Training accuracy(f1): 0.50 (0.67)
Test accuracy(f1): 0.50 (0.66)

Logistic Regression Classifier
Time to train: 0.04
Training accuracy(f1): 0.61 (0.63)
Test accuracy(f1): 0.61 (0.62)

KNN Classifier
Time to train: 0.17
Training accuracy(f1): 0.95 (0.95)
Test accuracy(f1): 0.90 (0.90)

Decision tree Classifier
Time to train: 0.06
Training accuracy(f1): 1(1)
Test accuracy(f1): 0.95(0.96)

SVM Classifier
Time to train: 267
Training accuracy(f1): 0.76(0.74)
Test accuracy(f1): 0.76 (0.75)

Model Performance Discussion: In terms of train as well as test accuracy, KNN and decision tree performed best. Their test scores were 90% for KNN and 95.9% for Decision Trees. All models had a test accuracy higher than test accuracy of dummy classifier (50%). However, the test F-1 scores were not always high, e.g., Logistic regression which had the lowest test F-1 score (62%) of all models except dummy classifier (which had F1 score of 66%). KNN and Decision trees had the highest F-1 score of 90% and 96%, respectively. In terms of time, Logisitic Regression and Decision Trees were the fastest, followed by KNN. SVM took substantially longer than all others.


Modeling: Improving the models using Hyperparameter tuning and adding additional performance metrics

Logistic regression, KNN, decision trees, and SVM models are tuned for hyperparameters using GridSearchCV. Furthermore, F-1 test scores are used as the scoring criteria such the F1 scores are maximized in the GridSearchCV. Classification reports are calculated and Receiver Operating Characteristic (ROC) curves are additionally plotted. The models are evaluated for their F-1 test scores and ROC Area Under Curve (AUC). These are summarized below, along with model fit times(in minutes):

Logistic Regression
Mean fit time: 0.02
Test F1(AUC): 0.61 (0.66)

KNN
Mean fit time: 0.18
Test F1(AUC): 0.95 (0.95)

Decision Trees
Mean fit time: 0.04
Test F1(AUC): 0.96 (0.96)

SVM
Mean fit time: 25.60
Test F1(AUC): 0.25 (0.51)


Additionally, as the paper by Moro&Laureno metioned Cumulative Gains/Lift curve as a commonly used metric in marketing campaigns, so this is also plotted for each model.

Hyperparameterized Models' Performance Discussion:
Decision Tree classifier performed best in terms of both F-1 test score and ROC AUC, followed closely by KNN model. These two models also were the fastest to fit/run. Their F-1 scores were 96%(Decision trees) and 95%(KNN), and their AUCs were 0.96(Decision Trees) and 0.95(KNN). Decision tree was fit/run faster than KNN.
The other two models (Logistic & SVM) did not really improve from their non-hyperparameterized verions, and SVM was very computationally inefficient (took about half an hour to fit wher SVC(probability=False) and atleast two hours when probability=True).
Lastly, in terms of Cumulative gain curves, Decision Trees followed by KNN were the best. For decision trees, sampling 50% of the clients could get about 95% subscribers and for KNN, sampling 50% of the clients could get more than 90% subscribers.
Based on these several performance indicators/metrics, Decision Trees is the best model. Then it is explored further below in terms of feature importance. The most important features in the best perming decision trees model are Loan, followed by default, and housing status of the client. This information can be used by banks to target clients for subscription.


Modeling: Additional features-does the modeling performance improve?

The question asked here was if the best performing model can be improved further by adding additional columns about call details and economic&social variables. Correlations heatmap showed some significant correlations among these additional variables and columns correlated highly (>0.6) with others were dropped.
This decision tree model with additional variables related to call information and economic&social indices performed well and had a test score of 91% and ROC AUC of 0.91. However it still did not outperform the hyperparameter tuned decision tree model from earlier which had a test score of 96% and ROC AUC of 0.959. In terms of cumulative gains, this additional features’ model showed that sampling 50% of the clients could get 85% subscribers, while the previous best performing decision tree model showed a somewhat higher value of approximately 95% subscribers.


Recommendations and Next steps (for Deployment)

The most important features for whether or not a contact would subscribe to the deposit are housing, default, and loan characteristics of the contact/client.
This is found with the best performing model called decision trees which has a test score (F-1 score) of 96%. ROC area under the curve for this model is also high, about 0.96. Cumulative gains curve shows that for the best performing model, with sampling just 50% of the clients/contacts, it is possible to net about 95% term subscribers.
Next steps would involve running more efficient models that are popular in the machine learning world today to see if the accuracy or performance metrics improve. this would involve classifiers like Extreme Gradient Boosting Classifier, Native Bayes Classifier, and especially the Random Forests Classifier (as it uses a combination of several decision trees and can perhaps be better than the single decision tree used here).

