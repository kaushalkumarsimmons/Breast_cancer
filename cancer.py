#  the code and the algorithm used to predict the Breast cancer
#-----import library for data handling-----#
import pandas as pd
import numpy as np
import itertools


#-----import library for grphical ineterpretation-----# 
import matplotlib.pyplot as plt
#plt.style.use('fivethirtyeight')  
import matplotlib.gridspec as gridspec
import seaborn as sns 

sns.set(style='whitegrid', palette='muted', font_scale=2) 
from pylab import rcParams
rcParams['figure.figsize'] = 12, 8   
import os
##'ggplot','fivethirtyeight','bmh' is clour scheme
##Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
##set the default color palette for figures #'whitegrid'=get white background
##figure colour = palette='dark' #font_scale is word size
##There are six variations of the default theme, called deep, muted, pastel, bright, dark, and colorblind.
##Eg: from pylab import rcParams rcParams['figure.figsize'] = 5, 10This makes the figure's width 5 inches, and its height 10 inches.


#-----Reading the datasets-----#
df=pd.read_csv('dataR2.csv')


#-----Understanding data structure-----#
df.isnull().sum()  
##This will give number of NaN values in every column.

#-----data features-----#
df.describe()  

#------classification labels-----#
sns.countplot(x='Classification',data=df)
plt.show()

#-----groupby() function is used to split the data into groups based on some criteria-----#
grouped = df.groupby('Classification').agg({'Age':['mean', 'std', min, max], 
                                       'Glucose':['mean', 'std', min, max],
                                       'BMI':['mean', 'std', min, max],
                                       'Insulin':['mean', 'std', min, max],
                                       'HOMA':['mean', 'std', min, max],
                                       'Leptin':['mean', 'std', min, max],
                                       'Adiponectin':['mean', 'std', min, max],
                                       'Resistin':['mean', 'std', min, max],
                                       'MCP.1':['mean', 'std', min, max]
                                      })
grouped.columns = ["_".join(x) for x in grouped.columns.ravel()] 
grouped
##joins by(_),each attribute with describe


#-----understanding the ditribution of features in two classes-----#
columns=df.columns[:9]      
plt.figure(figsize=(15,90)) 
gs = gridspec.GridSpec(15, 1)  

for i, cn in enumerate(df[columns]): 
    ax = plt.subplot(gs[i]) 
    sns.distplot(df[cn][df.Classification == 0], bins=20) 
    sns.distplot(df[cn][df.Classification == 1], bins=20)
    ax.set_xlabel('')
    plt.legend(df["Classification"])
    ax.set_title('Histogram of feature:' + str(cn)) 
plt.show()
##:9 no. of feature(dig. illustration)
##'15' width, '90' height 
##15 verticle space, 1 horizontal space
##enumerate: Tells one by one
##:ax-object or array of Axes objects.
##subplot() function can be called to plot two or more plots in one figure. 
##width of histogram(bin=20)
##sequence of character (#+ str(cn))

#-----correlation plot-----#
sns.heatmap(df[df.columns[:9]].corr(),annot=True,cmap='RdYlGn')
plt.show()
## A heat map is a two-dimensional representation of data in which values are represented by colors. maps allow the viewer to understand complex data sets.There can be many ways to display heat maps, but they all share one thing in common -- they use color to communicate relationships between data values that would be would be much harder to understand if presented numerically in a spreadsheet.
## annot parameter which will add correlation numbers to each cell in the visuals.
##cmap='RdYlGn' for colour assignment


df.corr()

# importing library for machine learning
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

# testing training data segregation
Classification=df['Classification']
data=df[df.columns[:9]]
train,test=train_test_split(df,test_size=0.25,random_state=0,stratify=df['Classification'])# stratify the outcome
train_X=train[train.columns[:9]]
test_X=test[test.columns[:9]]
train_Y=train['Classification']
test_Y=test['Classification']


# building models
abc=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree', 'Random forest', 'Naive Bayes']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),
        KNeighborsClassifier(n_neighbors=3),DecisionTreeClassifier(), 
        RandomForestClassifier(n_estimators=100,random_state=0), GaussianNB()]
for i in models:
    model = i
    model.fit(train_X,train_Y)
    prediction=model.predict(test_X)
    abc.append(metrics.accuracy_score(prediction,test_Y))
models_dataframe=pd.DataFrame(abc,index=classifiers)   
models_dataframe.columns=['Accuracy']
models_dataframe.sort_values(['Accuracy'], ascending=[0])


modelRF= RandomForestClassifier(n_estimators=100,random_state=0)
modelRF.fit(train_X,train_Y)
predictionRF=modelRF.predict(test_X)
pd.Series(modelRF.feature_importances_,index=train_X.columns).sort_values(ascending=False)


#-----for K-fold cross validation-----#
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score 
from sklearn.preprocessing import StandardScaler 
#score evaluation


kfold = KFold(n_splits=10, random_state=None) 


#-----Gaussian Standardisation-----#
features=df[df.columns[:9]]
features_standard=StandardScaler().fit_transform(features)
X=pd.DataFrame(features_standard,columns=['HOMA', 'Glucose', 'Adiponectin', 'MCP.1', 
                                          'Insulin', 'BMI', 'Resistin', 'Age','Leptin'])
X['Classification']=df['Classification']


#-----k- fold cross validation-----#
xyz=[]
accuracy=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree', 'Random forest', 'Naive Bayes']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),
        KNeighborsClassifier(n_neighbors=3),DecisionTreeClassifier(), 
        RandomForestClassifier(n_estimators=100,random_state=0), GaussianNB()]

for i in models:
    model = i
    cv_result = cross_val_score(model,X[X.columns[:9]], X['Classification'], cv = kfold, scoring = "accuracy")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    accuracy.append(cv_result)

cv_models_dataframe=pd.DataFrame(xyz, index=classifiers)   
cv_models_dataframe.columns=['CV Mean']    
cv_models_dataframe
cv_models_dataframe.sort_values(['CV Mean'], ascending=[0])


#-----Box plot for Cross validation accuracies with different classifiers-----#
box=pd.DataFrame(accuracy,index=[classifiers])
boxT = box.T
ax = sns.boxplot(data=boxT, orient="h", palette="Set2", width=.6)
ax.set_yticklabels(classifiers)
ax.set_title('Cross validation accuracies with different classifiers')
ax.set_xlabel('Accuracy')
plt.show()


from sklearn import metrics
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score, 
                             classification_report, f1_score, average_precision_score, precision_recall_fscore_support)


#-----ROC curve for SVM model-----#
modelSVMlinear=svm.SVC(kernel='linear', probability=True)
modelSVMlinear.fit(train_X,train_Y)
y_pred_prob_SVMlinear = modelSVMlinear.predict_proba(test_X)[:,1]
fpr_SVMlinear, tpr_SVMlinear, thresholds_SVMlinear = roc_curve(test_Y-1, y_pred_prob_SVMlinear, pos_label=0)
roc_auc_SVMlinear = auc(fpr_SVMlinear, tpr_SVMlinear)
precision_SVMlinear, recall_SVMlinear, th_SVMlinear = precision_recall_curve(test_Y, y_pred_prob_SVMlinear)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_SVMlinear, tpr_SVMlinear, label='SVM linear (area = %0.3f)' % roc_auc_SVMlinear)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves from the SVM models')
plt.legend(loc='best')
plt.show()

##plt.plot(fpr_SVMlinear[2], tpr_SVMlinear[2], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2]) plt.plot([1, 2], [1, 2], color='navy', lw=lw, linestyle='--')

##function of kernel is to take data as input and transform it into the required form, For example linear, nonlinear, polynomial, radial basis function (RBF), and sigmoid.


#-----ROC curve for Logistic regression model-----#
modellogisticregession=LogisticRegression()
modellogisticregession.fit(train_X,train_Y)
y_pred_prob_logisticregression = modellogisticregession.predict_proba(test_X)[:,1]
fpr_logisticregression, tpr_logisticregression, thresholds_logisticregression = roc_curve(test_Y-1, y_pred_prob_logisticregression, pos_label=0)
roc_auc_logisticregression = auc(fpr_logisticregression, tpr_logisticregression)
precision_logisticregression, recall_logisticregression, th_logisticregression = precision_recall_curve(test_Y, y_pred_prob_logisticregression)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_logisticregression, tpr_logisticregression, label='Logistic Regression (area = %0.3f)' % roc_auc_logisticregression)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves from the Logistic regression models')
plt.legend(loc='best')
plt.show()

#-----ROC curve for KNN model-----#
modelKNN=KNeighborsClassifier(n_neighbors=3)
modelKNN.fit(train_X,train_Y)
y_pred_prob_KNN = modelKNN.predict_proba(test_X)[:,1]
fpr_KNN, tpr_KNN, thresholds_KNN = roc_curve(test_Y-1, y_pred_prob_KNN, pos_label=0)
roc_auc_KNN = auc(fpr_KNN, tpr_KNN)
precision_KNN, recall_KNN, th_KNN = precision_recall_curve(test_Y, y_pred_prob_KNN)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_KNN, tpr_KNN, label='KNN (area = %0.3f)' % roc_auc_KNN)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves from the KNN models')
plt.legend(loc='best')
plt.show()


#-----ROC curve for RandomForest model-----#
modelRandomForest=RandomForestClassifier(n_estimators=100,random_state=0)
modelRandomForest.fit(train_X,train_Y)
y_pred_prob_RandomForest = modelRandomForest.predict_proba(test_X)[:,1]
fpr_RandomForest, tpr_RandomForest, thresholds_RandomForest = roc_curve(test_Y-1, y_pred_prob_RandomForest, pos_label=0)
roc_auc_RandomForest = auc(fpr_RandomForest, tpr_RandomForest)
precision_RandomForest, recall_RandomForest, th_RandomForest= precision_recall_curve(test_Y, y_pred_prob_RandomForest)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_RandomForest, tpr_RandomForest, label='RandomForest (area = %0.3f)' % roc_auc_RandomForest)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves from the Random Forest models')
plt.legend(loc='best')
plt.show()


#-----ROC curve for GaussianNB model-----#
modelGaussianNB=GaussianNB()
modelGaussianNB.fit(train_X,train_Y)
y_pred_prob_GaussianNB = modelGaussianNB.predict_proba(test_X)[:,1]
fpr_GaussianNB, tpr_GaussianNB, thresholds_GaussianNB = roc_curve(test_Y-1, y_pred_prob_GaussianNB, pos_label=0)
roc_auc_GaussianNB = auc(fpr_GaussianNB, tpr_GaussianNB)
precision_GaussianNB, recall_GaussianNB, th_GaussianNB = precision_recall_curve(test_Y, y_pred_prob_GaussianNB)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_GaussianNB, tpr_GaussianNB, label='GaussianNB (area = %0.3f)' % roc_auc_GaussianNB)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves from the GaussianNB models')
plt.legend(loc='best')
plt.show()


#-----ROC curve for DecisionTree model-----#
modelDecisionTree=DecisionTreeClassifier()
modelDecisionTree.fit(train_X,train_Y)
y_pred_prob_DecisionTree = modelDecisionTree.predict_proba(test_X)[:,1]
fpr_DecisionTree, tpr_DecisionTree, thresholds_DecisionTree = roc_curve(test_Y-1, y_pred_prob_DecisionTree, pos_label=0)
roc_auc_DecisionTree = auc(fpr_DecisionTree, tpr_DecisionTree)
precision_DecisionTree, recall_DecisionTree, th_DecisionTree = precision_recall_curve(test_Y, y_pred_prob_DecisionTree)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_DecisionTree, tpr_DecisionTree, label='DecisionTree (area = %0.3f)' % roc_auc_DecisionTree)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves from the Decision Tree models')
plt.legend(loc='best')
plt.show()


#-----ROC curve for svmradial model-----#
modelSVMRadial=svm.SVC(kernel='rbf', probability=True)
modelSVMRadial.fit(train_X,train_Y)

y_pred_prob_SVMRadial = modelSVMRadial.predict_proba(test_X)[:,1]
fpr_SVMRadial, tpr_SVMRadial, thresholds_SVMRadial = roc_curve(test_Y-1, y_pred_prob_SVMRadial, pos_label=0)
roc_auc_SVMRadial = auc(fpr_SVMRadial, tpr_SVMRadial)
precision_SVMRadial, recall_SVMRadial, th_SVMRadial = precision_recall_curve(test_Y, y_pred_prob_SVMRadial)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_SVMRadial, tpr_SVMRadial, label='SVMRadial (area = %0.3f)' % roc_auc_SVMRadial)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves from the svm radial models')
plt.legend(loc='best')
plt.show()


#-----ROC curve for SVM model-----#
modelSVMlinear=svm.SVC(kernel='linear', probability=True)
modelSVMlinear.fit(train_X,train_Y)
y_pred_prob_SVMlinear = modelSVMlinear.predict_proba(test_X)[:,1]
fpr_SVMlinear, tpr_SVMlinear, thresholds_SVMlinear = roc_curve(test_Y-1, y_pred_prob_SVMlinear, pos_label=0)
roc_auc_SVMlinear = auc(fpr_SVMlinear, tpr_SVMlinear)
precision_SVMlinear, recall_SVMlinear, th_SVMlinear = precision_recall_curve(test_Y, y_pred_prob_SVMlinear)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_SVMlinear, tpr_SVMlinear, label='SVM linear (area = %0.3f)' % roc_auc_SVMlinear)

# ROC curve for Logistic regression model
modellogisticregession=LogisticRegression()
modellogisticregession.fit(train_X,train_Y)
y_pred_prob_logisticregression = modellogisticregession.predict_proba(test_X)[:,1]
fpr_logisticregression, tpr_logisticregression, thresholds_logisticregression = roc_curve(test_Y-1, y_pred_prob_logisticregression, pos_label=0)
roc_auc_logisticregression = auc(fpr_logisticregression, tpr_logisticregression)
precision_logisticregression, recall_logisticregression, th_logisticregression = precision_recall_curve(test_Y, y_pred_prob_logisticregression)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_logisticregression, tpr_logisticregression, label='Logistic Regression (area = %0.3f)' % roc_auc_logisticregression)

# ROC curve for RandomForest model
modelRandomForest=RandomForestClassifier(n_estimators=100,random_state=0)
modelRandomForest.fit(train_X,train_Y)
y_pred_prob_RandomForest = modelRandomForest.predict_proba(test_X)[:,1]
fpr_RandomForest, tpr_RandomForest, thresholds_RandomForest = roc_curve(test_Y-1, y_pred_prob_RandomForest, pos_label=0)
roc_auc_RandomForest = auc(fpr_RandomForest, tpr_RandomForest)
precision_RandomForest, recall_RandomForest, th_RandomForest= precision_recall_curve(test_Y, y_pred_prob_RandomForest)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_RandomForest, tpr_RandomForest, label='RandomForest (area = %0.3f)' % roc_auc_RandomForest)

# ROC curve for GaussianNB model
modelGaussianNB=GaussianNB()
modelGaussianNB.fit(train_X,train_Y)
y_pred_prob_GaussianNB = modelGaussianNB.predict_proba(test_X)[:,1]
fpr_GaussianNB, tpr_GaussianNB, thresholds_GaussianNB = roc_curve(test_Y-1, y_pred_prob_GaussianNB, pos_label=0)
roc_auc_GaussianNB = auc(fpr_GaussianNB, tpr_GaussianNB)
precision_GaussianNB, recall_GaussianNB, th_GaussianNB = precision_recall_curve(test_Y, y_pred_prob_GaussianNB)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_GaussianNB, tpr_GaussianNB, label='GaussianNB (area = %0.3f)' % roc_auc_GaussianNB)

# ROC curve for DecisionTree model
modelDecisionTree=DecisionTreeClassifier()
modelDecisionTree.fit(train_X,train_Y)
y_pred_prob_DecisionTree = modelDecisionTree.predict_proba(test_X)[:,1]
fpr_DecisionTree, tpr_DecisionTree, thresholds_DecisionTree = roc_curve(test_Y-1, y_pred_prob_DecisionTree, pos_label=0)
roc_auc_DecisionTree = auc(fpr_DecisionTree, tpr_DecisionTree)
precision_DecisionTree, recall_DecisionTree, th_DecisionTree = precision_recall_curve(test_Y, y_pred_prob_DecisionTree)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_DecisionTree, tpr_DecisionTree, label='DecisionTree (area = %0.3f)' % roc_auc_DecisionTree)

# ROC curve for KNN model
modelRandomforest=KNeighborsClassifier(n_neighbors=3)
modelKNN.fit(train_X,train_Y)
y_pred_prob_KNN = modelKNN.predict_proba(test_X)[:,1]
fpr_KNN, tpr_KNN, thresholds_KNN = roc_curve(test_Y-1, y_pred_prob_KNN, pos_label=0)
roc_auc_KNN = auc(fpr_KNN, tpr_KNN)
precision_KNN, recall_KNN, th_KNN = precision_recall_curve(test_Y, y_pred_prob_KNN)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_KNN, tpr_KNN, label='KNN (area = %0.3f)' % roc_auc_KNN)

#ROC curve for SVM model
modelSVMRadial=svm.SVC(kernel='rbf', probability=True)
modelSVMRadial.fit(train_X,train_Y)
y_pred_prob_SVMRadial = modelSVMRadial.predict_proba(test_X)[:,1]
fpr_SVMRadial, tpr_SVMRadial, thresholds_SVMRadial = roc_curve(test_Y-1, y_pred_prob_SVMRadial, pos_label=0)
roc_auc_SVMRadial = auc(fpr_SVMRadial, tpr_SVMRadial)
precision_SVMRadial, recall_SVMRadial, th_SVMRadial = precision_recall_curve(test_Y, y_pred_prob_SVMRadial)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_SVMRadial, tpr_SVMRadial, label='SVMRadial (area = %0.3f)' % roc_auc_SVMRadial)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves from the investigated models')
plt.legend(loc='best')
plt.show()


# t-SNA visualiztaion of data-sets
df_std = StandardScaler().fit_transform(df)
y = df.iloc[:,-1].values
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0)
x_t = tsne.fit_transform(df_std)
color_map = {2:'red', 1:'blue'}
plt.figure()
plt.figure()
plt.scatter(x_t[np.where(y == 0), 0], x_t[np.where(y == 0), 1], marker='x', color='g', 
            linewidth='1', alpha=0.8, label='No Cancer')
plt.scatter(x_t[np.where(y == 1), 0], x_t[np.where(y == 1), 1], marker='v', color='r',
            linewidth='1', alpha=0.8, label='Cancer')

plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper right')
plt.title('t-SNE visualization of diabetes data')
plt.show()


