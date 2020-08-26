import numpy as np
import pandas as pd
import sklearn as sks
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler
from sklearn.metrics import jaccard_score,log_loss,f1_score
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree


from datetime import timedelta
#load the datset
data = pd.read_csv('loan_train.csv')
data.head()
print("csv =  \n" ,data.head())

#checking null values in the dataset
data.isnull()
print("null = \n", data.isnull())

#droping unwanted data columns
data.drop([ 'Unnamed: 0' , 'Unnamed: 0.1'  ],axis =1 ,  inplace=True )
print("after droping column = \n ", data.head(20))

#exploring the data
sns.boxplot(x ='education', y ='age' , data = data)
plt.title("education based on the age ")
#plt.show()

sns.distplot(data['Principal'])
#plt.show()

Var_Corr = data.corr()
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
#plt.show()

sns.countplot(x = 'Gender', data = data)
plt.title("count of male and female ")
#plt.show()

#converting the date columns into acceptable format
data['effective_date'] = pd.to_datetime(data['effective_date'], infer_datetime_format= True )
print(data['effective_date'].dtype)
data['due_date'] = pd.to_datetime(data['due_date'], infer_datetime_format= True)
print(data['due_date'].dtype)

#data.groupby(['effective_date'])['due_date'].value_counts(normalize=True)

#adding new column to check the remaining days
data['collection_due'] = data['due_date'] - data['effective_date']
data['collection_due']=data['collection_due'].dt.days

#converting the target variable to numerical values
print(data['loan_status'].unique())
data['loan_status'].replace(to_replace= ['PAIDOFF','COLLECTION'],value = [0,1],inplace= True)
print("target value replaced \n",data.head())

#same thing to the gender

print(data['Gender'].unique())
data['Gender'].replace(to_replace= ['male','female'],value = [0,1],inplace= True)
print("gender value replaced \n",data.head())

#one hot encoding
#doing onehotencoding to education coloumn express the data
encoder = OneHotEncoder()
encoder_df = pd.DataFrame(encoder.fit_transform(data[['education']]).toarray())
data = data.join(encoder_df)
print(data.head(20))
print(data.columns)

#ploting a heatmap after transformation for better eda
Var_Corr = data.corr()
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
#plt.show()



#implementing models with transformed dataset
#defining the x and y variable

X = data.drop([ 'loan_status', 'effective_date', 'due_date', 'education' ],axis=1)
Y = data['loan_status']
print("x = \n", X)
print("y \n", Y)

#scaling the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 2 , shuffle=True )

#KNN

grid_params = {
    'n_neighbors'  : [2,3,4,5,6,7,8,9,10] ,
     'weights' : ['uniform' ,'distance'] ,
}

grid = GridSearchCV ( estimator=KNeighborsClassifier() , param_grid=grid_params,verbose =1 ,scoring='roc_auc')
grid_results = grid.fit(X_train,Y_train)

knn_pred =grid.predict(X_test)
#Score of Knn Model
print("F1 Sore of the KNN Classification model = " , f1_score(Y_test , knn_pred , average='weighted'))
print("Jacard Score of the KNN Classification model =  " ,  jaccard_score(Y_test , knn_pred , average = 'weighted' ))
print("The Best K Value Has been Found as " , grid.best_estimator_)

#logistic regression
logs = LogisticRegression(solver = 'lbfgs' , C = 2 , class_weight = 'balanced' , max_iter = 400 , n_jobs=4)
logs.fit(X_train,Y_train)
#params_logs = {'c': ['100','10','1'] ,'class_weight':['balanced'],'solver ' : [ 'lbfgs', 'liblinear' ]}
logs_pred = logs.predict(X_test)
logs_proba = logs.predict_proba(X_test)

#score of the logistic model
print("F1 Sore of the logistic regreesion = " , f1_score(Y_test , logs_pred, average='weighted'))
print("Jacard Score of the logistic regression =  " ,  jaccard_score(Y_test , logs_pred , average = 'weighted' ))
print("The log loss of the logistic regression model = ", log_loss(Y_test, logs_proba ))

#svm
svm = svm.SVC(probability= True)
SVM_est = svm.fit(X_train,Y_train)
SVM_pred = svm.predict(X_test)
#predictef scores of svm
print("F1 score of the SVM =", f1_score(Y_test, SVM_pred, average= 'weighted'))
print("Jacard score of the SVM model = ",jaccard_score(Y_test, SVM_pred, average= 'weighted'))

#decision tree
Tree_clf = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
Tree_clf.fit(X_train,Y_train)
Tree_pred = Tree_clf.predict(X_test)

#predicting the scores
print("F1 scores of the decison tree model = ", f1_score(Y_test, Tree_pred, average='weighted'))
print("the jaccard score of the decision tree =",jaccard_score(Y_test, Tree_pred, average='weighted'))
#plotting the decison tree score
plot_tree(Tree_clf)
#plt.show()








#######################applying the same transformations for the test data###########################


#testing all the models
data_2 = pd.read_csv('loan_test.csv')
data_2.drop( 'Unnamed: 0' , axis =1 , inplace=True )
data_2.reset_index()
data_2.drop('Unnamed: 0.1' ,axis =1 ,  inplace=True )

#converting the date columns into acceptable format
data_2['effective_date'] = pd.to_datetime(data['effective_date'], infer_datetime_format= True )
print(data_2['effective_date'].dtype)
data_2['due_date'] = pd.to_datetime(data_2['due_date'], infer_datetime_format= True)
print(data_2['due_date'].dtype)

#data.groupby(['effective_date'])['due_date'].value_counts(normalize=True)

#adding new column to check the remaining days
data_2['collection_due'] = data_2['due_date'] - data_2['effective_date']
data_2['collection_due']=data_2['collection_due'].dt.days

#converting the target variable to numerical values
print(data_2['loan_status'].unique())
data_2['loan_status'].replace(to_replace= ['PAIDOFF','COLLECTION'],value = [0,1],inplace= True)
print("target value replaced \n",data_2.head())

#same thing to the gender

print(data_2['Gender'].unique())
data_2['Gender'].replace(to_replace= ['male','female'],value = [0,1],inplace= True)
print("gender value replaced \n",data_2.head())

#one hot encoding
#doing onehotencoding to education coloumn express the data
encoder = OneHotEncoder()
encoder_df = pd.DataFrame(encoder.fit_transform(data[['education']]).toarray())
data_2 = data_2.join(encoder_df)
print(data_2.head(20))
print(data_2.columns)

X = data_2.drop([ 'loan_status' , 'effective_date' , 'due_date' , 'education' ],axis =1 )
Y= data_2['loan_status']
Scaler = StandardScaler()
X = Scaler.fit_transform(X)

n_lst=[]
j_lst=[]
f_lst=[]
l_lst=[]

#logistic regression for test data
testscores = []
logs_pred1 = logs.predict(X)
logs_proba1 = logs.predict_proba(X)

print("the f1 score off the test data = ",f1_score(Y, logs_pred1, average='weighted'))
print("the jaccard score of the test set = ",jaccard_score(Y, logs_pred1, average='weighted'))
print("the logloss for the test set = ", log_loss(Y, logs_proba1))
f_lst.append(f1_score(Y , logs_pred1 , average='weighted', labels=np.unique(logs_pred1)))
l_lst.append(log_loss(Y , logs_proba1 ))
j_lst.append(jaccard_score(Y , logs_pred1 , average = 'weighted'  ))
n_lst.append('LogisticRegression')


#knn for the test data
knn_pred =grid.predict(X)
knn_prob1 = grid.predict_proba(X)

print("F1 Sore of the KNN Classification model = " , f1_score(Y , knn_pred , average='weighted'))
print("Jacard Score of the KNN Classification model =  " ,  jaccard_score(Y , knn_pred , average = 'weighted'))
print("The Best K Value Has been Found as " , grid.best_estimator_)
print("logloss for knn in the test data = ",log_loss(Y, knn_prob1))
f_lst.append(f1_score(Y , knn_pred , average='weighted'  ))
j_lst.append(jaccard_score(Y , knn_pred  , average = 'weighted' ))
l_lst.append(log_loss(Y , knn_prob1 ))
n_lst.append('KNN')





#Decisioon tree for the test data

Tree_pred1 = Tree_clf.predict(X)
Tree_proba1 = Tree_clf.predict_proba(X)

print("the f1 score of decision tree in the test data = ", f1_score(Y, Tree_pred1, average= 'weighted' ))
print("the jaccard score of the decison tree in the test data  = ", jaccard_score(Y, Tree_pred1, average= 'weighted'))
print("the logloss for decision tree in the test datset = ",log_loss(Y, Tree_proba1))
f_lst.append(f1_score(Y , Tree_pred1 , average='weighted' ))
j_lst.append(jaccard_score(Y , Tree_pred1 , average = 'weighted' ))
l_lst.append(log_loss(Y , Tree_proba1 ))
n_lst.append('Descion Tree')


#SVM for the test data
SVM_pred1 = svm.predict(X)
SVM_proba = svm.predict_proba(X)

print("the f1 score for the svm in test data = ",f1_score(Y, SVM_pred1, average= 'weighted'))
print("the jaccard score for the svm in the test data =  ",jaccard_score(Y, SVM_pred1, average= "weighted"))
print(" log loss for SVM in the test datset =  ",log_loss(Y, SVM_proba))
f_lst.append(f1_score(Y , SVM_pred1 , average='weighted' ))
j_lst.append(jaccard_score(Y , SVM_pred1 , average = 'weighted' ))
l_lst.append(log_loss(Y , SVM_proba ))
n_lst.append('SVM')

# Final Report
Report=pd.DataFrame(columns=['Algorithm' ,'Jaccard','F1-Score','LogLoss'])
Report['Algorithm'] = n_lst
Report['Jaccard'] = j_lst
Report['F1-Score'] = f_lst
Report['LogLoss'] = l_lst
print("Report\n"  , Report)





























