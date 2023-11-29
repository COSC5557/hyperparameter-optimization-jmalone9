import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt 
from skopt import BayesSearchCV
import time

wine = pd.read_csv('winequality-white.csv', sep=';', header = 'infer')
y = wine.iloc[:,11]
x = wine.drop(wine.columns[11], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=99)

accuracies = []
algorithms = []
times = []
bestAcc = 0.0
bestAlgorithm = "None"

def outputScore(name):
    acc = accuracy_score(y_test, y_pred) * 100
    #print("Accuracy Score for ",name,": {0:.5}%".format(acc))
    print("Accuracy Score for ",name,acc,"%")
    
#make function for evaluation here

############################################################################################################################################################################
#random forest classifier
############################################################################################################################################################################
t0 = time.time()
rf = RandomForestClassifier(random_state=99)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
outputScore("Random Forest Default")
acc1 = accuracy_score(y_test, y_pred) * 100
t1 = time.time()
total = t1-t0
times.append(total)
algorithms.append("Random Forest Default")
accuracies.append(acc1)


t0 = time.time()
params = {
    #'bootstrap': [True], #always says best is true though its not default
    'max_depth': [25, 50, None], #default is none
    'n_estimators': [75, 100, 125], #higher seemingly always better, default is 100
    'min_samples_split': [2, 4, 6], #default values (the first one) seems to always be chosen
    'min_samples_leaf': [1, 3, 5] #default values (the first one) seems to always be chosen
    #'warm_start': [True] #always says best is true though its not default
}

rf = GridSearchCV(estimator = rf, param_grid = params, scoring='accuracy', cv = 5, n_jobs = -1)
rf.fit(x_train, y_train)
bestModel = rf.best_estimator_
bestModel.fit(x_train, y_train)
y_pred = bestModel.predict(x_test)
acc2 = accuracy_score(y_test, y_pred) * 100
accChange = acc2 - acc1
outputScore("Random Forest GridSearch")
#print("Best Parameters: ", rf.best_params_, "\n")
t1 = time.time()
total = t1-t0
times.append(total)
algorithms.append("Random Forest GridSearch")
accuracies.append(acc2)

t0 = time.time()
opt = BayesSearchCV(RandomForestClassifier(random_state = 99), params, n_jobs = 10, scoring = 'accuracy', verbose = 0, random_state = 99, n_iter = 15, cv = 5, return_train_score = True)
opt.fit(x_train, y_train)
#print("Best Paremeters found by baysesian: ", opt.best_params_)
y_pred = opt.predict(x_test)
testAcc = accuracy_score(y_test, y_pred) * 100
outputScore("Random Forest Bayesian")
t1 = time.time()
total = t1-t0
times.append(total)
algorithms.append("Random Forest Bayesian")
accuracies.append(testAcc)
############################################################################################################################################################################
#nearest neighbor classifier
############################################################################################################################################################################
t0 = time.time()
knn = KNeighborsClassifier() 
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
outputScore("Nearest Neighbors Default")
acc1 = accuracy_score(y_test, y_pred) * 100
t1 = time.time()
total = t1-t0
times.append(total)
algorithms.append("Nearest Neighbors Default")
accuracies.append(acc1)

t0 = time.time()
#seems to pick default 5 neighbors, but will pick 1 if its available to overfit itself, lowest leaf size it can, ball_tree and p=1 when it can
params = {
    'n_neighbors': [3, 4, 5, 6, 10], #default 5 used small numbers for this because it never picks the big ones
    'leaf_size': [10, 30, 50, 100, 200], #in this problem leaf size doesnt seem to affect accuracy
    'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
    'p': [1, 2] 
}

knn = GridSearchCV(estimator = knn, param_grid = params, scoring='accuracy', cv = 5, n_jobs = -1)
knn.fit(x_train, y_train)
bestModel = knn.best_estimator_
bestModel.fit(x_train, y_train)
y_pred = bestModel.predict(x_test)
acc2 = accuracy_score(y_test, y_pred) * 100
accChange = acc2 - acc1
outputScore("Nearest Neighbors GridSearch")
t1 = time.time()
total = t1-t0
times.append(total)
algorithms.append("Nearest Neighbors GridSearch")
accuracies.append(acc2)
#print("Best Parameters: ", knn.best_params_, "\n")

t0 = time.time()
opt = BayesSearchCV(KNeighborsClassifier() , params, n_jobs = 10, scoring = 'accuracy', verbose = 0, random_state = 99, n_iter = 15, cv = 5, return_train_score = True)
opt.fit(x_train, y_train)
#print("Best Paremeters found by baysesian: ", opt.best_params_)
y_pred = opt.predict(x_test)
testAcc = accuracy_score(y_test, y_pred) * 100
print("Accuracy Score for ","Nearest Neighbors Bayesian",testAcc,"%")
t1 = time.time()
total = t1-t0
times.append(total)
algorithms.append("Nearest Neighbors Bayesian")
accuracies.append(testAcc)





###################################################################################################
'''algorithms.append("Nearest Neighbors GridSearch")
accuracies.append(acc1)
'''
plt.figure(figsize = (10,4))
plt.barh(algorithms, accuracies)
plt.xlabel("Accuracy")
plt.ylabel("Algorithms")
plt.title("Algorithms with Accuracy")
plt.show()

#code for this from https://stackoverflow.com/questions/48053979/print-2-lists-side-by-side user SCB
sortedAlgList = "\n".join("{}: {:0.5f}% accuracy, {:0.5f} seconds".format(y, x, z) for x, y, z in sorted(zip(accuracies, algorithms, times), key = lambda x: (x[0], -x[2]), reverse = True))
print("List of algorithms sorted best to worst:\n")
print(sortedAlgList)
print("Best Algorithm by accuracy and time is:", sortedAlgList.partition(":")[0])
