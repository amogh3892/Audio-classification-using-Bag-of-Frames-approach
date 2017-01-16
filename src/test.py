#test
from time import time
def svm_predict(training_samples, training_labels, test_samples, test_lables, kernel = "rbf" , C = 1.0):
	from sklearn.svm import SVC

	clf = SVC(kernel = kernel, C =C)

	t0 = time()
	clf.fit(training_samples,training_labels)
	training_time = round(time()-t0, 3)

	t0 = time()
	pred = clf.predict(test_samples)
	test_time = round(time()-t0, 3)

	from sklearn.metrics import accuracy_score

	acc = accuracy_score(test_lables,pred)


	no_features = np.array(training_samples).shape[1]
	training_samples = np.array(training_samples).shape[0]
	test_samples = np.array(test_samples).shape[0]

	with open("Temp\\results.txt","w") as outfile:
		outfile.write("Alogirthm : {}\n".format("SVM"))
		outfile.write("kernel = {}\n".format(kernel))
		outfile.write("C = {}\n".format(C))
		outfile.write("No of features : {}\n".format(no_features))
		outfile.write("No of training samples : {}\n".format(training_samples))
		outfile.write("No of test samples : {}\n".format(test_samples))
		outfile.write("Training time : {}\n".format(training_time))
		outfile.write("Test time : {}\n".format(test_time))
		outfile.write("Accuracy : {}\n".format(acc))

	with open("Temp\\result_labels.csv","wb") as outfile:
		np.savetxt(outfile,pred)

	
def naive_bayes_predict(training_samples, training_labels, test_samples, test_lables):
	from sklearn.naive_bayes import GaussianNB

	clf = GaussianNB()

	t0 = time()
	clf.fit(training_samples,training_labels)
	training_time = round(time()-t0, 3)

	t0 = time()
	pred = clf.predict(test_samples)
	test_time = round(time()-t0, 3)

	from sklearn.metrics import accuracy_score

	acc = accuracy_score(pred,test_lables)

	no_features = np.array(training_samples).shape[1]
	training_samples = np.array(training_samples).shape[0]
	test_samples = np.array(test_samples).shape[0]

	with open("Temp\\results.txt","w") as outfile:
		outfile.write("Alogirthm : {}\n".format("Naive Bayes"))
		outfile.write("No of features : {}\n".format(no_features))
		outfile.write("No of training samples : {}\n".format(training_samples))
		outfile.write("No of test samples : {}\n".format(test_samples))
		outfile.write("Training time : {}\n".format(training_time))
		outfile.write("Test time : {}\n".format(test_time))
		outfile.write("Accuracy : {}\n".format(acc))

	with open("Temp\\result_labels.csv","wb") as outfile:
		np.savetxt(outfile,pred)



def decision_tree_predict(training_samples, training_labels, test_samples, test_lables, criterion = "gini", min_samples_split = 2):
	from sklearn.tree import DecisionTreeClassifier

	clf = DecisionTreeClassifier(criterion = criterion,min_samples_split = min_samples_split)

	t0 = time()
	clf.fit(training_samples,training_labels)
	training_time = round(time()-t0, 3)

	t0 = time()
	pred = clf.predict(test_samples)
	test_time = round(time()-t0, 3)

	from sklearn.metrics import accuracy_score

	acc = accuracy_score(pred,test_lables)

	no_features = np.array(training_samples).shape[1]
	training_samples = np.array(training_samples).shape[0]
	test_samples = np.array(test_samples).shape[0]

	with open("Temp\\results.txt","w") as outfile:
		outfile.write("Alogirthm : {}\n".format("Decision Tree"))
		outfile.write("criterion = {}\n".format(criterion))
		outfile.write("min_samples_split = {}\n".format(min_samples_split))
		outfile.write("No of features : {}\n".format(no_features))
		outfile.write("No of training samples : {}\n".format(training_samples))
		outfile.write("No of test samples : {}\n".format(test_samples))
		outfile.write("Training time : {}\n".format(training_time))
		outfile.write("Test time : {}\n".format(test_time))
		outfile.write("Accuracy : {}\n".format(acc))

	with open("Temp\\result_labels.csv","wb") as outfile:
		np.savetxt(outfile,pred)


def knn_predict(training_samples, training_labels, test_samples, test_lables,k_neighbours = 5,weights = "uniform",algorithm = "auto"):
	from sklearn.neighbors import KNeighborsClassifier

	clf = KNeighborsClassifier(n_neighbors = k_neighbours, weights =weights, algorithm = algorithm)

	t0 = time()
	clf.fit(training_samples,training_labels)
	training_time = round(time()-t0, 3)

	t0 = time()
	pred = clf.predict(test_samples)
	test_time = round(time()-t0, 3)

	from sklearn.metrics import accuracy_score

	acc = accuracy_score(pred,test_lables)

	no_features = np.array(training_samples).shape[1]
	training_samples = np.array(training_samples).shape[0]
	test_samples = np.array(test_samples).shape[0]

	with open("Temp\\results.txt","w") as outfile:
		outfile.write("Alogirthm : {}\n".format("KNN"))
		outfile.write("K  = {}\n".format(k_neighbours))
		outfile.write("weight = {}\n".format(weights))
		outfile.write("algorithm = {}\n".format(algorithm))
		outfile.write("No of features : {}\n".format(no_features))
		outfile.write("No of training samples : {}\n".format(training_samples))
		outfile.write("No of test samples : {}\n".format(test_samples))
		outfile.write("Training time : {}\n".format(training_time))
		outfile.write("Test time : {}\n".format(test_time))
		outfile.write("Accuracy : {}\n".format(acc))

	with open("Temp\\result_labels.csv","wb") as outfile:
		np.savetxt(outfile,pred)


def adaboost_predict(training_samples, training_labels, test_samples, test_lables,n_estimators=50, learning_rate=1.0):
	from sklearn.ensemble import AdaBoostClassifier

	clf = AdaBoostClassifier(n_estimators = n_estimators, learning_rate =learning_rate)

	t0 = time()
	clf.fit(training_samples,training_labels)
	training_time = round(time()-t0, 3)

	t0 = time()
	pred = clf.predict(test_samples)
	test_time = round(time()-t0, 3)

	from sklearn.metrics import accuracy_score

	acc = accuracy_score(pred,test_lables)

	no_features = np.array(training_samples).shape[1]
	training_samples = np.array(training_samples).shape[0]
	test_samples = np.array(test_samples).shape[0]

	with open("Temp\\results.txt","w") as outfile:
		outfile.write("Alogirthm : {}\n".format("Adaboost"))
		outfile.write("Estimators  = {}\n".format(n_estimators))
		outfile.write("Learning rate = {}\n".format(learning_rate))
		outfile.write("No of features : {}\n".format(no_features))
		outfile.write("No of training samples : {}\n".format(training_samples))
		outfile.write("No of test samples : {}\n".format(test_samples))
		outfile.write("Training time : {}\n".format(training_time))
		outfile.write("Test time : {}\n".format(test_time))
		outfile.write("Accuracy : {}\n".format(acc))

	with open("Temp\\result_labels.csv","wb") as outfile:
		np.savetxt(outfile,pred)


def randomforest_predict(training_samples,training_labels,test_samples,test_lables,n_estimators =100,criterion = 'gini',min_samples_split=2):
	from sklearn.ensemble import RandomForestClassifier

	clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split)

	t0 = time()
	clf.fit(training_samples,training_labels)
	training_time = round(time()-t0, 3)

	t0 = time()
	pred = clf.predict(test_samples)
	test_time = round(time()-t0, 3)

	from sklearn.metrics import accuracy_score

	acc = accuracy_score(test_lables,pred)

	no_features = np.array(training_samples).shape[1]
	training_samples = np.array(training_samples).shape[0]
	test_samples = np.array(test_samples).shape[0]

	with open("Temp\\results.txt","w") as outfile:
		outfile.write("Alogirthm : {}\n".format("Random Forest"))
		outfile.write("Estimators  = {}\n".format(n_estimators))
		outfile.write("Criterion = {}\n".format(criterion))
		outfile.write("min_samples_split = {}\n".format(min_samples_split))
		outfile.write("No of features : {}\n".format(no_features))
		outfile.write("No of training samples : {}\n".format(training_samples))
		outfile.write("No of test samples : {}\n".format(test_samples))
		outfile.write("Training time : {}\n".format(training_time))
		outfile.write("Test time : {}\n".format(test_time))
		outfile.write("Accuracy : {}\n".format(acc))


	with open("Temp\\result_labels.csv","wb") as outfile:
		np.savetxt(outfile,pred)




import numpy as np 
features_train = np.loadtxt("Temp\\derived_features_train.csv",delimiter = ",")
labels_train = np.loadtxt("Temp\\derived_labels_train.csv",delimiter = ",")
features_test = np.loadtxt("Temp\\derived_features_test.csv",delimiter=",")
labels_test = np.loadtxt("Temp\\derived_labels_test.csv",delimiter=",")



from sklearn.preprocessing import MinMaxScaler


#first normalize horizontally 
features_train = np.transpose(features_train)
features_test = np.transpose(features_test)

model_normalizer_horizontal = MinMaxScaler()
model_normalizer_horizontal.fit(features_train)

features_train = model_normalizer_horizontal.transform(features_train)


model_normalizer_horizontal = MinMaxScaler()
model_normalizer_horizontal.fit(features_test)
features_test = model_normalizer_horizontal.transform(features_test)

features_train = np.transpose(features_train)
features_test = np.transpose(features_test)


#normalize vertically

model_normalizer_vertical = MinMaxScaler()
model_normalizer_vertical.fit(features_train)

features_train = model_normalizer_vertical.transform(features_train)
features_test = model_normalizer_vertical.transform(features_test)


import sys
classifier = sys.argv[1]
classifier = classifier.lower()

if classifier == "svm":
	svm_predict(features_train,labels_train,features_test,labels_test,kernel = "linear", C = 1.0)
elif classifier == "nb":
	naive_bayes_predict(features_train,labels_train,features_test,labels_test)
elif classifier == "dt":
	decision_tree_predict(features_train,labels_train,features_test,labels_test,criterion="gini",min_samples_split=20)
elif classifier == "knn":
	knn_predict(features_train,labels_train,features_test,labels_test,k_neighbours =11,weights = "distance",algorithm = "kd_tree")
elif classifier == "adaboost":
	adaboost_predict(features_train,labels_train,features_test,labels_test,n_estimators=100, learning_rate=1)
elif classifier == "rf":
	randomforest_predict(features_train,labels_train,features_test,labels_test,n_estimators=1000,criterion='gini',min_samples_split=2)






