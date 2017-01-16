import os
import numpy as np 

import pickle

import sys


print "\nGenerating bag of frames..."

clusters = sys.argv[1]

if clusters.isdigit() == False:
	"No of clusters has to be an integer"
	sys.exit()


#loading the normalizer for kmeans 
with open("Temp//kmeans_normalizer.p","rb") as infile:
	kmeans_normalizer = pickle.load(infile)
infile.close()

#consider kmeans with 50 cluster centers 
with open("Temp/kmeans_{}.p".format(clusters),"rb") as infile:
	kmeans = pickle.load(infile)
infile.close()



no_of_centers = kmeans.cluster_centers_.shape[0]


sourceFolder = "Temp//csv"
sourceFolderFullPath = os.path.abspath(sourceFolder)

dirs = os.listdir(sourceFolderFullPath)

#getting derived training data

derived_features = []
derived_labels = []

for dir in dirs:


	label = int(dir)

	featuresTrainPath = "{}/{}/train".format(sourceFolderFullPath,dir)

	i = 0
	j = 0
	for file in os.listdir(featuresTrainPath):
		
		if "csv" in file :

			featuresFileFullPath = os.path.join(featuresTrainPath,file)

			#getting training features of a file
			temp_features = np.loadtxt(featuresFileFullPath,delimiter = ",")

			# #pca 
			# temp_features = pca.transform(temp_features)

			#remove rows with nan values 
			temp_features = temp_features[~np.isnan(temp_features).any(axis=1)]

			#normalizer the features for kmeans 
			temp_features = kmeans_normalizer.transform(temp_features)


			#get kmeans output 
			kmeans_output = kmeans.predict(temp_features)


			#getting bag of words as sample
			unique, counts = np.unique(kmeans_output, return_counts=True)
			sample = [0]*no_of_centers
			for j in range(no_of_centers):
				if j in unique:
					sample[j] = counts[np.where(unique == j)][0]

				
	        derived_features.append(sample)
	        derived_labels.append(int(label))



derived_features = np.array(derived_features).astype(np.float64)
derived_labels = np.expand_dims(derived_labels,axis=1).astype(np.int32)
np.savetxt("Temp\\derived_features_train.csv",derived_features,fmt='%1.3f',delimiter=",")
np.savetxt("Temp\\derived_labels_train.csv",derived_labels,fmt='%1.3f',delimiter = ",")


#getting derived test features.

derived_features = []
derived_labels = []

for dir in dirs:
	label = int(dir)

	featuresTestPath = "{}/{}/test".format(sourceFolderFullPath,dir)

	i = 0
	j = 0
	for file in os.listdir(featuresTestPath):
		
		if "csv" in file :

			featuresFileFullPath = os.path.join(featuresTestPath,file)

			#getting training features of a file
			temp_features = np.loadtxt(featuresFileFullPath,delimiter = ",")


			#remove rows with nan values 
			temp_features = temp_features[~np.isnan(temp_features).any(axis=1)]

			#normalizer the features for kmeans 
			temp_features = kmeans_normalizer.transform(temp_features)

			#get kmeans output 
			kmeans_output = kmeans.predict(temp_features)


			#getting bag of words as sample
			unique, counts = np.unique(kmeans_output, return_counts=True)
			sample = [0]*no_of_centers
			for j in range(no_of_centers):
				if j in unique:
					sample[j] = counts[np.where(unique == j)][0]

				
	        derived_features.append(sample)
	        derived_labels.append(int(label))


derived_features = np.array(derived_features).astype(np.float64)
derived_labels = np.expand_dims(derived_labels,axis=1).astype(np.int32)
np.savetxt("Temp\\derived_features_test.csv",derived_features,fmt='%1.3f',delimiter=",")
np.savetxt("Temp\\derived_labels_test.csv",derived_labels,fmt='%1.3f',delimiter = ",")


print "Bag of frames generated and saved\n Training completed. Please run test.py by selecting a classifier to test the data"