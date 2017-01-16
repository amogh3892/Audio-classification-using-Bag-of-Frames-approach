
import os
import numpy as np 

print "\n Generating Training and Test Data"

sourceFolder = "Temp//csv"
sourceFolderFullPath = os.path.abspath(sourceFolder)


dirs = os.listdir(sourceFolderFullPath)


#getting training data
features = None
labels = None
for dir in dirs:
	label = int(dir)

	featuresTrainPath = "{}/{}/train".format(sourceFolderFullPath,dir)

	for file in os.listdir(featuresTrainPath):
		if "csv" in file :
			featuresFileFullPath = os.path.join(featuresTrainPath,file)

			temp_features = np.loadtxt(featuresFileFullPath,delimiter = ",")

			#remove rows with nan values 
			temp_features = temp_features[~np.isnan(temp_features).any(axis=1)]

			temp_labels = [label]*temp_features.shape[0]
			temp_labels = np.expand_dims(temp_labels,axis = 1)


			if features is None:
				features = temp_features
				labels = temp_labels
			else:
				features = np.concatenate((features,temp_features))
				labels = np.concatenate((labels,temp_labels))


print "Total Training Frames : {}".format(features.shape[0])


np.savetxt("Temp\\features_train.csv",features,fmt='%1.3f',delimiter=",")
np.savetxt("Temp\\labels_train.csv",labels,delimiter = ",")



#getting test data
features = None
labels = None
for dir in dirs:
	label = int(dir)

	featuresTestPath = "{}/{}/test".format(sourceFolderFullPath,dir)

	for file in os.listdir(featuresTestPath):
		if "csv" in file :
			featuresFileFullPath = os.path.join(featuresTestPath,file)

			temp_features = np.loadtxt(featuresFileFullPath,delimiter = ",")

			#remove rows with nan values 
			temp_features = temp_features[~np.isnan(temp_features).any(axis=1)]

			temp_labels = [label]*temp_features.shape[0]
			temp_labels = np.expand_dims(temp_labels,axis = 1)


			if features is None:
				features = temp_features
				labels = temp_labels
			else:
				features = np.concatenate((features,temp_features))
				labels = np.concatenate((labels,temp_labels))




np.savetxt("Temp\\features_test.csv",features,fmt='%1.3f',delimiter=",")
np.savetxt("Temp\\labels_test.csv",labels,delimiter = ",")

print "Total Test Frames : {}".format(features.shape[0])

print "Features Generated"


			