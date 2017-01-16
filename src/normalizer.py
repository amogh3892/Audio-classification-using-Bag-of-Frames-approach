import numpy as np
import os  

print "\n Normalizing training data for kmeans clustering..."

DATA_SOURCE = "Temp\\features_train.csv"
NORMALIZER_FILENAME = "kmeans_normalizer.p"

if os.path.exists(DATA_SOURCE):
	features = np.loadtxt(DATA_SOURCE,delimiter = ",")

	from sklearn.preprocessing import MinMaxScaler
	normalizer = MinMaxScaler()
	normalizer.fit(features)



	import pickle
	with open("Temp\\{}".format(NORMALIZER_FILENAME),"wb") as outfile:
		pickle.dump(normalizer,outfile)
	
	print "kmeans normalizer saved"

else:
	print "No file features_train.csv, Generate the features first."
