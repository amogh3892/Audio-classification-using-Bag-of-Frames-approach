import os
import numpy as np 
import sys 

print "Running kmeans clustering .. "

no_clusters = sys.argv[1]

if no_clusters.isdigit() == False:
	"No of clusters has to be an integer"
	sys.exit()


no_clusters = int(no_clusters)
features_train = np.loadtxt("Temp\\features_train.csv",delimiter = ",")


import pickle
with open("Temp\\kmeans_normalizer.p","rb") as infile:
	normalizer = pickle.load(infile)


features_train = normalizer.transform(features_train)


from sklearn.cluster import KMeans


kmeans = KMeans(n_clusters = no_clusters).fit(features_train)



with open("Temp\\kmeans_{}.p".format(no_clusters),"wb") as outfile:
	pickle.dump(kmeans,outfile)

outfile.close()


print "\nClusters generated and saved"


