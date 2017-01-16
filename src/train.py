
import sys 
import subprocess


if len(sys.argv) != 3:
	print "\nArguements not valid\npython train.py window_length clusters "
	sys.exit()

window_length = sys.argv[1]

if window_length.isdigit() == False:
	"Window length has to be an integer"
	sys.exit()


#Retrieving features from the audio files.
command_run = subprocess.call("python retrieve_audio_features.py {}".format(window_length))
if command_run != 0:
	sys.exit()

#Generating the training and test data set
command_run = subprocess.call("python generate_training_test_data.py")
if command_run != 0:
	sys.exit()


#normalizing the training dataset
command_run = subprocess.call("python normalizer.py")
if command_run != 0:
	sys.exit()

# no of clusters for kmenas clustering 
clusters = sys.argv[2]

if clusters.isdigit() == False:
	"No of clusters has to be an integer"
	sys.exit()


# running kmeans clustering algo on the normalized data
command_run = subprocess.call("python kmeans_train.py {}".format(clusters))
if command_run != 0:
	sys.exit()

# derived features after taking the bag of frames for each of the audio file.
command_run = subprocess.call("python generate_derived_training_test_data.py {}".format(clusters))
if command_run != 0:
	sys.exit()


