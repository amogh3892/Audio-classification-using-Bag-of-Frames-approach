
import librosa

import numpy as np

import os
import shutil
import sys

sourceFolder = "Data"


def get_features(y,sr):
	n_fft = sys.argv[1]

	if n_fft.isdigit() == False:
		"Window length has to be an integer"
		sys.exit()

	n_fft = int(n_fft)	
	hop_length =n_fft/4

	features = None

	#MFCCS
	mfccs =  librosa.feature.mfcc(y=y, sr=sr, n_mfcc = 60 , n_fft = n_fft, hop_length = hop_length)
	features = mfccs

	#Delta mfccs
	delta_mfccs =  librosa.feature.delta(mfccs)
	features = np.concatenate((features,delta_mfccs))


	#rmse
	rmse =  librosa.feature.rmse(y=y , n_fft = n_fft , hop_length = hop_length)
	features = np.concatenate((features,rmse))


	#spectral centroid
	spectral_centroid =  librosa.feature.spectral_centroid(y=y, sr=sr, n_fft = n_fft, hop_length = hop_length)
	features = np.concatenate((features,spectral_centroid))


	#spectral badwidth
	spectral_bandwidth =  librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft = n_fft, hop_length = hop_length)
	features = np.concatenate((features,spectral_bandwidth))


	#spectral contrast
	spectral_contrast =  librosa.feature.spectral_contrast(y=y, sr=sr, n_fft = n_fft, hop_length = hop_length)
	features = np.concatenate((features,spectral_contrast))


	#spectral rolloff
	spectral_rolloff =  librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft = n_fft, hop_length = hop_length)
	features = np.concatenate((features,spectral_rolloff))



	#zero crossing rate
	zero_crossing_rate =  librosa.feature.zero_crossing_rate(y=y, frame_length = n_fft, hop_length = hop_length)
	features = np.concatenate((features,zero_crossing_rate))


	return np.transpose(features)




if os.path.exists(sourceFolder):

	sourceFolderFullPath = os.path.abspath(sourceFolder)

	if os.path.exists("Temp"):
		shutil.rmtree("Temp")
	
	os.makedirs("Temp")



	dirs = os.listdir(sourceFolderFullPath)

	if dirs != []:

		for dir in dirs:
			try:
				label = int(dir)
			except:
				"The labels given to the categories should be numerical."
				sys.exit()

			audioTrainPath = "{}/{}/train".format(sourceFolderFullPath,dir)
			audioTestPath = "{}/{}/test".format(sourceFolderFullPath,dir)

			print "Retrieving audio features of category {}".format(label)
			
			for subdir in [audioTrainPath,audioTestPath]:

				csvSubdir =  subdir.replace("Data","Temp\\csv")
				os.makedirs(csvSubdir)
			
				for file in os.listdir(subdir):
					if "ogg" in file or "wav" in file:
						soundFileFullPath = os.path.join(subdir,file)
						featuresFile = file.replace(".ogg","")
						y,sr = librosa.load(soundFileFullPath)
						features = get_features(y,sr)


						featuresFileFullPath = os.path.join(csvSubdir,"{}.csv".format(featuresFile))
						np.savetxt(featuresFileFullPath,features,fmt='%1.3f',delimiter=",")


	else:
		"The Data directory is empty"

else:
	print "The Data path doesn't exist"









