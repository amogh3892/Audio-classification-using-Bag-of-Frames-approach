
# Audio Classification using Bag-of-Frames approach

## Project Description 
Human speech can be broken down into elementary phonemes and can be modeled using algorithms like Hidden Markov Models (HMM). Stationary patterns like rhythm and melody can be used in classification of music. In contrast, Non speech sounds are random, unstructured, and lack the high level structures observed in speech and music, which makes it difficult to model them using HMM. In this project, the Bag of Frames approach is used to classify audio where a codebook of vectors is generated using K-Means clustering on the training data and  Bag of Frames for each of the audio clip is obtained using the codebook. These Bag of Frames are used as input to the classifiers for classification. 

The steps involved in the Bag of Frames approach for Environmental Sound Classification is described as follows: 


A.	Feature Extraction
    For the purpose of feature extraction, the audio clip is divided into several segments by choosing a particular window length. 
    Then features are extracted for each of the audio segment.
    Python libraries Librosa[3] and Scikits are used to extract audio features like MFCC, delta MFCC, Linear Predictive Coding(LPC)           coefficients along with other frequency domain features like Mel Spectrogram, Spectral Centroid, Spectral Bandwidth, Spectral Roll Off     and temporal domain features like Root Mean Square Error (RMSE) and Zero Crossing Rate. 
