
# Audio Classification using Bag-of-Frames approach

## Requirements

### Python 2.7.10 

### Python modules 

    1.  Librosa 0.4.3
    2.  numpy 1.11.3
    3.  sklearn 0.18
    
## Execution     

Divide the audio into smaller clips of 5-20 secs each . One audio clip is converted into one feature which contains the Bag of Frames for that audio clip. 

1. Place the training and the test data in a folder by numbering the categories. Example, if the source folder is "Data"
  
  Data -> 1 -> test -> "test audio files of category 1" <br />
  Data -> 1 -> train -> "train audio files of category 1"<br />
  Data -> 2 -> test -> "test audio files of category 2"<br />
  Data -> 2 -> train -> "test audio files of category 2"<br />
  
2. ```javascript python train.py "window_length" "no_of_clusters" ```

    window_length   : Window length to divide the audio clip to <br />
    no_of_clusters  : No of cluster centroids for k-means clustering   <br />
  
3. Run python test.py "classifier" 

   classifiers : svm,nb,dt,knn,adaboost,rf<br />
   Change the parameters in the test.py file to change the parameters of the classifiers.<br />

## Project Description 
Human speech can be broken down into elementary phonemes and can be modeled using algorithms like Hidden Markov Models (HMM). Stationary patterns like rhythm and melody can be used in classification of music. In contrast, Non speech sounds are random, unstructured, and lack the high level structures observed in speech and music, which makes it difficult to model them using HMM. In this project, the Bag of Frames approach is used to classify audio where a codebook of vectors is generated using K-Means clustering on the training data and  Bag of Frames for each of the audio clip is obtained using the codebook. These Bag of Frames are used as input to the classifiers for classification. 

The steps involved in the Bag of Frames approach for Environmental Sound Classification is described as follows: 


A.	Feature Extraction
    
    1. For the purpose of feature extraction, the audio clip is divided into several segments by choosing a particular window length.  
    2. Then features are extracted for each of the audio segment.
    3. Python libraries Librosa and Scikits are used to extract audio features like MFCC, delta MFCC, Linear Predictive Coding(LPC) coefficients along with other frequency domain features like Mel Spectrogram, Spectral Centroid, Spectral Bandwidth, Spectral Roll Off   and temporal domain features like Root Mean Square Error (RMSE) and Zero Crossing Rate. 
    
    
B.	K-Means Clustering and Codebook generation

    1. Once the features are extracted, the whole training and test data is divided into training and test dataset. 
    2. Feature scaling and normalization of training data is done accross each feature.
    3. The normalized training data is fed into K-Means clustering algorithm with the number of clusters usually much higher than the total number of classes and the cluster centroids are obtained for the normalized training set. 
    4. These cluster centroids form the codebook. 

C.	Bag of Frames
    
    1. In the next step, the feature samples from each of the audio clip are vector quantized with respect to the codebook generated and Bag of Frames is obtained from K-Means output.
    
D.	Classification

    1. The Bag of Frames is first normalized across each audio clip and later normalized across each of the features. The resultant vectors are labelled accordingly and then used to train a supervised classifier like SVM, KNN or Random Forest.
    
    
The test phase includes similar steps where features extracted from the audio clips are normalized and vector quantized using the codebook, followed by obtaining the Bag of Frames for each audio clip.  The normalized Bag of Frames are then given as input to the classifier to obtain the final output.


