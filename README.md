# Naive-Bayes-Classifier
Code for a Naive Bayes Text Classifier 
Used to classify articles as either from The Economist or The Onion (original training data not included).
A Naive Bayes Classifier intended for (binary) text classification. The class distribution was a basic bernoulli distribution, while the class conditional distributions (of each feature) are beta distributions. 
The beta distribution parameters was learnt from a prior distribution of beta(1.001,1.9) using a MAP estimate. 
The parameter for the class distribution was learnt using an MLE estimate. 

Also played around with augmenting feature vectors. 
