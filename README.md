# Scania Truck APS Failure Classification

Classification with neural network exercise with data set from https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks

To determine from a dataset of daily usages of heavy Scania trucks, 170 anonymised input variables were collected, with the objectives of predicting the binary outcome of the Air Pressure System (APS) failure.  “Positive” class failures are classified as related to the APS, while “Negative” denotes otherwise.  

This problem was tackled via classification modelling with neural networks and ensemble methods. 5000 entries were randomly sampled from the training data, and 1000 entries form the testing data.

The objective of the classification is to predict the failure types and minimise the total misclassification, as a false positives leads to unnecessary maintenance (10 cost units), and false negatives results in an engineering defect (500 cost units). For this data set, Multilayer Perceptron with Backward Propagation performed the best in terms of increasing sensitivity and reducing penalty cost. Ensemble models has shown no significant improvement in accuracy and sensitivity over individual models.

Caret package in R is a great wrapper for working with many types of classification and regression models. However, it has its limitations as it does not support all functions fully, e.g.: has no probability function for Extreme Learning Machine.  We can customise our own function for Caret, but it rendered the model inoperable in CaretEnsemble as it does not work with custom models. Radial Basis Function crashed when called from within Caret. Keras packages returned very low accuracies.  
