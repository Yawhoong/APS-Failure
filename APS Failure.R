

#### CLASSIFICATION NEURAL NETWORK WITH R ####
# USING NEURAL NETWORK TO PREDICT SCANIA TRUCK APS FAILURE  #


#### 1. Pre-processing ####

# Most packages can be installed normally, except caret and keras

# caret:
# install.packages("caret", dependencies = c("Depends", "Suggests"))

# keras:
# devtools::install_github("rstudio/keras")
# library(keras)
# install_keras()
# May need to install Python (ie install Jupyter Notebook) in the Tensorflow environment of Anaconda for Keras to work in RStudio


# to start parallel processing
library(doParallel)
registerDoParallel()
getDoParWorkers()


# check working directory and make sure training and testing csv are in the folder
getwd()

# read in the training data
dfTrain <- read.csv(file = "aps_failure_training_set.csv", header=TRUE, stringsAsFactors = FALSE, na.strings = "na" )

# read in the testing data
dfTest <- read.csv(file = "aps_failure_test_set.csv", header=TRUE, stringsAsFactors = FALSE, na.strings = "na" )

# inspect the 2 data sets
str(dfTrain)
str(dfTest)

# inspect the proportion of the outcome in the data set 
prop.table(table(dfTrain$class))
prop.table(table(dfTest$class))


# inspect the data frame for NA
sum(is.na(dfTrain))
sum(is.na(dfTest))

# check how many rows are without NA
sum(complete.cases(dfTrain))
sum(complete.cases(dfTest))

# randomly sample the data sets to create smaller sets to reduce computational time; set replace to false so no repeated data rows 
# set seed to ensure reproducibility
set.seed(101)
sampleTrain <- sample.int(n = nrow(dfTrain), size = 5000, replace = F)

set.seed(101)
sampleTest <- sample.int(n = nrow(dfTest), size = 1000, replace = F)

dfTrain5000 <- dfTrain[sampleTrain, ]
dfTest1000 <- dfTest[sampleTest, ]


# inspect the structure of the 2 data sets 
str(dfTrain5000)
str(dfTest1000)

# inspect the data frame to ensure proportion of the outcome is equally represented as the original data set
prop.table(table(dfTrain5000$class))
prop.table(table(dfTest1000$class))


# as the results are binary, set the output column as categorical (factor)
dfTrain5000$class <- as.factor(dfTrain5000$class)
dfTest1000$class <- as.factor(dfTest1000$class)

# no other columns appear to be categorical data, no further as.factor is performed and no dummy variables will be created

# since columns cd_000 and ch_000 have constant numbers, they are removed
dfTrain5000$cd_000 <- NULL
dfTrain5000$ch_000 <- NULL
dfTest1000$cd_000 <- NULL
dfTest1000$ch_000 <- NULL


# Use k-nearest neighbour imputation model from Caret to estimate NA 
library(caret)
preProcess_NA_model <- preProcess(dfTrain5000, method='knnImpute')
preProcess_NA_model

library(RANN)
dfTrainFilled <- predict(preProcess_NA_model, newdata = dfTrain5000)
anyNA(dfTrainFilled) # return False if no more NA 

dfTestFilled <- predict(preProcess_NA_model, dfTest1000)  
anyNA(dfTestFilled) # return False if no more NA 


# normalise the data range between 0 and 1
preProcess_range_model <- preProcess(dfTrainFilled, method='range')

dfTrainNorm <- predict(preProcess_range_model, newdata = dfTrainFilled)
dfTestNorm <- predict(preProcess_range_model, newdata = dfTestFilled)

head(dfTrainNorm[ , 1:20])
head(dfTestNorm[ , 1:20])


# plot density plots between each variable and output to inspect their relationships
featurePlot(x = dfTrainNorm[ , 2:169], 
            y = dfTrainNorm$class, 
            plot = "density",
            strip = strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))


# feature selection using recursive feature elimination (RFE) with random forest function and k-fold cross validation
set.seed(101)
options(warn=-1)

subsets <- c(5, 10, 15)

ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

lmProfile <- rfe(x=dfTrainNorm[ , 2:169], y=dfTrainNorm$class,
                 sizes = subsets,
                 rfeControl = ctrl)

lmProfile



#### 2. Individual Neural Networks ####

# Define the training control for all models
fitControl <- trainControl(
  method = 'cv',                   # k-fold cross validation
  number = 5,                      # number of folds
  savePredictions = 'final',       # saves predictions for optimal tuning parameter
  classProbs = T,                  # should class probabilities be returned
  summaryFunction=twoClassSummary  # results summary function
)


#### 2.1  NNet ####
# Single Hidden Layer Feed Forward Neural Networks #

# caret::train ensures 'train' function from caret, and not RSNNS, is used
library(nnet)

set.seed(101)
model_nnet <- caret::train(class ~ ., data=dfTrainNorm, method='nnet', metric='ROC', trControl = fitControl, tuneLength = 5, maxit=1000)
model_nnet

# check the model accuracy wrt number of variables used in prediction
plot(model_nnet, main="Model Accuracies with NNet")

# check the importance of variables
varimp_nnet <- varImp(model_nnet)
plot(varimp_nnet, main="Variable Importance with NNet")

# Predict on testData
predicted_nnet <- predict(model_nnet, dfTestNorm)

# Compute the confusion matrix
caret::confusionMatrix(reference = dfTestNorm$class, data = predicted_nnet, mode='everything', positive='pos')

library(NeuralNetTools)
par(mar = numeric(4), family = 'serif')
plotnet(model_nnet, pos_col = "green", neg_col = "red")


#### 2.2  MLP ####
# Multi-Layer Perceptron with Back Propagation #

# MLP with 3 layers and fixed node numbers

library(RSNNS)

mlp_grid = expand.grid(layer1 = c(60, 70, 80),
                       layer2 = c(30, 40, 50),
                       layer3 = c(5, 10, 20))
set.seed(101)
model_mlpML <- caret::train(class ~ ., data = dfTrainNorm, method='mlpML', metric='ROC', trControl = fitControl, tuneGrid = mlp_grid)
model_mlpML

# check the model accuracy wrt number of variables used in prediction
plot(model_mlpML, main="Model Accuracies with MLP")

# check the importance of variables
varimp_mlpML <- varImp(model_mlpML)
plot(varimp_mlpML, main="Variable Importance with MLP")

varimp_mlpMLnoGrid <- varImp(model_mlpMLnoGrid)
plot(varimp_mlpMLnoGrid, main="Variable Importance with MLP")

# Predict on testData
predicted_mlpML <- predict(model_mlpML, dfTestNorm)

# Compute the confusion matrix
caret::confusionMatrix(reference = dfTestNorm$class, data = predicted_mlpML, mode='everything', positive='pos')

library(NeuralNetTools)
par(mar = numeric(4), family = 'serif')
plotnet(model_mlpML2, pos_col = "green", neg_col = "red")


# 2.3  ELM ####
# Extreme Learning Machine with Single Hidden Layer Feed Forward Neural Networks #

library(elmNN)

# Manually adjust the probability model in elmNN as it's absent in the existing Caret package
elm_fun <- getModelInfo("elm")[[1]]
elm_fun$prob <- function (modelFit, newdata, submodels = NULL)  {
  out <- exp(predict(modelFit, newdata))
  t(apply(out, 1, function(x) x/sum(x)))
}

set.seed(101)

model_elm <- caret::train(class ~ ., data=dfTrainNorm, method=elm_fun, metric="ROC", trControl = fitControl, tuneGrid = expand.grid(nhid = c(5, 10, 20, 30), actfun = c("sig", "sin", "radbas") ) )

model_elm

# check the model accuracy wrt number of variables used in prediction
plot(model_elm, main="Model Accuracies with ELM")

# check the importance of variables
varimp_elm <- varImp(model_elm)
plot(varimp_elm, main="Variable Importance with ELM")

# Predict on testData
predicted_elm <- predict(model_elm, dfTestNorm)

# Compute the confusion matrix
caret::confusionMatrix(reference = dfTestNorm$class, data = predicted_elm, mode='everything', positive='pos')


#### 3. Model Comparison ####

models_compare <- resamples(list(nnet=model_nnet, mlpML=model_mlpML, elm=model_elm))

# Summary of the models performances
summary(models_compare)
dotplot(models_compare)
xyplot(models_compare)
modelCor(models_compare)
splom(models_compare)

# Draw box plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_compare, scales=scales)


#### 4. Ensemble Neural Network ####

#### 4.1  Averaging ####

# Start a new data frame for all the predicted results
dfTestEnsemble <- as.data.frame(dfTestNorm$class)
colnames(dfTestEnsemble) <- 'class'

# extract the prediction probabilities
dfTestEnsemble$nnet_pred <- predicted_nnet
dfTestEnsemble$nnet <-predict(model_nnet, dfTestNorm, type='prob')

dfTestEnsemble$mlpML_pred <- predicted_mlpML
dfTestEnsemble$mlpML <- predict(model_mlpML, dfTestNorm, type='prob')

dfTestEnsemble$elm_pred <- predicted_elm
dfTestEnsemble$elm <- predict(model_elm, dfTestNorm, type='prob')


#Taking average of predictions
dfTestEnsemble$avg_pred <- (dfTestEnsemble$nnet$pos + dfTestEnsemble$mlpML$pos + dfTestEnsemble$elm$pos)/3

#Splitting into binary classes at 0.5
dfTestEnsemble$avg_pred <- as.factor(ifelse(dfTestEnsemble$avg_pred > 0.5,'pos','neg'))

caret::confusionMatrix(reference = dfTestEnsemble$class, data = dfTestEnsemble$avg_pred, mode='everything', positive='pos')


#### 4.2  Majority Voting ####

dfTestEnsemble$maj_pred <- as.factor(
  ifelse (dfTestEnsemble$nnet_pred =='pos' & dfTestEnsemble$mlpML_pred =='pos', 'pos',
          ifelse (dfTestEnsemble$nnet_pred =='pos' & dfTestEnsemble$elm_pred =='pos', 'pos',
                 ifelse (dfTestEnsemble$mlpML_pred =='pos' & dfTestEnsemble$elm_pred == 'pos','pos','neg'))))

caret::confusionMatrix(reference = dfTestEnsemble$class, data = dfTestEnsemble$maj_pred, mode='everything', positive='pos')



#### 4.3  Stacking ####

#Predicting the out of fold prediction probabilities for training data
dfTrainEnsemble <- dfTrainNorm
dfTrainEnsemble$OOF_pred_nnet <- model_nnet$pred$pos[order(model_nnet$pred$rowIndex)]
dfTrainEnsemble$OOF_pred_mlpML <- model_mlpML$pred$pos[order(model_mlpML$pred$rowIndex)]
dfTrainEnsemble$OOF_pred_elm <- model_elm$pred$pos[order(model_elm$pred$rowIndex)]

#Predicting probabilities for the test data
dfTestEnsemble$OOF_pred_nnet <- predict(model_nnet, dfTestNorm, type = 'prob')$pos
dfTestEnsemble$OOF_pred_mlpML <- predict(model_mlpML, dfTestNorm, type = 'prob')$pos
dfTestEnsemble$OOF_pred_elm <- predict(model_elm, dfTestNorm, type = 'prob')$pos

predictors_top <- c('OOF_pred_nnet','OOF_pred_mlpML','OOF_pred_elm') 


#### 4.3.1  GBM ####
# Stack using Stochastic Gradient Boosting
model_gbm <- caret::train(dfTrainEnsemble[ , predictors_top], dfTrainEnsemble$class, method='gbm', metric='ROC', trControl=fitControl, tuneLength=3)
model_gbm

dfTestEnsemble$gbm_stacked <- predict(model_gbm, dfTestEnsemble[ , predictors_top])

caret::confusionMatrix(reference = dfTestEnsemble$class, data = dfTestEnsemble$gbm_stacked, mode='everything', positive='pos')


#### 4.3.2  GLM ####
# Stack using Logistic Regression
model_glm <- caret::train(dfTrainEnsemble[ , predictors_top], dfTrainEnsemble$class, method='glm', metric='ROC', trControl=fitControl, tuneLength=3)
model_glm

dfTestEnsemble$glm_stacked <- predict(model_glm, dfTestEnsemble[ , predictors_top])

caret::confusionMatrix(reference = dfTestEnsemble$class, data = dfTestEnsemble$glm_stacked, mode='everything', positive='pos')


#### 4.3.3  RF ####
# Stack using Random Forest
model_rf <- caret::train(dfTrainEnsemble[ , predictors_top], dfTrainEnsemble$class, method='rf', metric='ROC', trControl=fitControl, tuneLength=2)
model_rf

dfTestEnsemble$rf_stacked <- predict(model_rf, dfTestEnsemble[ , predictors_top])

caret::confusionMatrix(reference = dfTestEnsemble$class, data = dfTestEnsemble$rf_stacked, mode='everything', positive='pos')



# To stop parallel processing
cl <- makeCluster(3)
stopCluster(cl)





