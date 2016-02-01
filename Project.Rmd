Data Preprocessing

# load libraries
library(caret)

# load the training set
pml.training <- read.csv("pml-training.csv", na.strings=c("NA",""))

# load the testing set
# Note: the testing set is not used in this analysis
# the set is only used for the second part of the assignment
# when the model is used to predict the classes
pml.testing <- read.csv("pml-testing.csv", na.strings=c("NA",""))

# summary(pml.training)
We are interested in variables that predict the movement The set contains a number of variables that can be removed:

X (= row number)
user_name (= the name of the subject)
cvtd_timestamp is removed because it is a factor instead of a numeric value and the raw_timestamp_part_1 + raw_timestamp_part_2 contain the same info in numeric format.

rIndex <- grep("X|user_name|cvtd_timestamp",names(pml.training))
pml.training <- pml.training[, -rIndex]
Some variable have near Zero variance which indicates that they do not contribute (enough) to the model. They are removed from the set.

nzv <- nearZeroVar(pml.training)
pml.training <- pml.training[, -nzv]
A number of variable contain (a lot of) NA's. Leaving them in the set not only makes the model creation slower, but also results in lower accuracy in the model. These variables will be removed from the set:

NAs <- apply(pml.training,2,function(x) {sum(is.na(x))}) 
pml.training <- pml.training[,which(NAs == 0)]
The original set is rather large (19622 obs. of 56 variables). We create a smaller training set of 80% of the original set

tIndex <- createDataPartition(y = pml.training$classe, p=0.2,list=FALSE) 
pml.sub.training <- pml.training[tIndex,] # 3927 obs. of 56 variables
pml.test.training <- pml.training[-tIndex,] # test set for cross validation
Model creation

We can now create a model based on the pre-processed data set. Note that at this point, we are still working with a large set of variables. We do have however a reduced number of rows.

A first attempt to create a model is done by fitting a single tree:

modFit <- train(pml.sub.training$classe ~.,data = pml.sub.training,method="rpart")
modFit

results <- modFit$results
round(max(results$Accuracy),4)*100
Note that running the train() function can take some time! The accuracy of the model is low: r round(max(results$Accuracy),4)*100 %

A second attempt to create a model is done by using Random forests:

ctrl   <- trainControl(method = "cv", number = 4, allowParallel = TRUE)
modFit <- train(pml.sub.training$classe ~.,data = pml.sub.training,method="rf",prof=TRUE, trControl = ctrl)
modFit

results <- modFit$results
round(max(results$Accuracy),4)*100
This second attempt provides us with a model that has a much higher accuracy: : r round(max(results$Accuracy),4)*100 %

Cross-validation

We now use the modFit to predict new values within the test set that we created for cross-validation:

pred <- predict(modFit,pml.test.training)
pml.test.training$predRight <- pred==pml.test.training$classe
table(pred, pml.test.training$classe)
As expected the predictions are not correct in all cases. We can calculate the accuracy of the prediction:

pRes <- postResample(pred, pml.test.training$classe)
pRes
The prediction fitted the test set even slightly better than the training set: r round(pRes[1],6)*100 %

Expected out of sample error

We can calculate the expected out of sample error based on the test set that we created for cross-validation:

cfM <- confusionMatrix(pred, pml.test.training$classe)
cfM
The expected out of sample error is: r 100 * round((1 - cfM$overall[1]),6) %

Note: The confusionMatrix function from the Caret package does provide all the information that we calculated 'by hand' in the first part of the Cross-validation. It shows that both methods provide the same answer.
