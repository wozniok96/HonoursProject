#Haberman Dataset

#Import libraries
library(caret)
library(dplyr)
library(e1071)
library(ROSE)
library(ggplot2)
library(smotefamily)
library(MLmetrics)
library(DMwR)
library(UBL)
#Save Dataset into a datatframe.
df = read.csv("haberman.csv")
typeof(df)
#Produce a Class Distribution graph
classDist<-table(df$survival_status)
barplot(classDist, main="Class Distribution", 
        xlab="Survival Status")

#Split data into training and testing
train.index <- createDataPartition(df$survival_status, p = .7, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

#Standardize the input values
trainStd <- train
testStd<- test

trainStd$Age<-(train$Age-mean(train$Age))/sd(train$Age)
trainStd$year<-(train$year-mean(train$year))/sd(train$year)
trainStd$positive_axillary_nodes<-(train$positive_axillary_nodes-mean(train$positive_axillary_nodes))/sd(train$positive_axillary_nodes)

testStd$Age<-(test$Age-mean(train$Age))/sd(train$Age)
testStd$year<-(test$year-mean(train$year))/sd(train$year)
testStd$positive_axillary_nodes<-(test$positive_axillary_nodes-mean(train$positive_axillary_nodes))/sd(train$positive_axillary_nodes)



#Factorize the class label (required for using the tune function)
trainStd$survival_status<-as.factor(trainStd$survival_status)
testStd$survival_status<-as.factor(testStd$survival_status)

#Check Original Dataset distribution.
trainClassDist<-table(trainStd$survival_status)
barplot(trainClassDist, main="Class Distribution", 
        xlab="Survival Status")


#Plot the original training set distribution
ggplot(trainStd, aes(x=Age, y=positive_axillary_nodes, color=survival_status)) + geom_point(shape=16)

#Create a dataframe for collectiong results.
resultsDf <- data.frame("MethodName"=NA, "Sensitivity" = NA, "Specificity" = NA, "F1_Score"=NA, "G_Mean"=NA )
RFresultsDf <- data.frame("MethodName"=NA, "Sensitivity" = NA, "Specificity" = NA, "F1_Score"=NA, "G_Mean"=NA )
##tuneCtrl = tune.control(sampling = "cross", cross = 3)

trctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

tuneValues <- expand.grid(sigma = c(0.05, 0.1, 0.5, 0.9),
                          C = c(0.5, 1, 2,5,10))

tuneRfValues <- expand.grid(.mtry=c(1,2,3))
####################################################################################################
#sensVec <- vector()
#specVec <- vector()
#F1Vec <- vector()
#gMeanVec <-vector()

#sensVecRF <- vector()
#specVecRF <- vector()
#F1VecRF <- vector()
#gMeanVecRF <-vector()

#Find the best model using range of parameters (gridsearch)
bestSVM<- train(survival_status~., data = trainStd, method = "svmRadial", metric="Kappa",scale = FALSE,
            trControl=trctrl,
            tuneGrid = tuneValues)

bestSVM
plot(bestSVM) 



#Test the best model on testing data
Prediction <- predict(bestSVM, testStd[1:3])
summary(Prediction)

length(Prediction)
typeof(testStd[,4])
#Create a confusion matrix for the best model.
Tab <- table(pred=Prediction, true=testStd[,4])
Results <- confusionMatrix(Tab, positive = "2")
Results



sensVec<-c(sensVec, Results$byClass[[1]])
specVec<-c(specVec, Results$byClass[[2]])
F1Vec<-c(F1Vec, F1_Score(testStd[,4], Prediction, positive = "2"))
gMeanVec<-c(gMeanVec,sqrt(Results$byClass[[1]]*Results$byClass[[2]]))


resultsDf[1,"MethodName"]<-"Original"
resultsDf[1, "Sensitivity"]<-mean(sensVec)
resultsDf[1, "Specificity"]<-mean(specVec)
resultsDf[1, "F1_Score"]<-mean(F1Vec)
resultsDf[1, "G_Mean"]<-mean(gMeanVec)



bestRF <-train(survival_status~., data = trainStd, method = "rf", metric="Kappa",scale = FALSE,
            ntree=500,
            trControl=trctrl,
            tuneGrid = tuneRfValues)

bestRF
plot(bestRF)

PredictionRF <- predict(bestRF, testStd[1:3])
summary(PredictionRF)

#Create a confusion matrix for the best model.
TabRF <- table(pred=PredictionRF, true=testStd[,4])
ResultsRF <- confusionMatrix(TabRF, positive = "2")
ResultsRF

sensVecRF<-c(sensVecRF, ResultsRF$byClass[[1]])
specVecRF<-c(specVecRF, ResultsRF$byClass[[2]])
F1VecRF<-c(F1VecRF, F1_Score(testStd[,4], PredictionRF, positive = "2"))
gMeanVecRF<-c(gMeanVecRF,sqrt(ResultsRF$byClass[[1]]*ResultsRF$byClass[[2]]))

RFresultsDf[1,"MethodName"]<-"Original"
RFresultsDf[1, "Sensitivity"]<-mean(sensVecRF)
RFresultsDf[1, "Specificity"]<-mean(specVecRF)
RFresultsDf[1, "F1_Score"]<-mean(F1VecRF)
RFresultsDf[1, "G_Mean"]<-mean(gMeanVecRF)


######################################################################################
#weighedsensVec <- vector()
#weighedspecVec <- vector()
#weighedF1Vec <- vector()
#weighedgMeanVec <-vector()



weighedTuneValues <- expand.grid(sigma = c(0.05, 0.1, 0.5, 0.9),
                          C = c(0.5, 1, 2,5,10),
                          Weight=c(0.3,0.4,0.5,0.6))

weighedBestSVM<- train(survival_status~., data = trainStd, method = "svmRadialWeights",  metric="Kappa",scale = FALSE,
                trControl=trctrl,
                tuneGrid = weighedTuneValues)

weighedBestSVM
plot(weighedBestSVM)

#Test the best model on testing data
weighedPrediction <- predict(weighedBestSVM, testStd[1:3])
summary(weighedPrediction)

weighedTab <- table(pred=weighedPrediction, true=testStd[,4])
weighedResults <- confusionMatrix(weighedTab, positive = "2")
weighedResults

weighedsensVec<-c(weighedsensVec, weighedResults$byClass[[1]])
weighedspecVec<-c(weighedspecVec, weighedResults$byClass[[2]])
weighedF1Vec<-c(weighedF1Vec, F1_Score(testStd[,4], weighedPrediction, positive = "2"))
weighedgMeanVec<-c(weighedgMeanVec,sqrt(weighedResults$byClass[[1]]*weighedResults$byClass[[2]]))

resultsDf[12,"MethodName"]<-"Weighed"
resultsDf[12, "Sensitivity"]<-mean(weighedsensVec)
resultsDf[12, "Specificity"]<-mean(weighedspecVec)
resultsDf[12, "F1_Score"]<-mean(weighedF1Vec)
resultsDf[12, "G_Mean"]<-mean(weighedgMeanVec)

 ################################################################################################################
#oversensVec <- vector()
#overspecVec <- vector()
#overF1Vec <- vector()
#overgMeanVec <-vector()

#oversensVecRF <- vector()
#overspecVecRF <- vector()
#overF1VecRF <- vector()
#overgMeanVecRF <-vector()

#Create new training set using oversampling.
overTrain <- ovun.sample(survival_status~., trainStd, method="over", p=.5)$data


overClassDist<-table(overTrain$survival_status)
barplot(overClassDist, main="Class Distribution", 
        xlab="Survival Status")

ggplot(overTrain, aes(x=Age, y=positive_axillary_nodes, color=survival_status)) + geom_point(shape=16)


#Find the best model using range of parameters (gridsearch)
bestOverSVM<- train(survival_status~., data = overTrain, method = "svmRadial", metric="Kappa",scale = FALSE,
                trControl=trctrl,
                tuneGrid = tuneValues)

bestOverSVM
plot(bestOverSVM)

#Test the best model on testing data
overPrediction <- predict(bestOverSVM, testStd[1:3])
summary(overPrediction)

#Create a confusion matrix for the best model.
overTab <- table(pred=overPrediction, true=testStd[,4])
overResults<-confusionMatrix(overTab, positive ="2")
overResults

oversensVec<-c(oversensVec, overResults$byClass[[1]])
overspecVec<-c(overspecVec, overResults$byClass[[2]])
overF1Vec<-c(overF1Vec, F1_Score(testStd[,4], overPrediction, positive = "2"))
overgMeanVec<-c(overgMeanVec,sqrt(overResults$byClass[[1]]*overResults$byClass[[2]]))


resultsDf[2,"MethodName"]<-"Oversampling"
resultsDf[2, "Sensitivity"]<-mean(oversensVec)
resultsDf[2, "Specificity"]<-mean(overspecVec)
resultsDf[2, "F1_Score"]<-mean(overF1Vec)
resultsDf[2, "G_Mean"]<-mean(overgMeanVec)



bestOverRF <-train(survival_status~., data = overTrain, method = "rf", metric="Kappa",scale = FALSE,
               ntree=500,
               trControl=trctrl,
               tuneGrid = tuneRfValues)

bestOverRF
plot(bestOverRF)

overPredictionRF <- predict(bestOverRF, testStd[1:3])
summary(overPredictionRF)

#Create a confusion matrix for the best model.
overTabRF <- table(pred=overPredictionRF, true=testStd[,4])
overResultsRF <- confusionMatrix(overTabRF, positive = "2")
overResultsRF

oversensVecRF<-c(oversensVecRF, overResultsRF$byClass[[1]])
overspecVecRF<-c(overspecVecRF, overResultsRF$byClass[[2]])
overF1VecRF<-c(overF1VecRF, F1_Score(testStd[,4], overPredictionRF, positive = "2"))
overgMeanVecRF<-c(overgMeanVecRF,sqrt(overResultsRF$byClass[[1]]*overResultsRF$byClass[[2]]))

RFresultsDf[2,"MethodName"]<-"Oversampling"
RFresultsDf[2, "Sensitivity"]<-mean(oversensVecRF)
RFresultsDf[2, "Specificity"]<-mean(overspecVecRF)
RFresultsDf[2, "F1_Score"]<-mean(overF1VecRF)
RFresultsDf[2, "G_Mean"]<-mean(overgMeanVecRF)

##################################################################################################
#undersensVec <- vector()
#underspecVec <- vector()
#underF1Vec <- vector()
#undergMeanVec <-vector()

#undersensVecRF <- vector()
#underspecVecRF <- vector()
#underF1VecRF <- vector()
#undergMeanVecRF <-vector()

#Create new training set using undersampling.
underTrain <- ovun.sample(survival_status~., trainStd, method="under", p=.5)$data


underClassDist<-table(underTrain$survival_status)
barplot(underClassDist, main="Class Distribution", 
        xlab="Survival Status")

ggplot(underTrain, aes(x=Age, y=positive_axillary_nodes, color=survival_status)) + geom_point(shape=16)


#Find the best model using range of parameters (gridsearch)
bestUnderSVM <- train(survival_status~., data = underTrain, method = "svmRadial", metric="Kappa",scale = FALSE,
                  trControl=trctrl,
                  tuneGrid = tuneValues)

bestUnderSVM
plot(bestUnderSVM)

#Test the best model on testing data
underPrediction <- predict(bestUnderSVM, testStd[1:3])
summary(underPrediction)

#Create a confusion matrix for the best model.
underTab <- table(pred=underPrediction, true=testStd[,4])
underResults<-confusionMatrix(underTab, positive ="2")
underResults

undersensVec<-c(undersensVec, underResults$byClass[[1]])
underspecVec<-c(underspecVec, underResults$byClass[[2]])
underF1Vec<-c(underF1Vec, F1_Score(testStd[,4], underPrediction, positive = "2"))
undergMeanVec<-c(undergMeanVec,sqrt(underResults$byClass[[1]]*underResults$byClass[[2]]))


resultsDf[3,"MethodName"]<-"Undersampling"
resultsDf[3, "Sensitivity"]<-mean(undersensVec)
resultsDf[3, "Specificity"]<-mean(underspecVec)
resultsDf[3, "F1_Score"]<-mean(underF1Vec)
resultsDf[3, "G_Mean"]<-mean(undergMeanVec)




bestUnderRF <-train(survival_status~., data = underTrain, method = "rf", metric="Kappa",scale = FALSE,
                   ntree=500,
                   trControl=trctrl,
                   tuneGrid = tuneRfValues)

bestUnderRF
plot(bestUnderRF)

underPredictionRF <- predict(bestUnderRF, testStd[1:3])
summary(underPredictionRF)

#Create a confusion matrix for the best model.
underTabRF <- table(pred=underPredictionRF, true=testStd[,4])
underResultsRF <- confusionMatrix(underTabRF, positive = "2")
underResultsRF

undersensVecRF<-c(undersensVecRF, underResultsRF$byClass[[1]])
underspecVecRF<-c(underspecVecRF, underResultsRF$byClass[[2]])
underF1VecRF<-c(underF1VecRF, F1_Score(testStd[,4], underPredictionRF, positive = "2"))
undergMeanVecRF<-c(undergMeanVecRF,sqrt(underResultsRF$byClass[[1]]*underResultsRF$byClass[[2]]))

RFresultsDf[3,"MethodName"]<-"Undersampling"
RFresultsDf[3, "Sensitivity"]<-mean(undersensVecRF)
RFresultsDf[3, "Specificity"]<-mean(underspecVecRF)
RFresultsDf[3, "F1_Score"]<-mean(underF1VecRF)
RFresultsDf[3, "G_Mean"]<-mean(undergMeanVecRF)

#######################################################################################################
#bothsensVec <- vector()
#bothspecVec <- vector()
#bothF1Vec <- vector()
#bothgMeanVec <-vector()

#bothsensVecRF <- vector()
#bothspecVecRF <- vector()
#bothF1VecRF <- vector()
#bothgMeanVecRF <-vector()


bothTrain <- ovun.sample(survival_status~., trainStd, method="both", p=.5)$data


bothClassDist<-table(bothTrain$survival_status)
barplot(bothClassDist, main="Class Distribution", 
        xlab="Survival Status")

ggplot(bothTrain, aes(x=Age, y=positive_axillary_nodes, color=survival_status)) + geom_point(shape=16)


#Find the best model using range of parameters (gridsearch)
bestBothSVM <-train(survival_status~., data = bothTrain, method = "svmRadial", metric="Kappa",scale = FALSE,
                    trControl=trctrl,
                    tuneGrid = tuneValues)

bestBothSVM
plot(bestBothSVM)

#Test the best model on testing data
bothPrediction <- predict(bestBothSVM, testStd[1:3])
summary(bothPrediction)

#Create a confusion matrix for the best model.
bothTab <- table(pred=bothPrediction, true=testStd[,4])
bothResults<-confusionMatrix(bothTab, positive ="2")
bothResults

bothsensVec<-c(bothsensVec, bothResults$byClass[[1]])
bothspecVec<-c(bothspecVec, bothResults$byClass[[2]])
bothF1Vec<-c(bothF1Vec, F1_Score(testStd[,4], bothPrediction, positive = "2"))
bothgMeanVec<-c(bothgMeanVec,sqrt(bothResults$byClass[[1]]*bothResults$byClass[[2]]))


resultsDf[4,"MethodName"]<-"Over/Under"
resultsDf[4, "Sensitivity"]<-mean(bothsensVec)
resultsDf[4, "Specificity"]<-mean(bothspecVec)
resultsDf[4, "F1_Score"]<-mean(bothF1Vec)
resultsDf[4, "G_Mean"]<-mean(bothgMeanVec)


bestBothRF <-train(survival_status~., data = bothTrain, method = "rf", metric="Kappa",scale = FALSE,
                    ntree=500,
                    trControl=trctrl,
                    tuneGrid = tuneRfValues)

bestBothRF
plot(bestBothRF)

bothPredictionRF <- predict(bestBothRF, testStd[1:3])
summary(bothPredictionRF)

#Create a confusion matrix for the best model.
bothTabRF <- table(pred=bothPredictionRF, true=testStd[,4])
bothResultsRF <- confusionMatrix(bothTabRF, positive = "2")
bothResultsRF

bothsensVecRF<-c(bothsensVecRF, bothResultsRF$byClass[[1]])
bothspecVecRF<-c(bothspecVecRF, bothResultsRF$byClass[[2]])
bothF1VecRF<-c(bothF1VecRF, F1_Score(testStd[,4], bothPredictionRF, positive = "2"))
bothgMeanVecRF<-c(bothgMeanVecRF,sqrt(bothResultsRF$byClass[[1]]*bothResultsRF$byClass[[2]]))

RFresultsDf[4,"MethodName"]<-"Over/Under"
RFresultsDf[4, "Sensitivity"]<-mean(bothsensVecRF)
RFresultsDf[4, "Specificity"]<-mean(bothspecVecRF)
RFresultsDf[4, "F1_Score"]<-mean(bothF1VecRF)
RFresultsDf[4, "G_Mean"]<-mean(bothgMeanVecRF)

#########################################################################################
# rosesensVec <- vector()
# rosespecVec <- vector()
# roseF1Vec <- vector()
# rosegMeanVec <-vector()
# 
# rosesensVecRF <- vector()
# rosespecVecRF <- vector()
# roseF1VecRF <- vector()
# rosegMeanVecRF <-vector()


#Create a new dataset using ROSE method
roseTrain <- ROSE(survival_status~., trainStd, p=.5)$data


roseClassDist<-table(roseTrain$survival_status)
barplot(roseClassDist, main="Class Distribution", 
        xlab="Survival Status")

ggplot(roseTrain, aes(x=Age, y=positive_axillary_nodes, color=survival_status)) + geom_point(shape=16)


#Find the best model using range of parameters (gridsearch)
bestRoseSVM<- train(survival_status~., data = roseTrain, method = "svmRadial", metric="Kappa",scale = FALSE,
                trControl=trctrl,
                tuneGrid = tuneValues)

bestRoseSVM
plot(bestRoseSVM)

#Test the best model on testing data
rosePrediction <- predict(bestRoseSVM, testStd[1:3])
summary(rosePrediction)

#Create a confusion matrix for the best model.
roseTab <- table(pred=rosePrediction, true=testStd[,4])
roseResults<-confusionMatrix(roseTab, positive ="2")
roseResults


rosesensVec<-c(rosesensVec, roseResults$byClass[[1]])
rosespecVec<-c(rosespecVec, roseResults$byClass[[2]])
roseF1Vec<-c(roseF1Vec, F1_Score(testStd[,4], rosePrediction, positive = "2"))
rosegMeanVec<-c(rosegMeanVec,sqrt(roseResults$byClass[[1]]*roseResults$byClass[[2]]))


resultsDf[5,"MethodName"]<-"ROSE"
resultsDf[5, "Sensitivity"]<-mean(rosesensVec)
resultsDf[5, "Specificity"]<-mean(rosespecVec)
resultsDf[5, "F1_Score"]<-mean(roseF1Vec)
resultsDf[5, "G_Mean"]<-mean(rosegMeanVec)




bestRoseRF <-train(survival_status~., data = roseTrain, method = "rf", metric="Kappa",scale = FALSE,
                    ntree=500,
                    trControl=trctrl,
                    tuneGrid = tuneRfValues)

bestRoseRF
plot(bestRoseRF)

rosePredictionRF <- predict(bestRoseRF, testStd[1:3])
summary(rosePredictionRF)

#Create a confusion matrix for the best model.
roseTabRF <- table(pred=rosePredictionRF, true=testStd[,4])
roseResultsRF <- confusionMatrix(roseTabRF, positive = "2")
roseResultsRF


rosesensVecRF<-c(rosesensVecRF, roseResultsRF$byClass[[1]])
rosespecVecRF<-c(rosespecVecRF, roseResultsRF$byClass[[2]])
roseF1VecRF<-c(roseF1VecRF, F1_Score(testStd[,4], rosePredictionRF, positive = "2"))
rosegMeanVecRF<-c(rosegMeanVecRF,sqrt(roseResultsRF$byClass[[1]]*roseResultsRF$byClass[[2]]))

RFresultsDf[5,"MethodName"]<-"ROSE"
RFresultsDf[5, "Sensitivity"]<-mean(rosesensVecRF)
RFresultsDf[5, "Specificity"]<-mean(rosespecVecRF)
RFresultsDf[5, "F1_Score"]<-mean(roseF1VecRF)
RFresultsDf[5, "G_Mean"]<-mean(rosegMeanVecRF)

###############################################################################################
# smotesensVec <- vector()
# smotespecVec <- vector()
# smoteF1Vec <- vector()
# smotegMeanVec <-vector()
# 
# smotesensVecRF <- vector()
# smotespecVecRF <- vector()
# smoteF1VecRF <- vector()
# smotegMeanVecRF <-vector()

#Create a new training set using SMOTE method
smoteTrain<-SMOTE(trainStd[,-4], trainStd[,4], K = 3, dup_size = 0)$data
colnames(smoteTrain)[4] <- "survival_status"
smoteTrain$survival_status <- as.factor(smoteTrain$survival_status)

smoteClassDist<-table(smoteTrain$survival_status)
barplot(smoteClassDist, main="Class Distribution", 
        xlab="Survival Status")

ggplot(smoteTrain, aes(x=Age, y=positive_axillary_nodes, color=survival_status)) + geom_point(shape=16)


#Find the best model using range of parameters (gridsearch)
bestSmoteSVM <- train(survival_status~., data =smoteTrain, method = "svmRadial", metric="Kappa",scale = FALSE,
                 trControl=trctrl,
                 tuneGrid = tuneValues)
bestSmoteSVM
plot(bestSmoteSVM)

#Test the best model on testing data
smotePrediction <- predict(bestSmoteSVM, testStd[1:3])
summary(smotePrediction)

#Create a confusion matrix for the best model.
smoteTab <- table(pred=smotePrediction, true=testStd[,4])
smoteResults<-confusionMatrix(smoteTab, positive ="2")
smoteResults

smotesensVec<-c(smotesensVec, smoteResults$byClass[[1]])
smotespecVec<-c(smotespecVec, smoteResults$byClass[[2]])
smoteF1Vec<-c(smoteF1Vec, F1_Score(testStd[,4], smotePrediction, positive = "2"))
smotegMeanVec<-c(smotegMeanVec,sqrt(smoteResults$byClass[[1]]*smoteResults$byClass[[2]]))


resultsDf[6,"MethodName"]<-"SMOTE"
resultsDf[6, "Sensitivity"]<-mean(smotesensVec)
resultsDf[6, "Specificity"]<-mean(smotespecVec)
resultsDf[6, "F1_Score"]<-mean(smoteF1Vec)
resultsDf[6, "G_Mean"]<-mean(smotegMeanVec)



bestSmoteRF <-train(survival_status~., data = smoteTrain, method = "rf", metric="Kappa",scale = FALSE,
                   ntree=500,
                   trControl=trctrl,
                   tuneGrid = tuneRfValues)



bestSmoteRF
plot(bestSmoteRF)

smotePredictionRF <- predict(bestSmoteRF, testStd[1:3])
summary(smotePredictionRF)

#Create a confusion matrix for the best model.
smoteTabRF <- table(pred=smotePredictionRF, true=testStd[,4])
smoteResultsRF <- confusionMatrix(smoteTabRF, positive = "2")
smoteResultsRF

smotesensVecRF<-c(smotesensVecRF, smoteResultsRF$byClass[[1]])
smotespecVecRF<-c(smotespecVecRF, smoteResultsRF$byClass[[2]])
smoteF1VecRF<-c(smoteF1VecRF, F1_Score(testStd[,4], smotePredictionRF, positive = "2"))
smotegMeanVecRF<-c(smotegMeanVecRF,sqrt(smoteResultsRF$byClass[[1]]*smoteResultsRF$byClass[[2]]))

RFresultsDf[6,"MethodName"]<-"SMOTE"
RFresultsDf[6, "Sensitivity"]<-mean(smotesensVecRF)
RFresultsDf[6, "Specificity"]<-mean(smotespecVecRF)
RFresultsDf[6, "F1_Score"]<-mean(smoteF1VecRF)
RFresultsDf[6, "G_Mean"]<-mean(smotegMeanVecRF)
#############################################################################################################
# blssensVec <- vector()
# blsspecVec <- vector()
# blsF1Vec <- vector()
# blsgMeanVec <-vector()
# 
# blssensVecRF <- vector()
# blsspecVecRF <- vector()
# blsF1VecRF <- vector()
# blsgMeanVecRF <-vector()


blsTrain<-BLSMOTE(trainStd[,-4], trainStd[,4],K=3,C=3,dupSize=0,method = "type1")$data
colnames(blsTrain)[4] <- "survival_status"
blsTrain$survival_status <- as.factor(blsTrain$survival_status)

blsClassDist<-table(blsTrain$survival_status)
barplot(blsClassDist, main="Class Distribution", 
        xlab="Survival Status")

ggplot(blsTrain, aes(x=Age, y=positive_axillary_nodes, color=survival_status)) + geom_point(shape=16)


#Find the best model using range of parameters (gridsearch)
bestBlsSVM <- train(survival_status~., data = blsTrain, method = "svmRadial", metric="Kappa",scale = FALSE,
                    trControl=trctrl,
                    tuneGrid = tuneValues)

bestBlsSVM
plot(bestBlsSVM)

#Test the best model on testing data
blsPrediction <- predict(bestBlsSVM, testStd[1:3])
summary(blsPrediction)

#Create a confusion matrix for the best model.
blsTab <- table(pred=blsPrediction, true=testStd[,4])
blsResults<-confusionMatrix(blsTab, positive ="2")
blsResults

blssensVec<-c(blssensVec, blsResults$byClass[[1]])
blsspecVec<-c(blsspecVec, blsResults$byClass[[2]])
blsF1Vec<-c(blsF1Vec, F1_Score(testStd[,4], blsPrediction, positive = "2"))
blsgMeanVec<-c(blsgMeanVec,sqrt(blsResults$byClass[[1]]*blsResults$byClass[[2]]))


resultsDf[7,"MethodName"]<-"B-L SMOTE"
resultsDf[7, "Sensitivity"]<-mean(blssensVec)
resultsDf[7, "Specificity"]<-mean(blsspecVec)
resultsDf[7, "F1_Score"]<-mean(blsF1Vec)
resultsDf[7, "G_Mean"]<-mean(blsgMeanVec)


bestBlsRF <-train(survival_status~., data = blsTrain, method = "rf", metric="Kappa",scale = FALSE,
                   ntree=500,
                   trControl=trctrl,
                   tuneGrid = tuneRfValues)

bestBlsRF
plot(bestBlsRF)

blsPredictionRF <- predict(bestBlsRF, testStd[1:3])
summary(blsPredictionRF)

#Create a confusion matrix for the best model.
blsTabRF <- table(pred=blsPredictionRF, true=testStd[,4])
blsResultsRF <- confusionMatrix(blsTabRF, positive = "2")
blsResultsRF

blsTabRF <- table(pred=blsPredictionRF, true=testStd[,4])
blsResultsRF <- confusionMatrix(blsTabRF, positive = "2")
blsResultsRF

blssensVecRF<-c(blssensVecRF, blsResultsRF$byClass[[1]])
blsspecVecRF<-c(blsspecVecRF, blsResultsRF$byClass[[2]])
blsF1VecRF<-c(blsF1VecRF, F1_Score(testStd[,4], blsPredictionRF, positive = "2"))
blsgMeanVecRF<-c(blsgMeanVecRF,sqrt(blsResultsRF$byClass[[1]]*blsResultsRF$byClass[[2]]))


RFresultsDf[7,"MethodName"]<-"B-L SMOTE"
RFresultsDf[7, "Sensitivity"]<-mean(blssensVecRF)
RFresultsDf[7, "Specificity"]<-mean(blsspecVecRF)
RFresultsDf[7, "F1_Score"]<-mean(blsF1VecRF)
RFresultsDf[7, "G_Mean"]<-mean(blsgMeanVecRF)
##########################################################################################################
# slssensVec <- vector()
# slsspecVec <- vector()
# slsF1Vec <- vector()
# slsgMeanVec <-vector()
# 
# slssensVecRF <- vector()
# slsspecVecRF <- vector()
# slsF1VecRF <- vector()
# slsgMeanVecRF <-vector()


slsTrain<-SLS(trainStd[,-4], trainStd[,4], K = 3, dupSize = 0)$data
colnames(slsTrain)[4] <- "survival_status"
slsTrain$survival_status <- as.factor(slsTrain$survival_status)

slsClassDist<-table(slsTrain$survival_status)
barplot(slsClassDist, main="Class Distribution", 
        xlab="Survival Status")

ggplot(slsTrain, aes(x=Age, y=positive_axillary_nodes, color=survival_status)) + geom_point(shape=16)


#Find the best model using range of parameters (gridsearch)
bestSlsSVM<- train(survival_status~., data = slsTrain, method = "svmRadial", metric="Kappa",scale = FALSE,
                trControl=trctrl,
                tuneGrid = tuneValues)
bestSlsSVM
plot(bestSlsSVM)

#Test the best model on testing data
slsPrediction <- predict(bestSlsSVM, testStd[1:3])
summary(slsPrediction)

#Create a confusion matrix for the best model.
slsTab <- table(pred=slsPrediction, true=testStd[,4])
slsResults<-confusionMatrix(slsTab, positive ="2")
slsResults

slssensVec<-c(slssensVec, slsResults$byClass[[1]])
slsspecVec<-c(slsspecVec, slsResults$byClass[[2]])
slsF1Vec<-c(slsF1Vec, F1_Score(testStd[,4], slsPrediction, positive = "2"))
slsgMeanVec<-c(slsgMeanVec,sqrt(slsResults$byClass[[1]]*slsResults$byClass[[2]]))


resultsDf[8,"MethodName"]<-"S-L SMOTE"
resultsDf[8, "Sensitivity"]<-mean(slssensVec)
resultsDf[8, "Specificity"]<-mean(slsspecVec)
resultsDf[8, "F1_Score"]<-mean(slsF1Vec)
resultsDf[8, "G_Mean"]<-mean(slsgMeanVec)





bestSlsRF <-train(survival_status~., data = slsTrain, method = "rf", metric="Kappa",scale = FALSE,
                   ntree=500,
                   trControl=trctrl,
                   tuneGrid = tuneRfValues)

bestSlsRF
plot(bestSlsRF)

slsPredictionRF <- predict(bestSlsRF, testStd[1:3])
summary(slsPredictionRF)

#Create a confusion matrix for the best model.
slsTabRF <- table(pred=slsPredictionRF, true=testStd[,4])
slsResultsRF <- confusionMatrix(slsTabRF, positive = "2")
slsResultsRF

slssensVecRF<-c(slssensVecRF, slsResultsRF$byClass[[1]])
slsspecVecRF<-c(slsspecVecRF, slsResultsRF$byClass[[2]])
slsF1VecRF<-c(slsF1VecRF, F1_Score(testStd[,4], slsPredictionRF, positive = "2"))
slsgMeanVecRF<-c(slsgMeanVecRF,sqrt(slsResultsRF$byClass[[1]]*slsResultsRF$byClass[[2]]))


RFresultsDf[8,"MethodName"]<-"S-L SMOTE"
RFresultsDf[8, "Sensitivity"]<-mean(slssensVecRF)
RFresultsDf[8, "Specificity"]<-mean(slsspecVecRF)
RFresultsDf[8, "F1_Score"]<-mean(slsF1VecRF)
RFresultsDf[8, "G_Mean"]<-mean(slsgMeanVecRF)
#######################################################################################################
# adasynsensVec <- vector()
# adasynspecVec <- vector()
# adasynF1Vec <- vector()
# adasyngMeanVec <-vector()
# 
# adasynsensVecRF <- vector()
# adasynspecVecRF <- vector()
# adasynF1VecRF <- vector()
# adasyngMeanVecRF <-vector()


adasynTrain<-ADAS(trainStd[,-4], trainStd[,4], K = 3)$data
colnames(adasynTrain)[4] <- "survival_status"
adasynTrain$survival_status <- as.factor(adasynTrain$survival_status)

adasynClassDist<-table(adasynTrain$survival_status)
barplot(adasynClassDist, main="Class Distribution", 
        xlab="Survival Status")

ggplot(adasynTrain, aes(x=Age, y=positive_axillary_nodes, color=survival_status)) + geom_point(shape=16)


#Find the best model using range of parameters (gridsearch)
bestAdasynSVM <- train(survival_status~., data = adasynTrain, method = "svmRadial", metric="Kappa",scale = FALSE,
                   trControl=trctrl,
                   tuneGrid = tuneValues)
bestAdasynSVM
plot(bestAdasynSVM)

#Test the best model on testing data
adasynPrediction <- predict(bestAdasynSVM, testStd[1:3])
summary(adasynPrediction)

#Create a confusion matrix for the best model.
adasynTab <- table(pred=adasynPrediction, true=testStd[,4])
adasynResults<-confusionMatrix(adasynTab, positive ="2")
adasynResults

adasynsensVec<-c(adasynsensVec, adasynResults$byClass[[1]])
adasynspecVec<-c(adasynspecVec, adasynResults$byClass[[2]])
adasynF1Vec<-c(adasynF1Vec, F1_Score(testStd[,4], adasynPrediction, positive = "2"))
adasyngMeanVec<-c(adasyngMeanVec,sqrt(adasynResults$byClass[[1]]*adasynResults$byClass[[2]]))

resultsDf[9,"MethodName"]<-"ADASYN"
resultsDf[9, "Sensitivity"]<-mean(adasynsensVec)
resultsDf[9, "Specificity"]<-mean(adasynspecVec)
resultsDf[9, "F1_Score"]<-mean(adasynF1Vec)
resultsDf[9, "G_Mean"]<-mean(adasyngMeanVec)


bestAdasynRF <-train(survival_status~., data = adasynTrain, method = "rf", metric="Kappa",scale = FALSE,
                   ntree=500,
                   trControl=trctrl,
                   tuneGrid = tuneRfValues)

bestAdasynRF
plot(bestAdasynRF)

adasynPredictionRF <- predict(bestAdasynRF, testStd[1:3])
summary(adasynPredictionRF)

#Create a confusion matrix for the best model.
adasynTabRF <- table(pred=adasynPredictionRF, true=testStd[,4])
adasynResultsRF <- confusionMatrix(adasynTabRF, positive = "2")
adasynResultsRF

adasynsensVecRF<-c(adasynsensVecRF, adasynResultsRF$byClass[[1]])
adasynspecVecRF<-c(adasynspecVecRF, adasynResultsRF$byClass[[2]])
adasynF1VecRF<-c(adasynF1VecRF, F1_Score(testStd[,4], adasynPredictionRF, positive = "2"))
adasyngMeanVecRF<-c(adasyngMeanVecRF,sqrt(adasynResultsRF$byClass[[1]]*adasynResultsRF$byClass[[2]]))


RFresultsDf[9,"MethodName"]<-"ADASYN"
RFresultsDf[9, "Sensitivity"]<-mean(adasynsensVecRF)
RFresultsDf[9, "Specificity"]<-mean(adasynspecVecRF)
RFresultsDf[9, "F1_Score"]<-mean(adasynF1VecRF)
RFresultsDf[9, "G_Mean"]<-mean(adasyngMeanVecRF)
###############################################################################################################
# dbssensVec <- vector()
# dbsspecVec <- vector()
# dbsF1Vec <- vector()
# dbsgMeanVec <-vector()
# 
# dbssensVecRF <- vector()
# dbsspecVecRF <- vector()
# dbsF1VecRF <- vector()
# dbsgMeanVecRF <-vector()

dbsTrain<-DBSMOTE(trainStd[,-4], trainStd[,4] )$data
colnames(dbsTrain)[4] <- "survival_status"
dbsTrain$survival_status <- as.factor(dbsTrain$survival_status)

dbsClassDist<-table(dbsTrain$survival_status)
barplot(dbsClassDist, main="Class Distribution", 
        xlab="Survival Status")

ggplot(dbsTrain, aes(x=Age, y=positive_axillary_nodes, color=survival_status)) + geom_point(shape=16)


#Find the best model using range of parameters (gridsearch)
bestDbsSVM<- train(survival_status~., data = dbsTrain, method = "svmRadial", metric="Kappa",scale = FALSE,
             trControl=trctrl,
             tuneGrid = tuneValues)
bestDbsSVM
plot(bestDbsSVM)



#Test the best model on testing data
dbsPrediction <- predict(bestDbsSVM, testStd[1:3])
summary(dbsPrediction)

#Create a confusion matrix for the best model.
dbsTab <- table(pred=dbsPrediction, true=testStd[,4])
dbsResults<-confusionMatrix(dbsTab, positive ="2")
dbsResults

dbssensVec<-c(dbssensVec, dbsResults$byClass[[1]])
dbsspecVec<-c(dbsspecVec, dbsResults$byClass[[2]])
dbsF1Vec<-c(dbsF1Vec, F1_Score(testStd[,4], dbsPrediction, positive = "2"))
dbsgMeanVec<-c(dbsgMeanVec,sqrt(dbsResults$byClass[[1]]*dbsResults$byClass[[2]]))

resultsDf[10,"MethodName"]<-"DB SMOTE"
resultsDf[10, "Sensitivity"]<-mean(dbssensVec)
resultsDf[10, "Specificity"]<-mean(dbsspecVec)
resultsDf[10, "F1_Score"]<-mean(dbsF1Vec)
resultsDf[10, "G_Mean"]<-mean(dbsgMeanVec)



bestDbsRF <-train(survival_status~., data = dbsTrain, method = "rf", metric="Kappa",scale = FALSE,
                   ntree=500,
                   trControl=trctrl,
                   tuneGrid = tuneRfValues)

bestDbsRF
plot(bestDbsRF)

dbsPredictionRF <- predict(bestDbsRF, testStd[1:3])
summary(dbsPredictionRF)

#Create a confusion matrix for the best model.
dbsTabRF <- table(pred=dbsPredictionRF, true=testStd[,4])
dbsResultsRF <- confusionMatrix(dbsTabRF, positive = "2")
dbsResultsRF

dbssensVecRF<-c(dbssensVecRF, dbsResultsRF$byClass[[1]])
dbsspecVecRF<-c(dbsspecVecRF, dbsResultsRF$byClass[[2]])
dbsF1VecRF<-c(dbsF1VecRF, F1_Score(testStd[,4], dbsPredictionRF, positive = "2"))
dbsgMeanVecRF<-c(dbsgMeanVecRF,sqrt(dbsResultsRF$byClass[[1]]*dbsResultsRF$byClass[[2]]))


RFresultsDf[10,"MethodName"]<-"DB SMOTE"
RFresultsDf[10, "Sensitivity"]<-mean(dbssensVecRF)
RFresultsDf[10, "Specificity"]<-mean(dbsspecVecRF)
RFresultsDf[10, "F1_Score"]<-mean(dbsF1VecRF)
RFresultsDf[10, "G_Mean"]<-mean(dbsgMeanVecRF)
###########################################################################################################
# tlsmotesensVec <- vector()
# tlsmotespecVec <- vector()
# tlsmoteF1Vec <- vector()
# tlsmotegMeanVec <-vector()
# 
# tlsmotesensVecRF <- vector()
# tlsmotespecVecRF <- vector()
# tlsmoteF1VecRF <- vector()
# tlsmotegMeanVecRF <-vector()

#Create a new training set using SMOTE method
tlsmoteTrain<-SMOTE(trainStd[,-4], trainStd[,4], K = 3, dup_size = 0)$data
colnames(tlsmoteTrain)[4] <- "survival_status"
tlsmoteTrain$survival_status <- as.factor(smoteTrain$survival_status)
#Apply Tomek Links removal
TLsmoteTrain <- TomekClassif(survival_status~., tlsmoteTrain, dist = "Euclidean", 
                          Cl = "all", rem = "both")[[1]]

tlsmoteClassDist<-table(TLsmoteTrain$survival_status)
barplot(tlsmoteClassDist, main="Class Distribution", 
        xlab="Survival Status")

ggplot(TLsmoteTrain, aes(x=Age, y=positive_axillary_nodes, color=survival_status)) + geom_point(shape=16)


#Find the best model using range of parameters (gridsearch)
besttlSmoteSVM <- train(survival_status~., data =TLsmoteTrain, method = "svmRadial", metric="Kappa",scale = FALSE,
                      trControl=trctrl,
                      tuneGrid = tuneValues)
besttlSmoteSVM
plot(besttlSmoteSVM)

#Test the best model on testing data
tlsmotePrediction <- predict(besttlSmoteSVM, testStd[1:3])
summary(tlsmotePrediction)

#Create a confusion matrix for the best model.
tlsmoteTab <- table(pred=tlsmotePrediction, true=testStd[,4])
tlsmoteResults<-confusionMatrix(tlsmoteTab, positive ="2")
tlsmoteResults

tlsmotesensVec<-c(tlsmotesensVec, tlsmoteResults$byClass[[1]])
tlsmotespecVec<-c(tlsmotespecVec, tlsmoteResults$byClass[[2]])
tlsmoteF1Vec<-c(tlsmoteF1Vec, F1_Score(testStd[,4], tlsmotePrediction, positive = "2"))
tlsmotegMeanVec<-c(tlsmotegMeanVec,sqrt(tlsmoteResults$byClass[[1]]*tlsmoteResults$byClass[[2]]))


resultsDf[11,"MethodName"]<-"TL SMOTE"
resultsDf[11, "Sensitivity"]<-mean(tlsmotesensVec)
resultsDf[11, "Specificity"]<-mean(tlsmotespecVec)
resultsDf[11, "F1_Score"]<-mean(tlsmoteF1Vec)
resultsDf[11, "G_Mean"]<-mean(tlsmotegMeanVec)



besttlSmoteRF <-train(survival_status~., data = TLsmoteTrain, method = "rf", metric="Kappa",scale = FALSE,
                    ntree=500,
                    trControl=trctrl,
                    tuneGrid = tuneRfValues)



besttlSmoteRF
plot(besttlSmoteRF)

tlsmotePredictionRF <- predict(besttlSmoteRF, testStd[1:3])
summary(tlsmotePredictionRF)

#Create a confusion matrix for the best model.
tlsmoteTabRF <- table(pred=tlsmotePredictionRF, true=testStd[,4])
tlsmoteResultsRF <- confusionMatrix(tlsmoteTabRF, positive = "2")
tlsmoteResultsRF

tlsmotesensVecRF<-c(tlsmotesensVecRF, tlsmoteResultsRF$byClass[[1]])
tlsmotespecVecRF<-c(tlsmotespecVecRF, tlsmoteResultsRF$byClass[[2]])
tlsmoteF1VecRF<-c(tlsmoteF1VecRF, F1_Score(testStd[,4], tlsmotePredictionRF, positive = "2"))
tlsmotegMeanVecRF<-c(tlsmotegMeanVecRF,sqrt(tlsmoteResultsRF$byClass[[1]]*tlsmoteResultsRF$byClass[[2]]))

RFresultsDf[11,"MethodName"]<-"TL SMOTE"
RFresultsDf[11, "Sensitivity"]<-mean(tlsmotesensVecRF)
RFresultsDf[11, "Specificity"]<-mean(tlsmotespecVecRF)
RFresultsDf[11, "F1_Score"]<-mean(tlsmoteF1VecRF)
RFresultsDf[11, "G_Mean"]<-mean(tlsmotegMeanVecRF)



























###############################################################################################################
SensGraph <- ggplot(resultsDf, aes(x=MethodName, y=Sensitivity, fill=Sensitivity))

SensGraph + geom_bar(stat = "identity", position=position_dodge())+geom_text(aes(label=round(Sensitivity, digits=2)), vjust=1.6, color="black",
            position = position_dodge(0.9), size=5)
###########################################################################################################
SpecGraph <- ggplot(resultsDf, aes(x=MethodName, y=Specificity,fill=Specificity))

SpecGraph + geom_bar(stat = "identity", position=position_dodge())+geom_text(aes(label=round(Specificity, digits=2)), vjust=1.6, color="black",
                                                                               position = position_dodge(0.9), size=5)
###########################################################################################################
F1Graph <- ggplot(resultsDf, aes(x=MethodName, y=F1_Score, fill=F1_Score))

F1Graph + geom_bar(stat = "identity", position=position_dodge())+geom_text(aes(label=round(F1_Score, digits=2)), vjust=1.6, color="black",
                                                                           position = position_dodge(0.9), size=5)
 ##########################################################################################################
GmeanGraph <- ggplot(resultsDf, aes(x=MethodName, y=G_Mean, fill=G_Mean))

GmeanGraph + geom_bar(stat = "identity", position=position_dodge())+geom_text(aes(label=round(G_Mean, digits=2)), vjust=1.6, color="black",
                                                                              position = position_dodge(0.9), size=5)
###########################################################################################################
###########################################################################################################
###########################################################################################################
SensGraphRF <- ggplot(RFresultsDf, aes(x=MethodName, y=Sensitivity, fill=Sensitivity))

SensGraphRF + geom_bar(stat = "identity", position=position_dodge())+geom_text(aes(label=round(Sensitivity, digits=2)), vjust=1.6, color="black",
                                                                               position = position_dodge(0.9), size=5)
###########################################################################################################
SpecGraphRF <- ggplot(RFresultsDf, aes(x=MethodName, y=Specificity, fill=Specificity))

SpecGraphRF + geom_bar(stat = "identity", position=position_dodge())+geom_text(aes(label=round(Specificity, digits=2)), vjust=1.6, color="black",
                                                                               position = position_dodge(0.9), size=5)
###########################################################################################################
F1GraphRF <- ggplot(RFresultsDf, aes(x=MethodName, y=F1_Score, fill=F1_Score))

F1GraphRF + geom_bar(stat = "identity", position=position_dodge())+geom_text(aes(label=round(F1_Score, digits=2)), vjust=1.6, color="black",
                                                                             position = position_dodge(0.9), size=5)
##########################################################################################################
GmeanGraphRF <- ggplot(RFresultsDf, aes(x=MethodName, y=G_Mean, fill=G_Mean))

GmeanGraphRF + geom_bar(stat = "identity", position=position_dodge())+geom_text(aes(label=round(G_Mean, digits=2)), vjust=1.6, color="black",
                                                                                position = position_dodge(0.9), size=5)





Results
overResults
underResults
roseResults
smoteResults
blsResults
adasynResults
dbsResults
weighedResults
tlsmoteResults
###########################################################################################################
ResultsRF
overResultsRF
underResultsRF
roseResultsRF
smoteResultsRF
blsResultsRF
adasynResultsRF
dbsResultsRF
tlsmoteResultsRF
############################################################################################################


save.image(file='HabermanMyEnvironment.RData')
dir()
load('HabermanMyEnvironment.RData')




