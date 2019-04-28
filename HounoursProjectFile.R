df = read.csv("creditcard.csv")
df
classDist <- table(df$Class)
barplot(classDist, main="Class Distribution", 
        xlab="Type of transaction")
df2 = df[c(2:31)]
library(caret)
train.index <- createDataPartition(df2$Class, p = .7, list = FALSE)
train <- df2[ train.index,]
test  <- df2[-train.index,]
max(df2$V1)
min(df2$V1)
train$standAmount<-(train$Amount-mean(train$Amount))/sd(train$Amount)
test$standAmount<-(test$Amount-mean(train$Amount))/sd(train$Amount)
within(train, rm("Amount"))
train$Amount<-NULL
test$Amount<-NULL
head(train)
train[,c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,30,29)]
test[,c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,30,29)]
library(e1071)

trctrl <- trainControl(method = "repeatedcv", number = 3, repeats = 2)


svm_Linear <- train(Class ~., data = train, method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)
