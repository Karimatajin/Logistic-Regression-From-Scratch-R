# Author : Karima Tajin
# Date : 5 March 2020
# Machine Learning in R 
# Project 3 : Logistic Regression

# Write a logistic regression algorithm in R using the logit (or sigmoid function) from scratch
# which prints the coefficients and accuracy metrics by using provided data
# Documents each step of the code and  describe what each line does
# for the algorithm, you may use basic statistical functions and graphing functions but not ML function
# explaining what solve() is doing if you use it
# test your model with test data
# compare your algorithm to the algorithms in R using glm() and predict()

###  the steps of Logistic Regression from scratch:
#1. Read the data
#2. Divide the data into training and test data
#3. Determine the number of iteration n
#4. Train the model using the training dataset
##4-1. Create a matrix X from the training data with k features and m observations
##4-2. Create the Y matrix
##4-3. Create a matrix(actually a vector),W, of k zeroes
##4-4. loop for n iterations
###4-4-1. Compute the sigmoid result, g(x)=1/(1+e^-(WX))
###4-4-2. Compute the gradient
#5. Make predictions using the test dataset
###5-1. Compute the predicted Y
#6. Test the model

# import tidyverse library:
library(tidyverse)
# get the working directory:
setwd("/Users/karimaidrissi/Desktop/DSSA 5201 ML/logistic regression")

# Load the data:
TajinRL <- read.csv("LogisticData.csv")

# dimension of the data:
dim(TajinRL) # there'r 10000 observations and 3 variables
# plotting the data:
plot <- ggplot(TajinRL, aes(x=X1, y=X2, col= Y)) + geom_point(aes(size=2))
plot
# we can see there is a strong relationship betweeen X1 and X2 data.

#To perform the logistic Regression model ,we need first to split our TajinRL into training and testing data. 
require(caTools) # using casTools package to split the data 
set.seed(100) # set the seed to get repeatable data
split1 = sample.split(TajinRL$Y, SplitRatio = 3/4) # choosing 75% of the data to be the training data 
split1 # TRUE and FALSE values for each observation in our data, 
# TRUE means that we should put that observation in training set
# FALSE means putting the observation in the testing set
# extracting training data and test data as two seperate dataframe
# using subset function to create the training and testing sets
train = subset(TajinRL, split1 == TRUE) # training set
test = subset(TajinRL, split1 == FALSE) # testing set
dim(train) # 750 training samples
dim(test) # 250 testing samples
rm(split1) #to keep the environment clean


X <- train[,-ncol(train)] # the independent variables X1 and X2
y <- train[,ncol(train)]# the dependent/outcome variable only Y

#3. Determine the number of iteration n
# the maximum number of iteration refers to KT
KT <- 150000

# First we will build some useful functions in Logistic Regression
# develop the sigmoid function :
# Sigmoid function has the shape of S, the function can be used to map values to(0,1)
# the equation of sigmoid function is S(X) = 1/ (1+ e(-z)) where:
# S(X) : the output is between 0 and 1 (probability estimate)
# z : input to the function, example Wx+b
# e : base of natural log
sigmoid = function(z){
  1 / (1 + exp(-z))
}

# implement the cost function :
# cost function in LR helps learner to correct or change behaviour to minimize mistakes.
# in order words, it's used to estimate how badly the models are preforming in terms of its ability 
# to estimate the relationship between X and Y.
#NB: %*% is the dot product in R
cost <- function(theta, X, y) {
  m <- nrow(y)  # number of training examples
  hx <- sigmoid(X %*% theta)
  J <- (t(-y)%*%log(hx)-t(1-y)%*%log(1-hx))/m
  return(J)
}

# develop the gradient function:
grad <- function(X, y, theta){
  m = nrow(X)
  X = as.matrix(X)
  d = as.matrix(sigmoid(X %*% theta))
  (1/m) * (t(X) %*% (d - y))
}
  
# Logistic Regression from scratch
Logistic_Regression <- function(X,y,KT){ # logistic regression needs 3 inputs unlike Linear regression that needs 2
  alpha = 0.001 
# y <- as.matrix(y) # ensuring y is a matrix
X = cbind(rep(1,nrow(X)),X)  # insert intercept column to X data matrix
theta <- matrix(rep(0, ncol(X)), nrow = ncol(X)) 
alpha=0.5
  # looping through the maximum number of iteration
  for (i in 1:KT){
    theta = theta - alpha * grad(X, y, theta)
  }
  return(theta)
}
  
# Predict the logistic Regression function:
logisticPrediction <- function(betas, newData){  
  X <- na.omit(newData) # remove all na in the newData file
  X = cbind(rep(1,nrow(X)),X) # combining X data with an extra column of 1 for the same amount of X data
  X <- as.matrix(X) # convert x data to a matrix
  return(sigmoid(X %*% betas)) # returning sigmoid function by multiplying x data with betas data
  } # close the logistic prediction function
  
  
# test the logistic regression function:
#glm is used to fit generalized linear models, specified by giving a symbolic
# description of the linear predictor and a description of the error distribution.
# glm(formula, family = gaussian, data, weights, subset,na.action, start = NULL, etastart, mustart, offset,control = list(...), model = TRUE, method = "glm.fit",x = FALSE, y = TRUE, singular.ok = TRUE, contrasts = NULL, ...)
  
testLogistic <- function(train,test, threshold) {
  model1 = glm(Y~., data=train, family=binomial) # using glm() function instead of lm() in linear regression 
    #  specify family = binomial to get the logistic regression model
  prediction1 = predict(model1, newdata=test, type = "response") # predict is a generic function for predictions from the results of various model fitting functions, means make predictions with the model1
                                                                   # to obtain the predicted probabilities we need to use type ="response".
  vals <- table(test$Y, TestPrediction > threshold)
  accuracy = (vals[1]+vals[4])/sum(vals) # the accuracy will measure the proportion of correct identifications
  print(paste("The R accuracy of the computed data",accuracy, sep = " "))
  sensitivity = vals[4]/(vals[2]+vals[4]) # return true positive rate
  specificity = vals[1]/(vals[1]+vals[3]) # return true negative rate
  print(paste("The R sensitivity of the computed data",sensitivity, sep = " ")) # printing R sensitivity
  print(paste("The R specificity of the computed data",specificity, sep = " ")) # printing R specificity
  } # end testing logistic regression
  
# using ROC to evalute and compare the predictive model
# ROC stands for Receiver Operating Characteristic, it is used to distinguish between the true positives and negatives in any predictive model
# RORC function is the same as testing the logist
ROC <- function(train, test) {
  model1 = glm(Y~.,data=train,family=binomial) 
  prediction1 = predict(model1, newdata=test, type="response")
  
  sensitivity = vector(mode = "numeric", length = 101)
  falsepositives = vector(mode = "numeric", length = 101)
  thresholds = seq(from = 0, to = 1, by = 0.01)
  for(i in seq_along(thresholds)) {
    vals <- table(test$Y, TestPrediction > thresholds[i])
    sensitivity[i] = vals[4]/(vals[2]+vals[4])
    falsepositives[i] = vals[3]/(vals[1]+vals[3]) # false positives, or 1 - specificity
    }  
  ggplot() + # plotting ROC curve that will sweep through all possible cutoffs
  geom_line(aes(falsepositives, sensitivity), colour="red") +
  geom_abline(slope = 1, intercept = 0, colour="blue") +
  labs(title="ROC Curve", x= "1 - Specificity (FP)", y="Sensitivity (TP)") +
  geom_text(aes(falsepositives, sensitivity), label=ifelse(((thresholds * 100) %% 10 == 0),thresholds,''),nudge_x=0,nudge_y=0)
}

## test our logistic regression function from scratch to find beta
beta <- Logistic_Regression(X,y,KT)
beta
#rep(1, nrow(X))  3.5967360944
#X1              -0.0003712027
#X2               1.5509343548



# testing our logistic prediction function 
X <- test[,-ncol(test)]
TestPrediction <- logisticPrediction(beta, X)
TestPrediction

# treshold value:
threshold= 0.5
# confusion matrix for threshold of 0.5
v1 <- table(test$Y, TestPrediction > threshold)
v1
#    FALSE TRUE
#0     1   17
#1     3  229

# calculating the accuracy of the computed data
#accuracy = (#true positives + #true negatives) /#all prediction
accuracy = (1 + 229 )/ 250
print(paste("The accuracy of the computed data",accuracy, sep = " "))
# caculating the sensitivity 
# sensivity = (#true positives) / (#true positives + #false negatives)
sensitivity = (229/(3+229))
print(paste("The sensitivity of the computed data",sensitivity, sep = " "))
# caculating the specificity
# specificity = #true negatives /( #true negatives + #false positives)
specificity = 1 /(1+17)
print(paste("The specificity of the computed data",specificity, sep = " "))

# Confusion matrix for threshold of 0.8
predictTrain= 0.8
v2 <- table(test$Y, TestPrediction > predictTrain)
v2

# FALSE TRUE
#0     7   11
#1    19  213
# accuracy
(7+213) / 250# accuracy is 0.88

# sensitivity
213 /(19+213) # sensitivity is   0.9181034

# Specificity
7 / (7+11)  # specificity is  0.3888889

# as we can see by increasing the value of threshold, the model's sensitivity increases and 
# specificity and accuracy decreases. So picking a good threshold values is often challenging
# ROC curve can help us to decide which value of the threshold is best
testLogistic(train, test, threshold)

# load ROCR package
#library(gplots)
#library(ROCR)

# using predictTrain to creat our ROC curve
#ROCRpred <- prediction(predictTrain, )

ROC(train, test)

# ROC starts at the point (0,0) and ends at the point (1,1) 
# ROC curve captures all thresholds simultaneously. the threshold decrease as you move from (0,0) to (1,1)
# if the threshold is higher the sensitivity is lower and the specificity is higher and vice versa true.
# A threshold around(0.4,0.8) on this ROC curve looks like a good choice in this case.



