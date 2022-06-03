
setwd("/home/jung01/JaeeunRyu")
load('df_logs.RData')


library(dplyr)
library(caret)

df_logs$loan_status <- as.numeric(ifelse(df_logs$loan_status == 'Fully Paid' , 1, 0) )

cat('Fully Paid: ', (nrow(df_logs %>% filter(df_logs$loan_status == 1)) / nrow(df_logs)))
cat('\nCharged Off: ', nrow(df_logs %>% filter(df_logs$loan_status == 0)) / nrow(df_logs))

# One-hot Encoding

dmy <- dummyVars(~., data = df_logs)
df_logs <- data.frame(predict(dmy, newdata = df_logs))

dim(df_logs)
head(df_logs)

require(caTools)
library(ROSE)
library(ncvreg)
library(pROC)
library(e1071)

opt_lambda_scad_logs <- c()
Accuracy <- c()
F1 <-c()
AUC <- c()
Sensitivity <- c()
Specificity<- c()
Precision <- c()
Recall <- c()

for (i in 1:20){
  set.seed(i)
  sample = sample.split(df_logs$loan_status, SplitRatio = .70)
  train = subset(df_logs, sample == TRUE)
  test  = subset(df_logs, sample == FALSE)
  
  
  X_train = train[ , !colnames(train) %in% 'loan_status']
  y_train = train['loan_status']
  X_test = test[ , !colnames(train) %in% 'loan_status'] 
  y_test = test['loan_status']
  
  cv_scad <- cv.ncvreg(X_train, as.factor(y_train$loan_status), family = 'binomial'
                       , penalty = 'SCAD', gamma = 3.7, returnX = TRUE ,nfolds = 5 )
  
  
  opt_lambda_scad_logs[i] <- cv_scad$lambda.min
  
  
  model_scad <- ncvreg(X_train, as.factor(y_train$loan_status), 
                       family = 'binomial', penalty = 'SCAD', gamma = 3.7, returnX = TRUE 
                       , lambda = opt_lambda_scad_logs[i])
  
  
  pred_scad <- predict(model_scad, as.matrix(X_test), type = 'class')
  
  CM <- confusionMatrix(as.factor(pred_scad), as.factor(y_test$loan_status), mode = "everything", positive = "0")
  roc_auc <- roc(as.factor(y_test$loan_status), as.numeric(pred_scad))
  
  Accuracy[i] <- CM$overall["Accuracy"]
  
  F1[i] <- CM$byClass["F1"]
  Sensitivity[i] <- CM$byClass["Sensitivity"]
  Specificity[i] <- CM$byClass["Specificity"]
  Precision[i] <- CM$byClass["Precision"]
  Recall[i] <- CM$byClass["Recall"]
  
  AUC[i] <- roc_auc$auc
}

head(opt_lambda_scad_logs)
logs_scad_result <- cbind(Accuracy, F1, AUC, Sensitivity, Specificity, Precision, Recall)


save(opt_lambda_scad_logs, file = 'opt_lambda_scad_logs.RData')
save(logs_scad_result, file = 'logs_scad_result.RData')


round(colMeans(logs_scad_result)*100, 2)
round(apply(logs_scad_result, 2, sd)/sqrt(20)*100 , 2)

