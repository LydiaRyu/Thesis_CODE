
load('df_imp.RData')

library(dplyr)
library(caret)

df_imp$loan_status <- as.numeric(ifelse(df_imp$loan_status == 'Fully Paid' , 1, 0) )

cat('Fully Paid: ', (nrow(df_imp %>% filter(df_imp$loan_status == 1)) / nrow(df_imp)))
cat('\nCharged Off: ', nrow(df_imp %>% filter(df_imp$loan_status == 0)) / nrow(df_imp))



# One-hot Encoding

dmy <- dummyVars(~., data = df_imp)
df_imp <- data.frame(predict(dmy, newdata = df_imp))




##### Imputation Dataset #####


require(caTools)
library(ROSE)
library(ncvreg)
library(pROC)
library(e1071)


opt_lambda_scad_imp <- c()
Accuracy <- c()
F1 <-c()
AUC <- c()
Sensitivity <- c()
Specificity<- c()
Precision <- c()
Recall <- c()

for (i in 1:20){
  set.seed(i)
  sample = sample.split(df_imp$loan_status, SplitRatio = .70)
  train = subset(df_imp, sample == TRUE)
  test  = subset(df_imp, sample == FALSE)
  
  
  X_train = train[ , !colnames(train) %in% 'loan_status']
  y_train = train['loan_status']
  X_test = test[ , !colnames(train) %in% 'loan_status'] 
  y_test = test['loan_status']
  
  cv_scad <- cv.ncvreg(X_train, as.factor(y_train$loan_status), family = 'binomial'
                       , penalty = 'SCAD', gamma = 3.7, returnX = TRUE ,nfolds = 5 )
  
  
  opt_lambda_scad_imp[i] <- cv_scad$lambda.min
  
  
  model_scad <- ncvreg(X_train, as.factor(y_train$loan_status), 
                       family = 'binomial', penalty = 'SCAD', gamma = 3.7, returnX = TRUE 
                       , lambda = opt_lambda_scad_imp[i])
  
  
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

head(opt_lambda_scad_imp)
imp_scad_result <- cbind(Accuracy, F1, AUC, Sensitivity, Specificity, Precision, Recall)


save(opt_lambda_scad_imp, file = 'opt_lambda_scad_imp.RData')
save(imp_scad_result, file = 'imp_scad_result.RData')

round(colMeans(imp_scad_result)*100, 2)
round(apply(imp_scad_result, 2, sd)/sqrt(20)*100 , 2)
