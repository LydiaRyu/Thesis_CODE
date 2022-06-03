
load('cor_col_0.9.RData')

library(dplyr)
library(caret)

cor_col_0.9$loan_status <- as.numeric(ifelse(cor_col_0.9$loan_status == 'Fully Paid' , 1, 0) )

cat('Fully Paid: ', (nrow(cor_col_0.9 %>% filter(cor_col_0.9$loan_status == 1)) / nrow(cor_col_0.9)))
cat('\nCharged Off: ', nrow(cor_col_0.9 %>% filter(cor_col_0.9$loan_status == 0)) / nrow(cor_col_0.9))



# One-hot Encoding

dmy <- dummyVars(~., data = cor_col_0.9)
cor_col_0.9 <- data.frame(predict(dmy, newdata = cor_col_0.9))




require(caTools)
library(ROSE)
library(ncvreg)
library(pROC)
library(e1071)


opt_lambda_scad_aug_n_2 <- c()
Accuracy <- c()
F1 <-c()
AUC <- c()
Sensitivity <- c()
Specificity<- c()
Precision <- c()
Recall <- c()

for (i in 1:20){
  set.seed(i)
  sample = sample.split(cor_col_0.9$loan_status, SplitRatio = .70)
  train = subset(cor_col_0.9, sample == TRUE)
  test  = subset(cor_col_0.9, sample == FALSE)
  
  n_size <- length(train$loan_status == 1) * 2
  df_rose <- ROSE(loan_status ~., data = train, N = n_size)$data
  
  
  X_train = df_rose[ , !colnames(train) %in% 'loan_status']
  y_train = df_rose['loan_status']
  X_test = test[ , !colnames(train) %in% 'loan_status'] 
  y_test = test['loan_status']
  
  cv_scad <- cv.ncvreg(X_train, as.factor(y_train$loan_status), family = 'binomial'
                       , penalty = 'SCAD', gamma = 3.7, returnX = TRUE ,nfolds = 5 )
  
  
  opt_lambda_scad_aug_n_2[i] <- cv_scad$lambda.min
  
  
  model_scad <- ncvreg(X_train, as.factor(y_train$loan_status), 
                       family = 'binomial', penalty = 'SCAD', gamma = 3.7, returnX = TRUE 
                       , lambda = opt_lambda_scad_aug_n_2[i])
  
  
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

head(opt_lambda_scad_aug_n_2)
aug_n_2_scad_result <- cbind(Accuracy,  AUC, F1, Recall,  Precision, Specificity, Sensitivity)

save(opt_lambda_scad_aug_n_2, file = 'opt_lambda_scad_aug_n_2.RData')
save(aug_n_2_scad_result, file = 'aug_n_2_scad_result.RData')

round(colMeans(aug_n_2_scad_result)*100, 2)
round(apply(aug_n_2_scad_result, 2, sd)/sqrt(20)*100 , 2)

