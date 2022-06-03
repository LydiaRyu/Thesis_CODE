
load('cor_col_0.9.RData')
dim(cor_col_0.9)
library(dplyr)

cor_col_0.9$loan_status <- as.numeric(ifelse(cor_col_0.9$loan_status == 'Fully Paid' , 1, 0) )

cat('Fully Paid: ', (nrow(cor_col_0.9 %>% filter(cor_col_0.9$loan_status == 1)) / nrow(cor_col_0.9)))
cat('\nCharged Off: ', nrow(cor_col_0.9 %>% filter(cor_col_0.9$loan_status == 0)) / nrow(cor_col_0.9))


library(caret)

# One-hot Encoding

dmy <- dummyVars(~., data = cor_col_0.9)
cor_col_0.9 <- data.frame(predict(dmy, newdata = cor_col_0.9))

dim(cor_col_0.9)


##### Tuning #####
#p <- length(X_train)

#param_ntree <- c(50, 100, 150)
#param_mtry <- c(as.integer(sqrt(p)-3) : as.integer(sqrt(p)+3))

#for (i in param_ntree) {
#  cat('ntree: ', i , '\n')
#  for (j in param_mtry){
#    cat('mtry: ', j ,'\n')
    
#    model_rf <- randomForest(as.factor(y_train$loan_status)  ~ ., data = X_train
#                             , ntree = i, mtry = j, importance = TRUE)
    
#    print(model_rf)
#  }
#}
#################

require(caTools)
library(ROSE)
library(pROC)
library(randomForest)


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
  
  model_rf <- randomForest(as.factor(y_train$loan_status) ~ ., data = X_train
                           ,  ntree = 100, mtry = 11, importance = TRUE)
  
  pred_rf <- predict(model_rf, X_test)
  
  CM <- confusionMatrix(as.factor(pred_rf), as.factor(y_test$loan_status), mode = "everything", positive = "0")
  roc_auc <- roc(as.factor(y_test$loan_status), as.numeric(pred_rf))
  
  Accuracy[i] <- CM$overall["Accuracy"]
  
  F1[i] <- CM$byClass["F1"]
  Sensitivity[i] <- CM$byClass["Sensitivity"]
  Specificity[i] <- CM$byClass["Specificity"]
  Precision[i] <- CM$byClass["Precision"]
  Recall[i] <- CM$byClass["Recall"]
  
  AUC[i] <- roc_auc$auc
}


result_rf <- cbind(Accuracy, F1, AUC, Sensitivity, Specificity, Precision, Recall)
head(result_rf)


save(result_rf, file = 'result_rf.RData')

round(colMeans(result_rf)*100, 2)
round(apply(result_rf, 2, sd)/sqrt(20)*100 , 2)

