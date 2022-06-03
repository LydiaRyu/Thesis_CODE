

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


library(xgboost)
require(caTools)
library(ROSE)
library(mltools)
library(pROC)
library(e1071)




Accuracy <- c() 
F1 <-c()
AUC <- c()
Sensitivity <- c()
Specificity<- c()
Precision <- c()
Recall <- c()
max_depth <- c()
etas <- c()
best_iter <- c()

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
  
  X_train_xgb <- data.matrix(X_train)
  y_train_xgb <- data.matrix(y_train)
  X_test_xgb <- data.matrix(X_test)
  y_test_xgb <- data.matrix(y_test)
  
  xgb_train <- xgb.DMatrix(data=X_train_xgb,label=y_train_xgb)
  xgb_test <- xgb.DMatrix(data=X_test_xgb,label=y_test_xgb)
  
  max.depths = c(4, 5, 6, 7)
  etas = c( 0.01, 0.05, 0.1, 0.15)
  
  params_table <- data.frame(max_depth, etas)
  params_table$best_iter <- 0
  params_table$best_score <- 0
  
  for (p in 1:nrow(params_table)) {
    
    params <- list(max_depth = params_table$max_depth[p],
                   etas = params_table$etas[p] )
    
    set.seed(i)
    
    xgb_cv <- xgb.cv(data = X_train_xgb, label = y_train_xgb,
                     params = params,
                     nfold = 5, metrics = "auc", objective = "binary:logistic", 
                     nrounds = 100, verbose = F, prediction = F )
    

    
    params_table$best_iter[p] <- xgb_model_cv$best_iter
    params_table$best_score[p] <- xgb_model_cv$best_score
    
    cat(p,':', xgb_model_cv$best_iter, xgb_model_cv$best_score, '\n')
  }
  
  
  params_table
  
  params_table[which.max(params_table$best_score),]

  
  best_case <- which.max(params_table$best_score)
  best_iter[i] <- params_table$best_iter[best_case]
  
  best_params <- list(
                      max_depth[i] = params_table$max_depth[best_case],
                      etas[i] = params_table$etas[best_case])
  
  
  model_xgb <- xgb.train(data = xgb_train,
                         nrounds= best_iter[i],  
                         objective= "binary:logistic",  
                         params = best_params,
                         nthread = 2,
                         eval_metric= "auc",             
                         watchlist=watchlist,
                         print_every_n = 50 )
  
  
  
  pred_xgb <- predict(model_xgb, X_test_xgb)
  pred_xgb <- ifelse(pred_xgb > 0.5, 1, 0)
  
  CM <- confusionMatrix(as.factor(pred_xgb), as.factor(y_test$loan_status), mode = "everything", positive = "0")
  roc_auc <- roc(as.factor(y_test$loan_status), as.numeric(pred_xgb))
  
  Accuracy[i] <- CM$overall["Accuracy"]
  
  F1[i] <- CM$byClass["F1"]
  Sensitivity[i] <- CM$byClass["Sensitivity"]
  Specificity[i] <- CM$byClass["Specificity"]
  Precision[i] <- CM$byClass["Precision"]
  Recall[i] <- CM$byClass["Recall"]
  
  AUC[i] <- roc_auc$auc
}


result_xgb_tuned <- cbind(Accuracy,  AUC, F1, Recall,  Precision, Specificity, Sensitivity)



save(result_xgb_tuned, file = 'result_xgb_tuned.RData')
save(max_depth, file = 'max_depth.RData' )
save(etas, file = 'etas.RData')
save(best_iter, file = 'best_iter.RData')


round(colMeans(result_xgb_tuned)*100, 2)
round(apply(result_xgb_tuned, 2, sd)/sqrt(20)*100 , 2)

