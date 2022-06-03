##### LightGBM #####

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


# install.packages("lightgbm")
library(lightgbm)
require(caTools)
library(ROSE)
library(pROC)

library(e1071)




Accuracy <- c() 
F1 <-c()
AUC <- c()
Sensitivity <- c()
Specificity<- c()
Precision <- c()
Recall <- c()
max_depth_lgb <- c()
learning_rate <- c()
best_iter_lgb <- c()

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

X_train_lgb <- data.matrix(X_train)
y_train_lgb <- data.matrix(y_train)
X_test_lgb <- data.matrix(X_test)
y_test_lgb <- data.matrix(y_test)

lgb_train <- lgb.Dataset(data=X_train_lgb,label=y_train_lgb)
lgb_test <- lgb.Dataset.create.valid(lgb_train, data = X_test_lgb, label = y_test_lgb)


max_depth <- c(3, 4, 5, 6)
learning_rate <- c(0.001, 0.01, 0.05, 0.1)

params_table <- data.frame(max_depth, learning_rate)
params_table$best_iter <- 0
params_table$best_score <- 0

for (p in 1:nrow(params_table)) {
  
  params <- list(objective = 'binary',
                 metric = 'auc',
                 feature_fraction = 0.7,
                 max_depth = params_table$max_depth[p],
                 learning_rate = params_table$learning_rate[p]
  )
  
  set.seed(i)
  
  lgb_model_cv <- lgb.cv(data = lgb_train, 
                         nfold = 5, 
                         nrounds = 100, 
                         params = params,
                         verbose = -1)
  
  params_table$best_iter[p] <- lgb_model_cv$best_iter
  params_table$best_score[p] <- lgb_model_cv$best_score
  
  cat(p,':', lgb_model_cv$best_iter, lgb_model_cv$best_score, '\n')
}




params_table

params_table[which.max(params_table$best_score),]


best_case <- which.max(params_table$best_score)
best_iter_lgb[i] <- params_table$best_iter[best_case]

best_params <- list(objective = 'binary',
                    metric = 'auc',
                    feature_fraction = 0.7,
                    max_depth = params_table$max_depth[best_case],
                    learning_rate = params_table$learning_rate[best_case])

max_depth_lgb[i] <- max_depth
learning_rate[i] <- learning_rate


set.seed(i)
lgb_model_final <- lgb.train(data = lgb_train,
                             nrounds = best_iter_lgb[i],
                             params = best_params,
                             verbose = -1)




pred_lgb <- predict(lgb_model_final, X_test_lgb)
pred_lgb <- ifelse(pred_lgb > 0.5, 1, 0)


CM <- confusionMatrix(as.factor(pred_lgb), as.factor(y_test_lgb), mode = "everything", positive = "0")
roc_auc <- roc(as.factor(y_test_lgb$loan_status), as.numeric(pred_lgb))

Accuracy[i] <- CM$overall["Accuracy"]

F1[i] <- CM$byClass["F1"]
Sensitivity[i] <- CM$byClass["Sensitivity"]
Specificity[i] <- CM$byClass["Specificity"]
Precision[i] <- CM$byClass["Precision"]
Recall[i] <- CM$byClass["Recall"]

AUC[i] <- roc_auc$auc

}

result_lgb_tuned <- cbind(Accuracy,  AUC, F1, Recall,  Precision, Specificity, Sensitivity)



save(result_lgb_tuned, file = 'result_lgb_tuned.RData')
save(max_depth_lgb, file = 'max_depth_lgb.RData' )
save(learning_rate, file = 'learning_rate.RData')
save(best_iter_lgb, file = 'best_iter_lgb.RData')


round(colMeans(result_lgb_tuned)*100, 2)
round(apply(result_lgb_tuned, 2, sd)/sqrt(20)*100 , 2)