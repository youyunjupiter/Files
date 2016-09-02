setwd("D:\\MouZhiHui\\Challenge\\12July");
library(ROCR)
library(pROC)
library(dummies)

Data = read.csv("t2_12July.csv");
Data_Cluster = read.csv("diagnosisClusters.csv");

Data$DiagA_desc = paste(Data$diag_1_desc, Data$diag_2_desc, Data$diag_3_desc, sep=" ");

Data_Train = Data[1:8000,];
Data_Train$cluster = as.factor(Data_Cluster$cluster);
Data_Test = Data[8001:10000,];

#install.packages("dummies", dependencies=TRUE) 

#N_diag_1D_3
SelVar = c(
"N_diag_1D_3",
"medical_speciality_missing",
"N_admission_source_2",
"N_AdmissionSeason_3",
"N_AdmissionYear",
"N_CH_insulin",
"N_discharge",
"N_nnumber_outpatient_YN",
"N_number_inpatient_LOG",
"N_payer_code_3",
"N_race_3",
"N_weight",
"N_YN_pioglitazone",
"number_diagnoses",
"cluster"
);
Data_Train.dum = cbind(as.factor(Data_Train$readmitted),dummy.data.frame(Data_Train[,SelVar]))
names(Data_Train.dum)[1] = "readmitted"

train.indeces = sample(1:nrow(Data_Train.dum), 7500)
Data_Train.trdum = Data_Train.dum[train.indeces, ]
Data_Train.tedum = Data_Train.dum[-train.indeces, ]

Data_Train.dum.tm  = cbind(Data_Train.dum,SVM_CLASSIFY.train$SVM_PROB)
Data_Train.dum.tm  = cbind(Data_Train.dum,RF_CLASSIFY.train$FORESTS_PROB)
Data_Train.dum.tm  = cbind(Data_Train.dum,BOOSTING_CLASSIFY.train$LOGITBOOST_PROB)
Data_Train.dum.tm  = cbind(Data_Train.dum,GLMNET_CLASSIFY.train$GLMNET_PROB)

eval()

eval = function(){

library(ROCR)
library(pROC)

train.indeces = sample(1:nrow(Data_Train.dum.tm), 7500)
Data_Train.trdum = Data_Train.dum.tm[train.indeces, ]
Data_Train.tedum = Data_Train.dum.tm[-train.indeces, ]

NumMethod = 16;
AUC_Train = rep(NA,NumMethod );
AUC_test = rep(NA,NumMethod );
ModelType = rep(NA,NumMethod );
aucyy = data.frame(AUC_Train, AUC_test, ModelType)

library(C50)
model <- C5.0(readmitted~ ., data = Data_Train.trdum)
results2 <- predict(object = model, newdata = Data_Train.tedum, type = "prob")
results2tr <- predict(object = model, newdata = Data_Train.trdum, type = "prob")
aucyy[1,1] = auc(as.numeric(Data_Train.trdum$readmitted)-1, results2tr[,2])
aucyy[1,2] = auc(as.numeric(Data_Train.tedum$readmitted)-1, results2[,2])
aucyy[1,3] = "C50"

library(e1071)
model <- svm(readmitted~ ., data = Data_Train.trdum, probability=TRUE)
results2 <- predict(object = model, newdata = Data_Train.tedum, probability=TRUE)
results2tr <- predict(object = model, newdata = Data_Train.trdum, probability=TRUE)
aucyy[2,1] = auc(as.numeric(Data_Train.trdum$readmitted)-1, attr(results2tr, "probabilities")[,2] )
aucyy[2,2] = auc(as.numeric(Data_Train.tedum$readmitted)-1, attr(results2, "probabilities")[,2] )
aucyy[2,3] = "SVM"

model <- naiveBayes(x = subset(Data_Train.trdum, select=-readmitted), 
	y = Data_Train.trdum$readmitted)
results2 <- predict(object = model, newdata = subset(Data_Train.tedum, select=-readmitted), type = "raw")
results2tr <- predict(object = model, newdata = subset(Data_Train.trdum, select=-readmitted), type = "raw")
aucyy[3,1] = auc(as.numeric(Data_Train.trdum$readmitted)-1, results2tr[,2])
aucyy[3,2] = auc(as.numeric(Data_Train.tedum$readmitted)-1, results2[,2])
aucyy[3,3] = "NB"

library(adabag)
model <- boosting(readmitted~ ., data = Data_Train.trdum)
results2 <- predict(object = model, newdata = Data_Train.tedum, type = "prob")
results2tr <- predict(object = model, newdata = Data_Train.trdum, type = "prob")
aucyy[4,1] = auc(as.numeric(Data_Train.trdum$readmitted)-1, results2tr$prob[,2])
aucyy[4,2] = auc(as.numeric(Data_Train.tedum$readmitted)-1, results2$prob[,2])
aucyy[4,3] = "Boosting"

library(rpart)
model <- rpart(readmitted~ ., data = Data_Train.trdum)
results2 <- predict(object = model, newdata = Data_Train.tedum, type = "prob")
results2tr <- predict(object = model, newdata = Data_Train.trdum, type = "prob")
aucyy[5,1] = auc(as.numeric(Data_Train.trdum$readmitted)-1, results2tr[,2])
aucyy[5,2] = auc(as.numeric(Data_Train.tedum$readmitted)-1, results2[,2])
aucyy[5,3] = "DRree"

library(class)
results2 <- knn(train = subset(Data_Train.trdum, select=-readmitted),
    test = subset(Data_Train.tedum, select=-readmitted),
    cl = Data_Train.trdum$readmitted,prob=TRUE)	
results2tr <- knn(train = subset(Data_Train.trdum, select=-readmitted),
    test = subset(Data_Train.trdum, select=-readmitted),
    cl = Data_Train.trdum$readmitted,prob=TRUE)	
aucyy[6,1] = auc(as.numeric(Data_Train.trdum$readmitted)-1, attr(results2tr,"prob"))
aucyy[6,2] = auc(as.numeric(Data_Train.tedum$readmitted)-1, attr(results2,"prob"))
aucyy[6,3] = "KNN"

library(glmnet)
#Ridge:Alpha = 0; Lasso: Alpha = 1; Elastic Net: Alpha btw 0-1
#model = glmnet(x = as.matrix(subset(Data_Train.trdum, select=-readmitted)), 
#	y = Data_Train.trdum$readmitted, alpha=0,family='binomial')
#results2 <- predict(object = model, newx= as.matrix(subset(Data_Train.tedum, select=-readmitted)),type="response")
#results2tr <- predict(object = model, newx= as.matrix(subset(Data_Train.trdum, select=-readmitted)),type="response")
#aucyy[7,1] = auc(Data_Train.trdum$readmitted, results2tr[,2])
#aucyy[7,2] = auc(Data_Train.tedum$readmitted, results2[,2])
#aucyy[7,3] = "GLM"

cv.model = cv.glmnet(x = as.matrix(subset(Data_Train.trdum, select=-readmitted)), 
	y = Data_Train.trdum$readmitted, alpha=0,family='binomial')
results2 <- predict(object = cv.model, newx= as.matrix(subset(Data_Train.tedum, select=-readmitted)),type="response",  s = "lambda.min")
results2tr <- predict(object = cv.model, newx= as.matrix(subset(Data_Train.trdum, select=-readmitted)),type="response",  s = "lambda.min")
aucyy[17,1] = auc(as.numeric(Data_Train.trdum$readmitted)-1, results2tr)
aucyy[17,2] = auc(as.numeric(Data_Train.tedum$readmitted)-1, results2)
aucyy[17,3] = paste("GLM",i,sep="")

for (i in 1:10){
i
cv.model = cv.glmnet(x = as.matrix(subset(Data_Train.trdum, select=-readmitted)), 
	y = Data_Train.trdum$readmitted, alpha=i/10,family='binomial')
results2 <- predict(object = cv.model, newx= as.matrix(subset(Data_Train.tedum, select=-readmitted)),type="response",  s = "lambda.min")
results2tr <- predict(object = cv.model, newx= as.matrix(subset(Data_Train.trdum, select=-readmitted)),type="response",  s = "lambda.min")
aucyy[(i+6),1] = auc(as.numeric(Data_Train.trdum$readmitted)-1, results2tr)
aucyy[(i+6),2] = auc(as.numeric(Data_Train.tedum$readmitted)-1, results2)
aucyy[(i+6),3] = paste("GLM",i,sep="")
}

aucyy

}

Data_Test.dum = cbind(as.factor(Data_Test$readmitted),dummy.data.frame(Data_Test[,SelVar]))
names(Data_Test.dum)[1] = "readmitted"
Data_Test.dum.tm  = cbind(Data_Test.dum,RF_CLASSIFY.test$FORESTS_PROB)
names(Data_Test.dum.tm)[length(names(Data_Test.dum.tm))] = "RF_CLASSIFY.train$FORESTS_PROB"

model.C50 <- C5.0(readmitted~ ., data = Data_Train.trdum)
results.test <- predict(object = model.C50, newdata = Data_Test.dum.tm, type = "prob")
results.test.f.C50 = data.frame(Data_Test$patientID, results.test[,2])
names(results.test.f.C50) = c("patientID", "Prob_C50")
write.csv(results.test.f.C50, "resultsSep_C50.csv")

model.svm <- svm(readmitted~ ., data = Data_Train.trdum, probability=TRUE)
results.test <- predict(object = model.svm, newdata = Data_Test.dum.tm, probability=TRUE)
results.test.f.SVM = data.frame(Data_Test$patientID, attr(results.test, "probabilities")[,2])
names(results.test.f.SVM) = c("patientID", "Prob_SVM")
write.csv(results.test.f.SVM, "resultsSep_SVM.csv")

model.boosting <- boosting(readmitted~ ., data = Data_Train.trdum)
results.test <- predict(object = model.boosting, newdata = Data_Test.dum.tm, type = "prob")
results.test.f.Boosting = data.frame(Data_Test$patientID, results.test$prob[,2])
names(results.test.f.Boosting) = c("patientID", "Prob_Boosting")
write.csv(results.test.f.Boosting, "resultsSep_Boosting.csv")

cv.model = cv.glmnet(x = as.matrix(subset(Data_Train.trdum, select=-readmitted)), 
	y = Data_Train.trdum$readmitted, alpha=0,family='binomial')
results.test <- predict(object = cv.model, newx= as.matrix(subset(Data_Test.dum.tm, select=-readmitted)),type="response",  s = "lambda.min")
results.test.f.GLM = data.frame(Data_Test$patientID, results.test)
names(results.test.f.GLM) = c("patientID", "Prob_GLM")
write.csv(results.test.f.GLM, "resultsSep_GLM.csv")

results.test.f = data.frame(Data_Test$patientID, results.test.f.C50[,2],
results.test.f.SVM[2], results.test.f.Boosting[2], results.test.f.GLM[2])
names(results.test.f) = c("patientID", "Prob_C50", "Prob_SVM", "Prob_Boosting", "Prob_GLM")
write.csv(results.test.f, "resultsSep.csv")


#memory consumer
library(randomForest)
set.seed(71)
names(Data_Train.trdum) = make.names(names(Data_Train.trdum))
model <- randomForest(readmitted~ ., data = Data_Train.trdum
#, importance=TRUE,proximity=TRUE
)

#################

#results <- predict(object = model, newdata = Data_Train.tedum, type = "class")
#table(results, Data_Train.tedum$readmitted)
results2 <- predict(object = model, newdata = Data_Train.tedum, type = "prob")
results2tr <- predict(object = model, newdata = Data_Train.trdum, type = "prob")
auc(Data_Train.trdum$readmitted, results2tr[,2])
auc(Data_Train.tedum$readmitted, results2[,2])

rocplot <- function(pred, truth, ...) {
  predob = prediction(pred, truth)
  perf = performance(predob, "tpr", "fpr")
  plot(perf, ...)
  area <- auc(truth, pred)
  area <- format(round(area, 4), nsmall = 4)
  text(x=0.8, y=0.1, labels = paste("AUC =", area))

  # the reference x=y line
  segments(x0=0, y0=0, x1=1, y1=1, col="gray", lty=2)
}

rocplot(probs, test$label, col="blue")




#To combine with text mining
library(tm)
DiagA.vec <- VectorSource(Data$DiagA_desc)
DiagA.corpus <- Corpus(DiagA.vec)	
DiagA.corpus <- tm_map(DiagA.corpus, content_transformer(tolower))
DiagA.corpus <- tm_map(DiagA.corpus, removePunctuation)
DiagA.corpus <- tm_map(DiagA.corpus, removeNumbers)
DiagA.corpus <- tm_map(DiagA.corpus, removeWords, stopwords("english"))
DiagA.corpus <- tm_map(DiagA.corpus, PlainTextDocument)

library(SnowballC)
DiagA.corpus <- tm_map(DiagA.corpus, stemDocument)
DiagA.corpus <- tm_map(DiagA.corpus, stripWhitespace)
#inspect(DiagA.corpus[1])

DiagA_matrix_te = DocumentTermMatrix(DiagA.corpus);
DiagA_matrix = DiagA_matrix_te;
#DiagA_matrix = removeSparseTerms(DiagA_matrix_te, 0.997)

DiagA_matrix = as.matrix(DiagA_matrix)
#Data_Train.tr = cbind(Data_Train.tr,DiagA_matrix)

library(cluster) 
d <- dist(DiagA_matrix, method="euclidian")   
fit <- hclust(d=d, method="ward.D")  
groups <- cutree(fit, k=5)
kfit <- kmeans(d, 5) 

###############
library(RTextTools)
DiagA_container = create_container(DiagA_matrix, Data$readmitted, trainSize=1:8000, testSize = 8001:10000,
virgin=FALSE)

SVM <- train_model(DiagA_container,"SVM")
GLMNET <- train_model(DiagA_container,"GLMNET")
#MAXENT <- train_model(DiagA_container,"MAXENT")
#SLDA <- train_model(DiagA_container,"SLDA")
BOOSTING <- train_model(DiagA_container,"BOOSTING")
#Bagging memory error
#BAGGING <- train_model(DiagA_container,"BAGGING")
RF <- train_model(DiagA_container,"RF")
#NNET <- train_model(DiagA_container,"NNET")
#TREE <- train_model(DiagA_container,"TREE")

SVM_CLASSIFY <- classify_model(DiagA_container, SVM)
GLMNET_CLASSIFY <- classify_model(DiagA_container, GLMNET)
#MAXENT_CLASSIFY <- classify_model(DiagA_container, MAXENT)
#SLDA_CLASSIFY <- classify_model(DiagA_container, SLDA)
BOOSTING_CLASSIFY <- classify_model(DiagA_container, BOOSTING)
#BAGGING_CLASSIFY <- classify_model(DiagA_container, BAGGING)
RF_CLASSIFY <- classify_model(DiagA_container, RF)
#NNET_CLASSIFY <- classify_model(DiagA_container, NNET)
#TREE_CLASSIFY <- classify_model(DiagA_container, TREE)

analytics <- create_analytics(DiagA_container,
cbind(SVM_CLASSIFY, GLMNET_CLASSIFY,
#MAXENT_CLASSIFY, SLDA_CLASSIFY,
BOOSTING_CLASSIFY,
#BAGGING_CLASSIFY,
RF_CLASSIFY
#NNET_CLASSIFY, 
#TREE_CLASSIFY
))
summary(analytics)

# CREATE THE data.frame SUMMARIES
topic_summary <- analytics@label_summary
alg_summary <- analytics@algorithm_summary
ens_summary <-analytics@ensemble_summary
doc_summary <- analytics@document_summary

#Ensemble agreement
create_ensembleSummary(analytics@document_summary)

#Cross Validation
SVM.cv <- cross_validate(DiagA_container, 4, "SVM")
GLMNET.cv <- cross_validate(DiagA_container, 4, "GLMNET")
MAXENT.cv <- cross_validate(DiagA_container, 4, "MAXENT")
SLDA.cv <- cross_validate(DiagA_container, 4, "SLDA")
BAGGING.cv <- cross_validate(DiagA_container, 4, "BAGGING")
BOOSTING.cv <- cross_validate(DiagA_container, 4, "BOOSTING")
RF.cv <- cross_validate(DiagA_container, 4, "RF")
NNET.cv <- cross_validate(DiagA_container, 4, "NNET")
TREE.cv <- cross_validate(DiagA_container, 4, "TREE")

write.csv(analytics@document_summary, "DocumentSummary.csv")

#For train data
DiagA_matrix.train= as.matrix(DiagA_matrix[1:8000,], originalMatrix=DiagA_matrix)
DiagA_container.train= create_container(DiagA_matrix.train, labels=rep(0,8000), trainSize=1:8000,virgin=FALSE)
SVM_CLASSIFY.train <- classify_model(DiagA_container.train, SVM)
GLMNET_CLASSIFY.train <- classify_model(DiagA_container.train, GLMNET)
BOOSTING_CLASSIFY.train <- classify_model(DiagA_container.train, BOOSTING)
RF_CLASSIFY.train <- classify_model(DiagA_container.train, RF)

#For test data
DiagA_matrix.test = as.matrix(DiagA_matrix[8001:10000,], originalMatrix=DiagA_matrix)
DiagA_container.test = create_container(DiagA_matrix.test, labels=rep(0,2000), trainSize=1:2000,virgin=FALSE)
SVM_CLASSIFY.test <- classify_model(DiagA_container.test, SVM)
GLMNET_CLASSIFY.test <- classify_model(DiagA_container.test, GLMNET)
BOOSTING_CLASSIFY.test <- classify_model(DiagA_container.test, BOOSTING)
RF_CLASSIFY.test <- classify_model(DiagA_container.test, RF)





