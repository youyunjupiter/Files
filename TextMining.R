Needed <- c("tm", "SnowballCC", "RColorBrewer", "ggplot2", "wordcloud", "biclust", "cluster", "igraph", "fpc")   
install.packages(Needed, dependencies=TRUE) 
Needed2 <- c("RTextTools")   
install.packages(Needed2, dependencies=TRUE) 

setwd("D:\\MouZhiHui\\Learning\\R\\textmining");
TEXTFILE = "pg100.txt"
download.file("http://www.gutenberg.org/cache/epub/100/pg100.txt", destfile = TEXTFILE)
shakespeare = readLines(TEXTFILE)
length(shakespeare)
head(shakespeare)
tail(shakespeare)
shakespeare = shakespeare[-(1:173)]
shakespeare = shakespeare[-(124195:length(shakespeare))]
shakespeare = paste(shakespeare, collapse = " ")
nchar(shakespeare)
shakespeare = strsplit(shakespeare, "<<[^>]*>>")[[1]]
dramatis.personae <- grep("Dramatis Personae", shakespeare, ignore.case = TRUE)
shakespeare = shakespeare[-dramatis.personae]

library(tm)
doc.vec <- VectorSource(shakespeare)
doc.corpus <- Corpus(doc.vec)	
summary(doc.corpus)
doc.corpus <- tm_map(doc.corpus, content_transformer(tolower))
doc.corpus <- tm_map(doc.corpus, removePunctuation)
doc.corpus <- tm_map(doc.corpus, removeNumbers)
doc.corpus <- tm_map(doc.corpus, removeWords, stopwords("english"))
doc.corpus <- tm_map(doc.corpus, PlainTextDocument)

library(SnowballC)
doc.corpus <- tm_map(doc.corpus, stemDocument)
doc.corpus <- tm_map(doc.corpus, stripWhitespace)
inspect(doc.corpus[1])
#doc.corpus[1]$content

TDM <- TermDocumentMatrix(doc.corpus)
DTM <- DocumentTermMatrix(doc.corpus)
inspect(DTM[1:10,1:10])
#words occured >2000 times
findFreqTerms(TDM, 2000)
findAssocs(TDM, "come", 0.8)

TDM.common = removeSparseTerms(TDM, 0.1)
inspect(TDM.common[1:10,1:10])

library(slam)
TDM.dense <- as.matrix(TDM.common)

library(reshape2)
TDM.dense = melt(TDM.dense, value.name = "count")
library(ggplot2)
ggplot(TDM.dense, aes(x = Docs, y = Terms, fill = log10(count))) +
     geom_tile(colour = "white") +
     scale_fill_gradient(high="#FF0000" , low="#FFFFFF")+
     ylab("") +
     theme(panel.background = element_blank()) +
     theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

findAssocs(DTM, c("love","act"), 0.7)
DTM.common = removeSparseTerms(DTM, 0.1)
DTM.dense <- as.matrix(DTM.common)
freq <- colSums(DTM.dense)   
freq["come"]

wf <- data.frame(word=names(freq), freq=freq)  
ggplot(subset(wf, freq>2000), aes(word, freq)) + 
geom_bar(stat="identity") + 
theme(axis.text.x=element_text(angle=45, hjust=1))   

library(wordcloud)
set.seed(142)   
wordcloud(names(freq), freq, min.freq=1000, scale=c(5, .1), colors=brewer.pal(6, "Dark2"))   
wordcloud(names(freq), freq, max.words=100, rot.per=0.2, colors=brewer.pal(6, "Dark2"))   

library(cluster) 
d <- dist(t(DTM.dense), method="euclidian")   
fit <- hclust(d=d, method="ward.D")  
plot(fit, hang=-1) 

plot.new()
plot(fit, hang=-1)
groups <- cutree(fit, k=5)   # "k=" defines the number of clusters you are using   
rect.hclust(fit, k=5, border="red") # draw dendogram with red borders around the 5 clusters 

library(fpc)   
kfit <- kmeans(d, 2)   
clusplot(as.matrix(d), kfit$cluster, color=T, shade=T, labels=2, lines=0)   

d2 = dist(DTM.dense, method="euclidian")
fit2 <- hclust(d=d2, method="ward.D")  
plot(fit2, hang=-1) 
kfit2 <- kmeans(d2, 2)   
clusplot(as.matrix(d2), kfit2$cluster, color=T, shade=T, labels=2, lines=0)   

DTM <- DocumentTermMatrix(doc.corpus)
DTM1 <- DocumentTermMatrix(doc.corpus,control=list(weighting=weightTfIdf, minWordLength=2, minDocFreq=5))


library(RTextTools)

#http://www.svm-tutorial.com/2014/11/svm-classify-text-r/
#https://journal.r-project.org/archive/2013-1/collingwood-jurka-boydstun-etal.pdf

test = c(rep(1,90),rep(0,92))
shakespeare_test = cbind(shakespeare,test)
shakespeare_test2 = create_matrix(shakespeare_test)

# Configure the training data
container <- create_container(shakespeare_test2, shakespeare_test[,2], trainSize=1:160,virgin=FALSE)
# train a SVM Model
model <- train_model(container, "SVM", kernel="linear", cost=1)
# new data
#predictionData <- list("love")
predictionData <- shakespeare[1:22]
# create a prediction document term matrix
predMatrix <- create_matrix(predictionData, originalMatrix=shakespeare_test2)
# create the corresponding container
predSize = length(predictionData);
predictionContainer <- create_container(predMatrix, labels=rep(0,predSize), testSize=1:predSize, virgin=FALSE)
# predict
results <- classify_model(predictionContainer, model)
results

##################
trace("create_matrix",edit=T)

###################
data(USCongress)

# CREATE THE DOCUMENT-TERM MATRIX
doc_matrix<- DocumentTermMatrix(
  Corpus(VectorSource(USCongress$text)), 
  control = list(
   wordLengths = c(3, Inf), 
   stopwords=TRUE, 
   removeNumbers=TRUE, 
   stemming=TRUE,
   removePunctuation = TRUE
  )
)
#Method 2: Not working due to NA
#doc_matrix <- create_matrix(USCongress$text, language="english", minDocFreq=1, maxDocFreq=Inf, 
#minWordLength=3, maxWordLength=Inf, ngramLength=1, originalMatrix=NULL, removeNumbers=TRUE, removePunctuation=TRUE, removeSparseTerms=0.1, 
#removeStopwords=TRUE,  stemWords=TRUE, stripWhitespace=TRUE, toLower=TRUE, weighting=weightTf)

container <- create_container(doc_matrix, USCongress$major, trainSize=1:4000,
testSize=4001:4449, virgin=FALSE)

SVM <- train_model(container,"SVM")
GLMNET <- train_model(container,"GLMNET")
MAXENT <- train_model(container,"MAXENT")
SLDA <- train_model(container,"SLDA")
BOOSTING <- train_model(container,"BOOSTING")
BAGGING <- train_model(container,"BAGGING")
RF <- train_model(container,"RF")
NNET <- train_model(container,"NNET")
TREE <- train_model(container,"TREE")

SVM_CLASSIFY <- classify_model(container, SVM)
GLMNET_CLASSIFY <- classify_model(container, GLMNET)
MAXENT_CLASSIFY <- classify_model(container, MAXENT)
SLDA_CLASSIFY <- classify_model(container, SLDA)
BOOSTING_CLASSIFY <- classify_model(container, BOOSTING)
BAGGING_CLASSIFY <- classify_model(container, BAGGING)
RF_CLASSIFY <- classify_model(container, RF)
NNET_CLASSIFY <- classify_model(container, NNET)
TREE_CLASSIFY <- classify_model(container, TREE)

analytics <- create_analytics(container,
cbind(SVM_CLASSIFY, GLMNET_CLASSIFY))
summary(analytics)

analytics <- create_analytics(container,
cbind(SVM_CLASSIFY, SLDA_CLASSIFY,
BOOSTING_CLASSIFY, BAGGING_CLASSIFY,
RF_CLASSIFY, GLMNET_CLASSIFY,
NNET_CLASSIFY, TREE_CLASSIFY,
MAXENT_CLASSIFY))
summary(analytics)

# CREATE THE data.frame SUMMARIES
topic_summary <- analytics@label_summary
alg_summary <- analytics@algorithm_summary
ens_summary <-analytics@ensemble_summary
doc_summary <- analytics@document_summary

#Ensemble agreement
create_ensembleSummary(analytics@document_summary)

#Cross Validation
SVM <- cross_validate(container, 4, "SVM")
GLMNET <- cross_validate(container, 4, "GLMNET")
MAXENT <- cross_validate(container, 4, "MAXENT")
SLDA <- cross_validate(container, 4, "SLDA")
BAGGING <- cross_validate(container, 4, "BAGGING")
BOOSTING <- cross_validate(container, 4, "BOOSTING")
RF <- cross_validate(container, 4, "RF")
NNET <- cross_validate(container, 4, "NNET")
TREE <- cross_validate(container, 4, "TREE")

write.csv(analytics@document_summary, "DocumentSummary.csv")













