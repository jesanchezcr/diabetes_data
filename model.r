#install.packages("neuralnet", dependencies = TRUE) # Se instala una sola vez
library(neuralnet) 
library(readr)
library(varhandle)
library(dplyr)
library(aod)
library(nnet)
library(class)
library(mice)
library(naniar)
library(e1071)


# Import data
setwd("/Users/jenny/Documents/ML/dataset_diabetes/")
diabetic_data <- as.data.frame(read.csv("diabetic_data.csv",head=TRUE,sep=",",stringsAsFactors = TRUE))
diabetic_data <- subset(diabetic_data, select=-c(encounter_id,patient_nbr,examide,citoglipton,acetohexamide,troglitazone,glimepiride.pioglitazone,metformin.pioglitazone,metformin.rosiglitazone))


barplot(table(factor(diabetic_data$readmitted)))
barplot(table(factor(diabetic_data$gender)))
md.pattern(diabetic)

diabetic_data$admission_type_id <- factor(diabetic_data$admission_type_id)
diabetic_data$race <- factor(diabetic_data$race)
diabetic_data$gender <- factor(diabetic_data$gender)


diabetic_data <- replace_with_na(data=diabetic_data,replace=list(admission_type_id = c(5,6,8)))
diabetic_data <- replace_with_na(data=diabetic_data,replace=list(race = c('?')))
diabetic_data <- replace_with_na(data=diabetic_data,replace=list(gender = c('Unknown/Invalid')))


factor_to_number = function(x){
  if (class(x)=='factor')
    x%>%as.numeric(factor(x))
  else
    x
}
diabetic_data <-as.data.frame(sapply(diabetic_data,factor_to_number))

diabetic_data$admission_type_id <- factor(diabetic_data$admission_type_id)
diabetic_data$race <- factor(diabetic_data$race)
diabetic_data$gender <- factor(diabetic_data$gender)


mice_diabetic_data <- mice(diabetic_data,m=5,maxit=50,meth='polyreg',seed=500)
summary(mice_diabetic_data)
completed_data <- complete(mice_diabetic_data,1)

# Split Data into Training and Testing in R 
sample_size = floor(0.7*nrow(completed_data))
set.seed(777)

# randomly split data in r
picked = sample(seq_len(nrow(completed_data)),size = sample_size)
development = completed_data[picked,]
holdout = completed_data[-picked,]
completed_data <- as.data.frame(completed_data)

sapply(diabetic_data,FUN= function(s)  hist(s, main=paste("Histogram of",s), xlab = s))

sapply(development,class)

development <- as.data.frame(development)
holdout <- as.data.frame(holdout)



#neuralnet

nn <- neuralnet(factor(readmitted) ~ .-c(patient_nbr), data=development, hidden=c(3,2,1), linear.output=TRUE, threshold=0.01)
nn$result.matrix
plot(nn)


nn.results <- predict(nn, holdout)
holdout <- as.data.frame(holdout)
results <- data.frame(actual = holdout$readmitted, prediction = nn.results)
#multinomial logistic regression

mylogit <- multinom(factor(readmitted) ~ ., data = development,MaxNWts =10000000)
mylogit.results <- predict(mylogit,holdout)
results.logit <- data.frame(actual = holdout$readmitted, prediction = mylogit.results)
count(filter(results.logit,actual==prediction))/count(results.logit)
summary(mylogit)

# svm supervised
svm.model <- e1071::svm(readmitted ~ ., data = as.data.frame(development))
svm.results <- predict(svm.model,holdout)
results.svm <- data.frame(actual = na.omit(holdout)$readmitted, prediction = svm.results)
count(filter(results.svm,actual==round(prediction)))/count(results.svm)


#knn
knn.model <- knn(na.omit(development),na.omit(holdout),factor(na.omit(development)$readmitted),k=13)
tab <- table(knn.model,as.data.frame(na.omit(holdout))$readmitted)
results.knn <- data.frame(actual = as.data.frame(na.omit(holdout))$readmitted,prediction = knn.model)
count(filter(results.knn,actual==prediction))/count(results.knn)



varhandle::unfactor(factor(as.numeric(factor(diabetic_data$gender))))

