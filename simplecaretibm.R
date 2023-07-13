setwd("D:/DATA ANALYTICS/BY DATASETS/HR ANALYTICS")
emp=read.csv("Attrition.csv")
library(tidymodels)
library(caret)
library(MLmetrics)
emp=emp%>%
  mutate(median_compensation = median(MonthlyIncome),
                 CompensationRatio = (MonthlyIncome/median(MonthlyIncome)),
                 CompensationLevel = case_when(
                   between(CompensationRatio, 0.75,1.25) ~ "Average",
                   between(CompensationRatio, 0, 0.75) ~ "Below",
                   between(CompensationRatio, 1.25, 2) ~ "Above"
                 ))

str(emp)
emp$CompensationLevel=as.factor(emp$CompensationLevel)
emp%>%summarise(emp$CompensationRatio)%>%group_by(emp$CompensationLevel)

ggplot(emp,aes(Attrition,JobSatisfaction,fill=Attrition))+
  geom_boxplot()

ggplot(emp,aes(factor(JobSatisfaction),MonthlyIncome,fill=Attrition))+
  geom_boxplot()

ggplot(emp,aes(factor(JobSatisfaction),CompensationRatio,fill=Attrition))+
  geom_boxplot()


# clean up data
emp1= emp %>%
  select(-c("DailyRate","EducationField", "EmployeeCount", 
            "MonthlyRate","StandardHours","TotalWorkingYears",
            "StockOptionLevel","Gender", "Over18", "OverTime",
            "median_compensation"))

# find numeric values
nums = unlist(lapply(emp1, is.numeric))
# save numeric variables for later
empnums = emp1[,nums]
# show numeric variables
head(empnums)
# calculate correlation matrix
correlationMatrix = cor(empnums)
# summarize the correlation matrix
correlationMatrix
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated = findCorrelation(correlationMatrix, cutoff=0.5)
# print colnames of highly correlated attributes
colnames(empnums[,highlyCorrelated])
correlationMatrix[,highlyCorrelated]

emp2=empnums[,-highlyCorrelated]
head(emp2)
colnames(emp2)
colnames(empnums)
# remove highly correlated variables to overcome multicollinearity
colnames(empnums)
highlyCorrelated = c(1,7,11,16,17,19)
empnums = empnums[,-highlyCorrelated]

str(empnums)
str(emp1)

emp1$Attrition=as.factor(emp1$Attrition)
emp1$BusinessTravel=as.factor(emp1$BusinessTravel)
emp1$Department=as.factor(emp1$Department)
emp1$JobRole=as.factor(emp1$JobRole)
emp1$MaritalStatus=as.factor(emp$MaritalStatus)

# select factor variables to convert, but leave Attrition out
varstodummy = emp1[,sapply(emp1, is.factor) & colnames(emp1) != "Attrition"]
head(varstodummy)

# Create dummy variables with caret
dummies = dummyVars( ~ ., data = varstodummy)
empdummy = predict(dummies, newdata = varstodummy)

colnames(empdummy)
empfull = data.frame(empdummy, empnums, Attrition = emp1$Attrition)
View(empfull)

# remove near zero variables (except for attr)
removecols = nearZeroVar(empfull, names = TRUE)
removecols
# Get all column names 
allcols = names(empfull)
# Remove from data
empfinal= empfull[ , setdiff(allcols, removecols)]


# create data folds for cross validation
myFolds = createFolds(empfinal$Attrition, k = 2)

f1 = function(data, lev = NULL, model = NULL) {
  f1val = MLmetrics::F1_Score(y_pred = data$pred,
                                y_true = data$obs,
                                positive = lev[1])
  c(F1 = f1val)
}


# Create reusable trainControl object: myControl
myControl = trainControl(
  method = "cv", 
  number = 3, 
  summaryFunction = f1,
  classProbs = TRUE, 
  verboseIter = TRUE,
  savePredictions = "final",
  index = myFolds
)

# Fit a simple baseline model
modelbaseline = train(
  Attrition ~ MonthlyIncome + JobSatisfaction + MonthlyIncome*JobSatisfaction,
  data = empfinal,
  metric = "F1",
  method = "glm",
  family = "binomial",
  trControl = myControl
)


modelbaseline
summary(modelbaseline)

#linear model
lg = train(
  Attrition ~ MonthlyIncome + JobSatisfaction + MonthlyIncome*JobSatisfaction,
  data = empfinal,
  method = "glm",
  family = "binomial",
  metric= "F1",
  trControl = myControl
)

#xgb boost model
xgb = train(
  Attrition ~ MonthlyIncome + JobSatisfaction + MonthlyIncome*JobSatisfaction,
  data = empfinal,
  method = "xgbTree",
  metric= "F1",
  trControl = myControl
)

#naive Baye's model
nb = train(
  Attrition ~ MonthlyIncome + JobSatisfaction + MonthlyIncome*JobSatisfaction,
  data = empfinal,
  method = "naive_bayes",
  metric= "F1",
  trControl = myControl
)

# Create model_list
modellist = list(baseline = modelbaseline, naivebayes = nb,  glmnet = lg, xgboost= xgb)
# Pass model_list to resamples(): resamples
resamples =resamples(modellist)
# Summarize the results
summary(resamples)
bwplot(resamples, metric = "F1")

# create confusion matrix for basline model
basePred = predict.train(modelbaseline, empfinal, type = "raw")
confusionMatrix(basePred, empfinal$Attrition, mode = "prec_recall")





