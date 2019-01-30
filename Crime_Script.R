require(lubridate)
require(chron)
require(tidyr)
require(plyr)
require(dplyr)
require(ggplot2)
require(ggmap)
require(maptools)
require(rgdal)
require(stringr)
library(data.table)
library(readr)
library(e1071)
library(randomForest)
library(sparklyr)
library(DBI)
library(purrr)
library(Hmisc)
library(corrplot)

spark_home <- "/opt/cloudera/parcels/SPARK2/lib/spark2"
spark_version <- "2.0.0"
sc <- spark_connect(master="yarn-client", version=spark_version, spark_home=spark_home)

crime <- fread("data/crime-latest.csv")
crime <- crime[complete.cases(crime),]
tempo <- as.POSIXlt(crime$Date, format= "%m/%d/%Y %I:%M:%S %p", tz="America/Chicago")
crime$dateonly <- as.POSIXct(strptime(tempo, format = "%Y-%m-%d", tz = "America/Chicago"))
crime$Day <- day(tempo)
crime$Month <- month(tempo)
crime$YearNew <- year(tempo)
crime$Hour <- hour(tempo)
crime$Minute <- minute(tempo)
crime$Weekday <- weekdays(tempo)
crime$MonthName <- month.abb[crime$Month]
crime$WeekdayInt <- tempo$wday

time_window <- chron(times = c("00:00:00", "06:00:00", "12:00:00", "18:00:00", "23:59:00"))
crime$time <- times(format(tempo, format= "%H:%M:%S"))
crime$TimeWindow <- cut(crime$time, breaks=time_window, labels=c("00-06","07-12", "13-18", "19-24"), include.lowest=TRUE)
setDT(crime)[, TimeWindowInt := .GRP, by = TimeWindow]

unique(crime$`Primary Type`)
crime$crime.type <- as.character(crime$`Primary Type`)
crime$crime.type <- ifelse(crime$crime.type %in% c("SEX OFFENSE", "CRIM SEXUAL ASSAULT", "PROSTITUTION", "OBSCENITY", "PUBLIC INDECENCY"), 'SEXUAL', crime$crime.type)
crime$crime.type <- ifelse(crime$crime.type %in% c("NARCOTICS", "LIQUOR LAW VIOLATION", "OTHER NARCOTIC VIOLATION"), 'DRUGS', crime$crime.type)
crime$crime.type <- ifelse(crime$crime.type %in% c("THEFT", "BURGLARY", "MOTOR VEHICLE THEFT", "ROBBERY", "CRIMINAL TRESPASS", "CRIMINAL DAMAGE", "ARSON"), 'THEFT', crime$crime.type)
crime$crime.type <- ifelse(crime$crime.type %in% c("HOMICIDE", "ASSAULT", "DOMESTIC VIOLENCE", "WEAPONS VIOLATION", "BATTERY", "CONCEALED CARRY LICENSE VIOLATION", "HUMAN TRAFFICKING"), 'LIFETHREATNING', crime$crime.type)
crime$crime.type <- ifelse(crime$crime.type %in% c("INTIMIDATION", "STALKING", "KIDNAPPING"), 'INTIMIDATION', crime$crime.type)
crime$crime.type <- ifelse(crime$crime.type %in% c("OTHER OFFENSE", "RITUALISM", "NON - CRIMINAL", "1134", "IUCR", "DECEPTIVE PRACTICE", "PUBLIC PEACE VIOLATION", "INTERFERENCE WITH PUBLIC OFFICER", "OFFENSE INVOLVING CHILDREN", "NON-CRIMINAL", "NON-CRIMINAL (SUBJECT SPECIFIED)"), 'OTHERS', crime$crime.type)
unique(crime$crime.type)
setDT(crime)[, CrimeTypeInt := .GRP, by = crime.type]
 
# ======Serious and non-serious crimes============
 
crime$serious <- with(crime, `Primary Type` %in% "SEX OFFENSE" | `Primary Type` %in% "CRIM SEXUAL ASSAULT" | `Primary Type` %in% "THEFT" | `Primary Type` %in% "BURGLARY" | `Primary Type` %in% "MOTOR VEHICLE THEFT" | `Primary Type` %in% "ROBBERY" | `Primary Type` %in% "CRIMINAL TRESPASS" | `Primary Type` %in% "CRIMINAL DAMAGE" | `Primary Type` %in% "ARSON" | `Primary Type` %in% "HOMICIDE" | `Primary Type` %in% "ASSAULT" | `Primary Type` %in% "DOMESTIC VIOLENCE" | `Primary Type` %in% "WEAPONS VIOLATION" | `Primary Type` %in% "BATTERY" | `Primary Type` %in% "CONCEALED CARRY LICENSE VIOLATION" | `Primary Type` %in% "HUMAN TRAFFICKING" | `Primary Type` %in% "STALKING" | `Primary Type` %in% "KIDNAPPING")
#==============================
crime$season <- as.factor(ifelse(crime$MonthName %in% c("Mar", "Apr", "May"), "spring", ifelse(crime$MonthName %in% c("Jun", "Jul", "Aug"), "summer", ifelse(crime$MonthName %in% c("Sep", "Oct", "Nov"), "fall", "winter"))))
 
 
 
 
crime$Arrest <- ifelse(as.character(crime$Arrest) == "TRUE", 1, 0)
crime$dateonlynew <- as.Date(crime$dateonly)
crime$Lon <- round(as.numeric(crime$Longitude), 2)
crime$Lat <- round(as.numeric(crime$Latitude), 2)



require(lubridate)
require(chron)
require(tidyr)
require(plyr)
require(dplyr)
require(ggplot2)
require(ggmap)
require(maptools)
require(rgdal)
require(stringr)
library(data.table)
library(readr)
library(e1071)
library(randomForest)
library(sparklyr)
library(DBI)
library(purrr)
library(Hmisc)
library(corrplot)

spark_home <- "/opt/cloudera/parcels/SPARK2/lib/spark2"
spark_version <- "2.0.0"
sc <- spark_connect(master="yarn-client", version=spark_version, spark_home=spark_home)

crime <- fread("data/crime-latest.csv")
crime <- crime[complete.cases(crime),]
tempo <- as.POSIXlt(crime$Date, format= "%m/%d/%Y %I:%M:%S %p", tz="America/Chicago")
crime$dateonly <- as.POSIXct(strptime(tempo, format = "%Y-%m-%d", tz = "America/Chicago"))
crime$Day <- day(tempo)
crime$Month <- month(tempo)
crime$YearNew <- year(tempo)
crime$Hour <- hour(tempo)
crime$Minute <- minute(tempo)
crime$Weekday <- weekdays(tempo)
crime$MonthName <- month.abb[crime$Month]
crime$WeekdayInt <- tempo$wday

time_window <- chron(times = c("00:00:00", "06:00:00", "12:00:00", "18:00:00", "23:59:00"))
crime$time <- times(format(tempo, format= "%H:%M:%S"))
crime$TimeWindow <- cut(crime$time, breaks=time_window, labels=c("00-06","07-12", "13-18", "19-24"), include.lowest=TRUE)
setDT(crime)[, TimeWindowInt := .GRP, by = TimeWindow]

unique(crime$`Primary Type`)
crime$crime.type <- as.character(crime$`Primary Type`)
crime$crime.type <- ifelse(crime$crime.type %in% c("SEX OFFENSE", "CRIM SEXUAL ASSAULT", "PROSTITUTION", "OBSCENITY", "PUBLIC INDECENCY"), 'SEXUAL', crime$crime.type)
crime$crime.type <- ifelse(crime$crime.type %in% c("NARCOTICS", "LIQUOR LAW VIOLATION", "OTHER NARCOTIC VIOLATION"), 'DRUGS', crime$crime.type)
crime$crime.type <- ifelse(crime$crime.type %in% c("THEFT", "BURGLARY", "MOTOR VEHICLE THEFT", "ROBBERY", "CRIMINAL TRESPASS", "CRIMINAL DAMAGE", "ARSON"), 'THEFT', crime$crime.type)
crime$crime.type <- ifelse(crime$crime.type %in% c("HOMICIDE", "ASSAULT", "DOMESTIC VIOLENCE", "WEAPONS VIOLATION", "BATTERY", "CONCEALED CARRY LICENSE VIOLATION", "HUMAN TRAFFICKING"), 'LIFETHREATNING', crime$crime.type)
crime$crime.type <- ifelse(crime$crime.type %in% c("INTIMIDATION", "STALKING", "KIDNAPPING"), 'INTIMIDATION', crime$crime.type)
crime$crime.type <- ifelse(crime$crime.type %in% c("OTHER OFFENSE", "RITUALISM", "NON - CRIMINAL", "1134", "IUCR", "DECEPTIVE PRACTICE", "PUBLIC PEACE VIOLATION", "INTERFERENCE WITH PUBLIC OFFICER", "OFFENSE INVOLVING CHILDREN", "NON-CRIMINAL", "NON-CRIMINAL (SUBJECT SPECIFIED)"), 'OTHERS', crime$crime.type)
unique(crime$crime.type)
setDT(crime)[, CrimeTypeInt := .GRP, by = crime.type]
 
# ======Serious and non-serious crimes============
 
crime$serious <- with(crime, `Primary Type` %in% "SEX OFFENSE" | `Primary Type` %in% "CRIM SEXUAL ASSAULT" | `Primary Type` %in% "THEFT" | `Primary Type` %in% "BURGLARY" | `Primary Type` %in% "MOTOR VEHICLE THEFT" | `Primary Type` %in% "ROBBERY" | `Primary Type` %in% "CRIMINAL TRESPASS" | `Primary Type` %in% "CRIMINAL DAMAGE" | `Primary Type` %in% "ARSON" | `Primary Type` %in% "HOMICIDE" | `Primary Type` %in% "ASSAULT" | `Primary Type` %in% "DOMESTIC VIOLENCE" | `Primary Type` %in% "WEAPONS VIOLATION" | `Primary Type` %in% "BATTERY" | `Primary Type` %in% "CONCEALED CARRY LICENSE VIOLATION" | `Primary Type` %in% "HUMAN TRAFFICKING" | `Primary Type` %in% "STALKING" | `Primary Type` %in% "KIDNAPPING")
#==============================
crime$season <- as.factor(ifelse(crime$MonthName %in% c("Mar", "Apr", "May"), "spring", ifelse(crime$MonthName %in% c("Jun", "Jul", "Aug"), "summer", ifelse(crime$MonthName %in% c("Sep", "Oct", "Nov"), "fall", "winter"))))
 
 
 
 
crime$Arrest <- ifelse(as.character(crime$Arrest) == "TRUE", 1, 0)
crime$dateonlynew <- as.Date(crime$dateonly)
crime$Lon <- round(as.numeric(crime$Longitude), 2)
crime$Lat <- round(as.numeric(crime$Latitude), 2)



#**Visualization**
#===================


#Now since we have done some sort of data exploration, cleansing and modification. Lets further explore and understand the pattern with Visualization.


#_Check the frequency of Crime Type in Chicago_
qplot(data = crime, crime.type, xlab = "Crimes", main ="Crimes in Chicago", fill = crime.type) + scale_y_continuous("Number of crimes")


#_Crime By year_
ggplot(data = crime) + geom_freqpoly(aes(x = Year, color = crime.type), binwidth = 1)

#_Crime type count by Time window which we created above_
crime <- crime[complete.cases(crime),]
qplot(data = crime, TimeWindow, xlab="Time Window", main = "Crimes by Time Window", fill = TimeWindow) + scale_y_continuous("Number of crimes")


#_Which day of the week has most of the crimes happening_
crime$Weekday <- factor(crime$Weekday, levels= c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"))
qplot(data = crime, Weekday, xlab= "Day of week", main= "Crimes by day of week", fill = Weekday) + scale_y_continuous("Number of crimes")

#_Crime by month and see how it varies._
crime$MonthName <- factor(crime$MonthName, levels= c("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))
qplot(data = crime, MonthName, xlab= "Month", main= "Crimes by month", fill = MonthName) + scale_y_continuous("Number of crimes")

#_Serious Crimes by Hour_
ggplot(crime, aes(x=Hour, fill=serious)) + geom_bar()

#_Serious Crimes by Season_
ggplot(crime, aes(x=season, fill=serious)) + geom_bar()


#**A simple heat map**

temp <- aggregate(crime$crime.type, by= list(crime$crime.type, crime$TimeWindow), FUN= length)
names(temp) <- c("crime.type", "TimeWindow", "count")
ggplot(temp, aes(x= crime.type, y= factor(TimeWindow))) +
  geom_tile(aes(fill= count)) +
  scale_x_discrete("Crime", expand = c(0,0)) +
  scale_y_discrete("Time of day", expand = c(0,-2)) +
  scale_fill_gradient("Number of crimes", low = "white", high = "blue") +
  theme_bw() + ggtitle("Crimes by time of day") +
  theme(panel.grid.major = element_line(colour = NA), panel.grid.minor = element_line(colour = NA))


#**Plot the crimes heat map on Chicago Map.**

chicago_map <- get_map(location = 'chicago', zoom = 11)
CrimeLocations <- as.data.frame(table(crime$Lon, crime$Lat))
names(CrimeLocations) <- c('long', 'lat', 'Frequency')
CrimeLocations$long <- as.numeric(as.character(CrimeLocations$long))
CrimeLocations$lat <- as.numeric(as.character(CrimeLocations$lat))
CrimeLocations <- subset(CrimeLocations, Frequency > 0)

ggmap(chicago_map) + geom_tile(data = CrimeLocations, aes(x = long, y = lat, alpha = Frequency),
                           fill = 'red') + theme(axis.title.y = element_blank(), axis.title.x = element_blank())



#**Feature Selection**
#======================


#Machine learning uses so called features (i.e. variables or attributes) to generate predictive models. Using a suitable combination of features is essential for obtaining high precision and accuracy. Because too many (unspecific) features pose the problem of overfitting the model, we generally want to restrict the features in our models to those, that are most relevant for the response variable we want to predict. Using as few features as possible will also reduce the complexity of our models, which means it needs less time and computer power to run and is easier to understand.
#There are several ways to identify how much each feature contributes to the model and to restrict the number of selected features. And there are three general classes of feature selection algorithms: filter methods, wrapper methods and embedded methods. Here we have used filter methods (with Pearson’s Correlation) and wrapper method (with Backward Elimination).


#**Filter Method with Pearson’s Correlation**
cor_data <- crime.data.new.reg1
cor_data$season <- NULL ###
cor_result <- cor(cor_data, method = "pearson")
corrplot(cor_result, order = "hclust")

library(caret)
correlationMatrix <- cor(cor_data, method = "pearson")
print(correlationMatrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.6) #Find the highly corelated data with cutoff of 60%.
print(highlyCorrelated)

#**Wrapper method with Backward Elimination using Spark ML would be used during the Model Building section**
#-------------------------------------------------------------------------------------------------------------



#**Model building, testing/training and selection with SparkML**
#===============================================================


#_Read from HDFS Parquet File with Spark_
sparklyr:::df_from_sql(sc, "SHOW TABLES")
crime_data_tbl <- spark_read_parquet(sc, 'crime_data_parq_tbl', "hdfs:///user/clouderanM/data1/crime")

#_Randomly partition the data into train and test sets._

crime_data_tbl_partition <- crime_data_tbl %>% sdf_partition(train = 0.75, test = 0.25, seed = 1919)

#_Create the partition table references as per above._
train_tbl <- crime_data_tbl_partition$train
test_tbl <- crime_data_tbl_partition$test
train_tbl


#We now train various models on the training data. Score the test data with the fitted models to choose the best fitted model (Without any model tuning parameters).


#_Model serious crime as a function of several predictors._

#ml_formula <- formula(serious ~ Area + Day + Month + Year + Hour + TimeWindowInt + HouseCrowded + LowerEducation + HI + CrimeTypeInt)
#ml_formula <- formula(serious ~ Area + Day + Month + Year + Hour + TimeWindowInt + HouseCrowded + LowerEducation + HI)
ml_formula <- formula(serious ~ . -CrimeTypeInt -Lon -Lat -seriousInt)

#_Train a logistic regression model_
ml_log <- ml_logistic_regression(train_tbl, ml_formula)

#_Train a logistic Nural Network model model_
ml_nn <- ml_multilayer_perceptron(train_tbl, ml_formula, layers = c(19,11,2))

#_Random forest model_
ml_rf <- ml_random_forest(train_tbl, ml_formula)

#_Decision Tree_
ml_dt <- ml_decision_tree(train_tbl, ml_formula)

#_Gradient Boosted Tree_
ml_gbt <- ml_gradient_boosted_trees(train_tbl, ml_formula)

#_Naive Bayes_
ml_nb <- ml_naive_bayes(train_tbl, ml_formula)


#Score the test data with all the trained models above. For the same would create a list of all the trained models and apply tthme to a score fucntion.


#_Bundle all the trained models into a single object_
ml_models <- list(
  "Logistic" = ml_log,
  "Decision Tree" = ml_dt,
  "Random Forest" = ml_rf,
  "Gradient Boosted Trees" = ml_gbt,
  "Neural Net" = ml_nn,
  "Naive Bayes" = ml_nb
)


#_Create a scoring function_
score_test_data <- function(model, data=test_tbl){
  pred <- sdf_predict(model, data)
  select(pred, serious, prediction)
}

#_Finally score the models_
ml_score <- lapply(ml_models, score_test_data)


#Now we can compare the model results by looking at the performance metrics: **lift, AUC, and accuracy**. Also we examine feature importance to see what features are most predictive of serious crimes as we didi in the feature selection section.



#_Lift compares how well the model predicts survival compared to random guessing. The function below to estimate model lift for each scored model on the test data._

calculate_lift <- function(scored_data) {
  scored_data %>%
    mutate(bin = ntile(desc(prediction), 10)) %>% 
    group_by(bin) %>% 
    summarize(count = sum(serious)) %>% 
    mutate(prop = count / sum(count)) %>% 
    arrange(bin) %>% 
    mutate(prop = cumsum(prop)) %>% 
    select(-count) %>% 
    collect() %>% 
    as.data.frame()
}


#_Now we can initialize the results and calculate the lift_
ml_gains <- data.frame(bin = 1:10, prop = seq(0, 1, len = 10), model = "Base")


#_Calculate the lift but first dettach Hmisc packge as it interfers with the summarize function of dyplyr_
library(Hmisc)
detach(package:Hmisc)

for(i in names(ml_score)){
  ml_gains <- ml_score[[i]] %>%
    calculate_lift %>%
    mutate(model = i) %>%
    rbind(ml_gains, .)
}

#_Since we have the gain details we can now plot the results_
ggplot(ml_gains, aes(x = bin, y = prop, colour = model)) +
  geom_point() + geom_line() +
  scale_color_brewer(type = "qual") + 
  labs(title = "Lift Chart for Predicting Serious Crimes - Test Data Set",
       subtitle = "Test Data Set",
       x = NULL,
       y = NULL)


#From the lift chart above we can find that the tree models (Gradient Boosted, Random Forest,Decision tree) should provide the best prediction. 


#**Model Accuracy**
#-------------------

#Receiver operating characteristic (ROC) curves are graphical plots that illustrate the performance of a binary classifier. They visualize the relationship between the true positive rate (TPR) against the false positive rate (FPR). The ideal model perfectly classifies all positive outcomes as true and all negative outcomes as false (i.e. TPR = 1 and FPR = 0). And the area under the curve (AUC) summarizes how good the model is across these threshold points simultaneously. An area of 1 indicates that for any threshold value, the model always makes perfect preditions. **which may never happen**. Good AUC values are between .6.6 and .8.8. 

#**While we cannot draw the ROC graph using Spark, we can extract the AUC values based on the predictions.**


#_Function for calculating accuracy_
calc_accuracy <- function(data, cutpoint = 0.5){
  data %>% 
    mutate(prediction = if_else(prediction > cutpoint, 1.0, 0.0)) %>%
    ml_classification_eval("prediction", "serious", "accuracy")
}

#_Calculate AUC and accuracy_
perf_metrics <- data.frame(
  model = names(ml_score),
  AUC = 100 * sapply(ml_score, ml_binary_classification_eval, "serious", "prediction"),
  Accuracy = 100 * sapply(ml_score, calc_accuracy),
  row.names = NULL, stringsAsFactors = FALSE)

#_Print AUC and Accuracy Table
print(perf_metrics)

#_Plot the results_
gather(perf_metrics, metric, value, AUC, Accuracy) %>%
  ggplot(aes(reorder(model, value), value, fill = metric)) + 
  geom_bar(stat = "identity", position = "dodge") + 
  coord_flip() +
  xlab("") +
  ylab("Percent") +
  ggtitle("Performance Metrics")


#As in the feature selection section filter and wraper methods were performed. Here from the above best performing models (tree based) we can use another way of wraper method which Recursive Feature elimination which is a greedy optimization algorithm which aims to find the best performing feature subset. It repeatedly creates models and keeps aside the best or the worst performing feature at each iteration.



#_Initialize the results_
feature_importance <- data.frame()

#_Calculate feature importance from the above best performing tree models._

for(i in c("Decision Tree", "Random Forest", "Gradient Boosted Trees")){
  feature_importance <- ml_tree_feature_importance(sc, ml_models[[i]]) %>%
    mutate(Model = i) %>%
    mutate(importance = as.numeric(levels(importance))[importance]) %>%
    mutate(feature = as.character(feature)) %>%
    rbind(feature_importance, .)
}

#_Plot the results_
feature_importance %>%
  ggplot(aes(reorder(feature, importance), importance, fill = Model)) + 
  facet_wrap(~Model) +
  geom_bar(stat = "identity") + 
  coord_flip() +
  xlab("") +
  ggtitle("Feature Importance")


#The above plot provides the features which were important for the three models.

#**Comparing the Run times of the above Models**

#The time to train a model is important. Use the following code to evaluate each model n times and plots the results.


#_Number of reps per model_
n <- 5

#_Format model formula as character_
format_as_character <- function(x){
  x <- paste(deparse(x), collapse = "")
  x <- gsub("\\s+", " ", paste(x, collapse = ""))
  x
}

#_Create model statements with timers_
format_statements <- function(y){
  y <- format_as_character(y[[".call"]])
  y <- gsub('ml_formula', ml_formula_char, y)
  y <- paste0("system.time(", y, ")")
  y
}

#_Convert model formula to character_
ml_formula_char <- format_as_character(ml_formula)

#_Create n replicates of each model statements with timers_
all_statements <- sapply(ml_models, format_statements) %>%
  rep(., n) %>%
  parse(text = .)

#_Evaluate all model statements_
res  <- map(all_statements, eval)

#_Compile results_
result <- data.frame(model = rep(names(ml_models), n),
                     time = sapply(res, function(x){as.numeric(x["elapsed"])})) 

#_Plot_
result %>% ggplot(aes(time, reorder(model, time))) + 
  geom_boxplot() + 
  geom_jitter(width = 0.4, aes(colour = model)) +
  scale_colour_discrete(guide = FALSE) +
  xlab("Seconds") +
  ylab("") +
  ggtitle("Model training times")


#From the above we can see that the Gradient boosted tree and Regression models take considerably longer to train than the other methods.

#_Create Random Forest (Entropy might be a little slower to compute because it makes use of the logarithm)_

fit_ml_rf <- ml_random_forest(
  train_tbl,
  response = "serious",
  features = c("Hour", "BelowPoverty", "Area", "District", "Year", "Month", "TimeWindowInt", "HI", "HouseCrowded", "PCI", "LowerEducation"),
  impurity = "gini", #
  max.bins = 64L, # default = 32L
  max.depth = 15L, # default = 5L
  num.trees = 100L,  
  learn.rate = 0.01, # default = 0.1
  col.sample.rate = 0.5,
  type = "classification",
)

#_Check the feature Importance again_
ml_tree_feature_importance(sc = sc, model = fit_ml_rf)


#_predict from the model for the training data_
fit_ml_rf %>%
  sdf_predict(train_tbl) %>%
  ml_classification_eval(label = "serious",
                         predicted_lbl = "prediction",
                         metric = "f1") # default, F1 score 


#_Now we can do predictions for the test data_
test_ml_rf <- sdf_predict(fit_ml_rf, test_tbl)

#_Calculate the F1 score for test data._
(test_rf_f1 <- test_ml_rf %>%
    ml_classification_eval(label = "serious",
                           predicted_lbl = "prediction",
                           metric = "f1"))

#_use ml_binary_classification_eval() to get back the AUC for the model.
(rf_test_auc <- test_ml_rf %>%
    ml_binary_classification_eval(label = "serious",
                                  score = "probability")) 



#**Keras Deep Learing with backend Tensor Flow - R (not on Spark)**
#====================================================================

#Keras is a high-level neural networks API developed with a focus on enabling fast experimentation. Keras has the following key features: (Reference: https://keras.rstudio.com/)
#-: Allows the same code to run on CPU or on GPU, seamlessly.

#-: User-friendly API which makes it easy to quickly prototype deep learning models.

#-: Built-in support for convolutional networks (for computer vision), recurrent networks (for sequence #######           processing),      and any combination of both.

#-: Supports arbitrary network architectures: multi-input or multi-output models, layer sharing, model sharing, etc.      This means that Keras is appropriate for building essentially any deep learning model, from a memory network to a     neural Turing machine.

#-: Is capable of running on top of multiple back-ends including TensorFlow, CNTK, or Theano.


#Now we can train and test on keras with TensorFlow backend

#_Load the Keras package_
require(keras)

#_Create a seperate Dataframe for Keras_
data.keras <- crime.data.new.reg

#_Remove all the features not important as we have seen in the feature selection methods._
data.keras$season <- NULL
#data.keras$District <- NULL
#data.keras$WeekdayInt <- NULL
data.keras$Lon <- NULL
data.keras$Lat <- NULL
#data.keras$Ward <- NULL
#data.keras$BelowPoverty <- NULL
#data.keras$Unemployed <- NULL
#data.keras$PCI <- NULL
data.keras$serious <- NULL



#_Load the data as Matrix_
crime.keras.data <- as.matrix(data.keras)

#_Remove the dimnames from the dataframe_
dimnames(crime.keras.data) <- NULL


#_Now we need to split the into training and test sets so that one can start building your mode._


#_Select a sample size_
index_sample <- sample(2, nrow(crime.keras.data), replace=TRUE, prob=c(0.70, 0.30))

#_Split the `crime.keras.data` data into train and test set_
crime.keras.data.training <- crime.keras.data[index_sample==1, 1:16]
crime.keras.data.test <- crime.keras.data[index_sample==2, 1:16]

#_Also split the class attribute of train and test_

crime.keras.data.trainingtarget <- crime.keras.data[index_sample==1, 17]
crime.keras.data.testtarget <- crime.keras.data[index_sample==2, 17]


#_One-Hot Encoding to transform the target attribute from a vector that contains values for each class value to a matrix with a boolean for each class value. This can be easily achieved by the **to_categorical()** function available in Keras._



#_One hot encode training target values_
crime.keras.data.trainLabels <- to_categorical(crime.keras.data.trainingtarget)

#_One hot encode test target values_
crime.keras.data.testLabels <- to_categorical(crime.keras.data.testtarget)

#_Print out the iris.testLabels to double check the result_
head(crime.keras.data.testLabels)


#**Building the Keras Model.**


#_We need to first initialize a sequential model with the help of the keras_model_sequential() function to initialize a sequential model_
model_rmsprop <- keras_model_sequential()


#_Since we are building a multi-layer perceptron model on keras, we would need a activation function that would  build a fully-connected layer to solve the problem statement we are working on. We will use the "relu" rectifier activation function which will be used in a hidden layer. Also would use the softmax activation function to be used in the output layer, which makes sure that the output values are in the range of 0 and 1 and may be used as predicted probabilities._

model_rmsprop %>% 
    layer_dense(units = 158, activation = 'relu', input_shape = c(16)) %>%
    layer_dropout(rate = 0.6) %>%
    layer_dense(units = 80, activation = 'relu') %>%
    layer_dropout(rate = 0.5) %>%
    #layer_dense(units = 72, activation = 'relu') %>% 
    #layer_dropout(rate = 0.3) %>%
    layer_dense(units = 22, activation = 'relu') %>% 
    layer_dropout(rate = 0.3) %>% 
    layer_dense(units = 8, activation = 'softmax')



#_To compile the above model, need to configure the model with the "RMSprop"" optimizer and the "categorical_crossentropy" loss function. Also the accuracy can be monitored during the training by passing 'accuracy' to the metrics argument._


RMSprop <- optimizer_rmsprop(lr = 0.1)
model_rmsprop %>% compile(optimizer=RMSprop, 
                  loss='binary_crossentropy', 
                  metrics='accuracy')

#_We can now fit the model to the training data; In this case,train the model for 40 epochs or iterations over all the samples in training and trainLabels, in batches of 500 samples._


history_rmsprop <- model_rmsprop %>% fit(
  crime.keras.data.training, crime.keras.data.trainLabels, 
  epochs = 60, batch_size = 100,
  #callbacks = callback_early_stopping(patience = 10, monitor = 'acc'), 
  validation_split = 0.2
 )


#_Plotting the results of fit_


#_Plot the history_
plot(history_rmsprop)


#_Plot the model loss_
plot(history_rmsprop$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l")
lines(history_rmsprop$metrics$val_loss, col="green")
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

#_Plot the model accuracy_
plot(history_rmsprop$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l")
lines(history_rmsprop$metrics$val_acc, col="green")
legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))



#Now we can predict the labels of the test data using the above model.


#_Predict the classes for the test data_
classes_rmsprop <- model_rmsprop %>% predict_classes(crime.keras.data.test, batch_size = 128)


#_Evaluating the model_

#_Display the Confusion matrix table_
table(crime.keras.data.testtarget, classes_rmsprop)

#_Evaluate on test data and labels_
score_rmsprop <- model_rmsprop %>% evaluate(crime.keras.data.test, crime.keras.data.testLabels, batch_size = 128)

#_Print the score_
print(score_rmsprop)

