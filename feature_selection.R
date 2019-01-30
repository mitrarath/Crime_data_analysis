
#corrplot(cor(crime.data.new.reg1), method = "circle")
cor_data <- crime.data.new.reg1
cor_data$season <- NULL
cor_result <- cor(cor_data, method = "pearson")
corrplot(cor_result, order = "hclust")

library(caret)
correlationMatrix <- cor(cor_data, method = "pearson")
print(correlationMatrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.6)
print(highlyCorrelated)



##### Feature selection using Linear Regression######

crime_data_tbl <- crime.data.new.reg1 %>%
  as_data_frame() %>%
  copy_to(sc, ., name = "crime_data_tbl")

crime_data_tbl_partition <- crime_data_tbl %>% sdf_partition(train = 0.75, test = 0.25, seed = 1919)

fit_rf <- ml_linear_regression(
  crime_data_tbl_partition$train, # the training partion
  response = "serious",
  features = colnames(crime_data_tbl_partition$train)[1:20],
  seed = 2017
)
summary(fit_rf)

fit_rf_1 <- ml_linear_regression(
  crime_data_tbl_partition$train, # the training partion
  response = "serious",
  features = c("Area","District", "Day", "Month", "Year", "Hour", "TimeWindowInt", "Lon", "Lat", "season", "HouseCrowded", "Unemployed", "LowerEducation", "PCI", "HI", "CrimeTypeInt"),
  seed = 2017
)
summary(fit_rf_1)



fit_rf_2 <- ml_linear_regression(
  crime_data_tbl_partition$train, # the training partion
  response = "serious",
  features = c("Area", "Day", "Month", "Year", "Hour", "TimeWindowInt", "HouseCrowded", "LowerEducation", "HI", "CrimeTypeInt"),
  seed = 2017
)
summary(fit_rf_2)

