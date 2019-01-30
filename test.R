crime_data_tbl <- as_data_frame() %>% sdf_copy_to(sc, crime.data.new.reg1)

crime_data_tbl <- crime.data.new.reg1 %>%
  as_data_frame() %>%
  copy_to(sc, ., name = "crime_data_tbl")

crime_data_tbl_partition <- crime_data_tbl %>% sdf_partition(train = 0.75, test = 0.25, seed = 1919)

fit_rf <- ml_random_forest(
  crime_data_tbl_partition$train, # the training partion
  response = "serious",
  features = colnames(crime_data_tbl_partition$train)[1:19],
  impurity = "entropy",
  max.bins = 32L, # default = 32L
  max.depth = 5L, # default = 5L
  num.trees = 100L,  
  learn.rate = 0.1, # default = 0.1
  col.sample.rate = 0.25, # aka featureSubsetStrategy
  type = "classification",
  seed = 2017
)


test_rf <- sdf_predict(fit_rf, crime_data_tbl_partition$test)

ml_tree_feature_importance(sc = sc, model = fit_rf)

(test_rf_f1 <- test_rf %>%
     ml_classification_eval(label = "serious",
                            predicted_lbl = "prediction",
                            metric = "f1"))

(rf_test_auc <- test_rf %>%
    ml_binary_classification_eval(label = "serious",
                                  score = "probability")) 
                                  

rf_test_df <- collect(test_rf)
confusionMatrix(data = rf_test_df$prediction, reference = rf_test_df$serious)




========Linear Regression==============


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


test_rf <- sdf_predict(fit_rf, crime_data_tbl_partition$test)

ml_tree_feature_importance(sc = sc, model = fit_rf)

(test_rf_f1 <- test_rf %>%
     ml_classification_eval(label = "serious",
                            predicted_lbl = "prediction",
                            metric = "f1"))

(rf_test_auc <- test_rf %>%
    ml_binary_classification_eval(label = "serious",
                                  score = "probability")) 
                                  

rf_test_df <- collect(test_rf)
confusionMatrix(data = rf_test_df$prediction, reference = rf_test_df$serious)