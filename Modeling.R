dbSendQuery(sc,"CREATE DATABASE IF NOT EXISTS crime_db")
tbl_change_db(sc,"crime_db")

#sdf_copy_to(sc, crime.data.new.reg1)
crime_data_tbl <- sdf_copy_to(sc, crime.data.new.reg1)

crime_data_tbl_partition <- crime_data_tbl %>% sdf_partition(train = 0.75, test = 0.25, seed = 1919)
train_tbl <- crime_data_tbl_partition$train
test_tbl <- crime_data_tbl_partition$test



#Train the models
#ml_formula <- formula(serious ~ Area + Day + Month + Year + Hour + TimeWindowInt + HouseCrowded + LowerEducation + HI + CrimeTypeInt)
#ml_formula <- formula(serious ~ Area + Day + Month + Year + Hour + TimeWindowInt + HouseCrowded + LowerEducation + HI)
ml_formula <- formula(serious ~ . -CrimeTypeInt -Lon -Lat)
ml_log <- ml_logistic_regression(train_tbl, ml_formula)
ml_nn <- ml_multilayer_perceptron(train_tbl, ml_formula, layers = c(19,10,2))
ml_rf <- ml_random_forest(train_tbl, ml_formula)
ml_dt <- ml_decision_tree(train_tbl, ml_formula)
## Gradient Boosted Tree
ml_gbt <- ml_gradient_boosted_trees(train_tbl, ml_formula)
ml_nb <- ml_naive_bayes(train_tbl, ml_formula)

#Validation data
ml_models <- list(
  "Logistic" = ml_log,
  "Decision Tree" = ml_dt,
  "Random Forest" = ml_rf,
  "Gradient Boosted Trees" = ml_gbt,
  "Neural Net" = ml_nn,
  "Naive Bayes" = ml_nb
)



score_test_data <- function(model, data=test_tbl){
  pred <- sdf_predict(model, data)
  select(pred, serious, prediction)
}

ml_score <- lapply(ml_models, score_test_data)


###Compare results####

# Lift function
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


# Initialize results
ml_gains <- data.frame(bin = 1:10, prop = seq(0, 1, len = 10), model = "Base")


# Calculate lift
#But first dettach Hmisc packge as it interfers with the summarize function of dyplyr##
library(Hmisc)
detach(package:Hmisc)

for(i in names(ml_score)){
  ml_gains <- ml_score[[i]] %>%
    calculate_lift %>%
    mutate(model = i) %>%
    rbind(ml_gains, .)
}


# Plot results
ggplot(ml_gains, aes(x = bin, y = prop, colour = model)) +
  geom_point() + geom_line() +
  scale_color_brewer(type = "qual") + 
  labs(title = "Lift Chart for Predicting Serious Crimes - Test Data Set",
       subtitle = "Test Data Set",
       x = NULL,
       y = NULL)


####AUC and accuracy

# Function for calculating accuracy
calc_accuracy <- function(data, cutpoint = 0.5){
  data %>% 
    mutate(prediction = if_else(prediction > cutpoint, 1.0, 0.0)) %>%
    ml_classification_eval("prediction", "serious", "accuracy")
}

# Calculate AUC and accuracy
perf_metrics <- data.frame(
  model = names(ml_score),
  AUC = 100 * sapply(ml_score, ml_binary_classification_eval, "serious", "prediction"),
  Accuracy = 100 * sapply(ml_score, calc_accuracy),
  row.names = NULL, stringsAsFactors = FALSE)


# Plot results
gather(perf_metrics, metric, value, AUC, Accuracy) %>%
  ggplot(aes(reorder(model, value), value, fill = metric)) + 
  geom_bar(stat = "identity", position = "dodge") + 
  coord_flip() +
  xlab("") +
  ylab("Percent") +
  ggtitle("Performance Metrics")



###Feature importance

# Initialize results
feature_importance <- data.frame()

# Calculate feature importance
for(i in c("Decision Tree", "Random Forest", "Gradient Boosted Trees")){
  feature_importance <- ml_tree_feature_importance(sc, ml_models[[i]]) %>%
    mutate(Model = i) %>%
    mutate(importance = as.numeric(levels(importance))[importance]) %>%
    mutate(feature = as.character(feature)) %>%
    rbind(feature_importance, .)
}

# Plot results
feature_importance %>%
  ggplot(aes(reorder(feature, importance), importance, fill = Model)) + 
  facet_wrap(~Model) +
  geom_bar(stat = "identity") + 
  coord_flip() +
  xlab("") +
  ggtitle("Feature Importance")




#####Compare run times


# Number of reps per model
n <- 5

# Format model formula as character
format_as_character <- function(x){
  x <- paste(deparse(x), collapse = "")
  x <- gsub("\\s+", " ", paste(x, collapse = ""))
  x
}

# Create model statements with timers
format_statements <- function(y){
  y <- format_as_character(y[[".call"]])
  y <- gsub('ml_formula', ml_formula_char, y)
  y <- paste0("system.time(", y, ")")
  y
}

# Convert model formula to character
ml_formula_char <- format_as_character(ml_formula)

# Create n replicates of each model statements with timers
all_statements <- sapply(ml_models, format_statements) %>%
  rep(., n) %>%
  parse(text = .)

# Evaluate all model statements
res  <- map(all_statements, eval)

# Compile results
result <- data.frame(model = rep(names(ml_models), n),
                     time = sapply(res, function(x){as.numeric(x["elapsed"])})) 

# Plot
result %>% ggplot(aes(time, reorder(model, time))) + 
  geom_boxplot() + 
  geom_jitter(width = 0.4, aes(colour = model)) +
  scale_colour_discrete(guide = FALSE) +
  xlab("Seconds") +
  ylab("") +
  ggtitle("Model training times")
  
  