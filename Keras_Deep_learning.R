require(keras)
#c("Area", "Day", "Month", "Year", "Hour", "TimeWindowInt", "HouseCrowded", "LowerEducation", "HI", "CrimeTypeInt")
data.keras <- crime.data.new.reg1
data.keras$season <- NULL
data.keras$District <- NULL
#data.keras$WeekdayInt <- NULL
data.keras$Lon <- NULL
data.keras$Lat <- NULL
data.keras$Ward <- NULL
data.keras$BelowPoverty <- NULL
data.keras$Unemployed <- NULL
data.keras$PCI <- NULL

crime.keras.data <- as.matrix(data.keras)


# Set crime.data.int `dimnames` to `NULL`

dimnames(crime.keras.data) <- NULL


# Determine sample size

index_sample <- sample(2, nrow(crime.keras.data), replace=TRUE, prob=c(0.70, 0.30))


# Split the `crime.data.int` data
crime.keras.data.training <- crime.keras.data[index_sample==1, 1:19]
crime.keras.data.test <- crime.keras.data[index_sample==2, 1:19]

# Split the class attribute

crime.keras.data.trainingtarget <- crime.keras.data[index_sample==1, 20]
crime.keras.data.testtarget <- crime.keras.data[index_sample==2, 20]


# One hot encode training target values

crime.keras.data.trainLabels <- to_categorical(crime.keras.data.trainingtarget)


# One hot encode test target values

crime.keras.data.testLabels <- to_categorical(crime.keras.data.testtarget)


# Print out the testLabels to double check the result

print(crime.keras.data.testLabels)


# Initialize a sequential model
model_sgd <- keras_model_sequential()

#model_sgd %>% 
#    layer_dense(units = 80, activation = 'relu', input_shape = c(19)) %>%
#    layer_dropout(rate = 0.5) %>%
#    layer_dense(units = 60, activation = 'relu') %>%
#    layer_dropout(rate = 0.4) %>%
#    layer_dense(units = 40, activation = 'relu') %>% 
#    layer_dropout(rate = 0.3) %>% 
#    layer_dense(units = 2, activation = 'softmax')


model_sgd %>% 
    layer_dense(units = 158, activation = 'relu', input_shape = c(19)) %>%
    layer_dropout(rate = 0.5) %>%
    #layer_dense(units = 90, activation = 'relu') %>%
    #layer_dropout(rate = 0.4) %>%
    #layer_dense(units = 72, activation = 'relu') %>% 
    #layer_dropout(rate = 0.3) %>%
    layer_dense(units = 22, activation = 'relu') %>% 
    layer_dropout(rate = 0.2) %>% 
    layer_dense(units = 2, activation = 'softmax')


# Define an optimizer
sgd <- optimizer_sgd(lr = 0.01)

# Compile the model
model_sgd %>% compile(optimizer='sgd', 
                  loss='categorical_crossentropy', 
                  metrics='accuracy')

# Fit the model to the training data
history_sgd <- model_sgd %>% fit(
  crime.keras.data.training, crime.keras.data.trainLabels, 
  epochs = 40, batch_size = 100,
  #callbacks = callback_early_stopping(patience = 5, monitor = 'acc'), 
  validation_split = 0.2
 )

# Plot the history
plot(history_sgd)

# Plot the model loss
plot(history_sgd$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l")
lines(history_sgd$metrics$val_loss, col="green")
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

# Plot the model accuracy
plot(history_sgd$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l")
lines(history_sgd$metrics$val_acc, col="green")
legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))


# Predict the classes for the test data
classes_sgd <- model_sgd %>% predict_classes(crime.keras.data.test, batch_size = 128)

# Confusion matrix
table(crime.keras.data.testtarget, classes_sgd)

# Evaluate on test data and labels
score_sgd <- model_sgd %>% evaluate(crime.keras.data.test, crime.keras.data.testLabels, batch_size = 128)

# Print the score
print(score_sgd)
