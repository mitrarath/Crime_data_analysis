

qplot(data = crime, crime.type, xlab = "Crimes", main ="Crimes in Chicago", fill = crime.type) + scale_y_continuous("Number of crimes")

### By year
ggplot(data = crime) + geom_freqpoly(aes(x = Year, color = crime.type), binwidth = 1)

##Crime type count by Time window which we created above###
crime <- crime[complete.cases(crime),]
qplot(data = crime, TimeWindow, xlab="Time Window", main = "Crimes by Time Window", fill = TimeWindow) + scale_y_continuous("Number of crimes")

## Which day of the week has most of the crimes happening ###
crime$Weekday <- factor(crime$Weekday, levels= c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"))
qplot(data = crime, Weekday, xlab= "Day of week", main= "Crimes by day of week", fill = Weekday) + scale_y_continuous("Number of crimes")

## Crime by month and year and see how it varies.###
crime$MonthName <- factor(crime$MonthName, levels= c("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))
qplot(data = crime, MonthName, xlab= "Month", main= "Crimes by month", fill = MonthName) + scale_y_continuous("Number of crimes") ## Crime by Month

## Serious Crimes by Hour
ggplot(crime, aes(x=Hour, fill=serious)) + geom_bar()

## Serious Crimes by Season
ggplot(crime, aes(x=season, fill=serious)) + geom_bar()

## A simple heat map####

temp <- aggregate(crime$crime.type, by= list(crime$crime.type, crime$TimeWindow), FUN= length)
names(temp) <- c("crime.type", "TimeWindow", "count")
ggplot(temp, aes(x= crime.type, y= factor(TimeWindow))) +
  geom_tile(aes(fill= count)) +
  scale_x_discrete("Crime", expand = c(0,0)) +
  scale_y_discrete("Time of day", expand = c(0,-2)) +
  scale_fill_gradient("Number of crimes", low = "white", high = "blue") +
  theme_bw() + ggtitle("Crimes by time of day") +
  theme(panel.grid.major = element_line(colour = NA), panel.grid.minor = element_line(colour = NA))

## Plot the crimes heat map on Chicago Map. #####

chicago_map <- get_map(location = 'chicago', zoom = 11)
CrimeLocations <- as.data.frame(table(crime$Lon, crime$Lat))
names(CrimeLocations) <- c('long', 'lat', 'Frequency')
CrimeLocations$long <- as.numeric(as.character(CrimeLocations$long))
CrimeLocations$lat <- as.numeric(as.character(CrimeLocations$lat))
CrimeLocations <- subset(CrimeLocations, Frequency > 0)

ggmap(chicago_map) + geom_tile(data = CrimeLocations, aes(x = long, y = lat, alpha = Frequency),
                           fill = 'red') + theme(axis.title.y = element_blank(), axis.title.x = element_blank())
                           

