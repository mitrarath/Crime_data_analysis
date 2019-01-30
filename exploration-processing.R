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