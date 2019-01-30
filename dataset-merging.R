### ======= Integrate census data with the crime Dataset ########
census_data <- fread("data/census_data.csv")
names(census_data)[names(census_data) == "Community Area Number"] <- "Community Area"
crime$`Community Area` <- as.integer(crime$`Community Area`)
census_data$`Community Area` <- as.integer(census_data$`Community Area`)
crime.data.new <- merge(crime, census_data, by = "Community Area")
crime.data.new1 <- subset(crime.data.new, select=c("Community Area", "District", "Day", "Month", "YearNew", "Hour", "TimeWindowInt", "WeekdayInt", "Lon", "Lat", "Ward", "season", "PERCENT OF HOUSING CROWDED", "PERCENT HOUSEHOLDS BELOW POVERTY", "PERCENT AGED 16+ UNEMPLOYED", "PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA", "PERCENT AGED UNDER 18 OR OVER 64", "PER CAPITA INCOME", "HARDSHIP INDEX", "CrimeTypeInt", "serious"))
setnames(crime.data.new1, old = c('Community Area', 'District', 'Day', 'Month', 'YearNew', 'Hour', 'TimeWindowInt', 'WeekdayInt', 'Lon', 'Lat', 'Ward', 'season', 'PERCENT OF HOUSING CROWDED', 'PERCENT HOUSEHOLDS BELOW POVERTY', 'PERCENT AGED 16+ UNEMPLOYED', 'PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA', 'PERCENT AGED UNDER 18 OR OVER 64', 'PER CAPITA INCOME', 'HARDSHIP INDEX', 'CrimeTypeInt', 'serious'), new = c('Area', 'District', 'Day', 'Month', 'Year', 'Hour', 'TimeWindowInt', 'WeekdayInt', 'Lon', 'Lat', 'Ward', 'season', 'HouseCrowded', 'BelowPoverty', 'Unemployed', 'LowerEducation', 'YoungAndOld', 'PCI', 'HI', 'CrimeTypeInt', 'serious'))
#setnames(crime.data.new1, old = c('Community Area', 'District', 'Day', 'Month', 'YearNew', 'Hour', 'TimeWindowInt', 'WeekdayInt', 'Lon', 'Lat', 'District', 'Ward', 'season', 'PERCENT OF HOUSING CROWDED', 'PERCENT HOUSEHOLDS BELOW POVERTY', 'PERCENT AGED 16+ UNEMPLOYED', 'PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA', 'PERCENT AGED UNDER 18 OR OVER 64', 'PER CAPITA INCOME', 'HARDSHIP INDEX', 'CrimeTypeInt', 'serious'), new = c('Area', 'Day', 'Month', 'Year', 'Hour', 'TimeWindowInt', 'WeekdayInt', 'Lon', 'Lat', 'District', 'Ward', 'season' 'HouseCrowded', 'BelowPoverty', 'Unemployed', 'LowerEducation', 'YoungAndOld', 'PCI', 'HI', 'CrimeTypeInt', 'serious'))
crime.data.new1$seriousInt <- ifelse(as.character(crime.data.new1$serious) == "TRUE", 1, 0)
str(crime.data.new1)

crime.data.new.reg <- subset(crime.data.new1, select=c("Area", "District", "Day", "Month", "Year", "Hour", "TimeWindowInt", "WeekdayInt", "Lon", "Lat", "Ward", "season", "HouseCrowded", "BelowPoverty", "Unemployed", "LowerEducation", "YoungAndOld", "PCI", "HI", "CrimeTypeInt", "serious"))
crime.data.new.reg$serious <- ifelse(as.character(crime.data.new.reg$serious) == "TRUE", 1, 0)

crime.data.new.reg1 <- crime.data.new.reg[sample(nrow(crime.data.new.reg), 100000), ]
