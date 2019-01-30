# Crime_data_analysis
Chicago Crime data Analysis with R and Spark

## Problem Statement
Crime is most commonly defined and understood worldwide as an “unlawful activity” and a socio-economic issue which effects everyone and every country. It’s not only a social problem but it can cause serious damage to a region’s economic strength and growth by determining whether people and businesses want to move or avoid certain countries/regions/places. But as a matter of fact, every country in this world must face this epidemic and try controlling, managing and reducing crime for the betterment of its citizens and enhancing its economic prowess.  With the increase of crimes, several law enforcement agencies in different countries are falling back to advanced data mining and machine learning techniques also known as predictive policing [1] to improve knowledge about crimes and protect their citizens. For instance “Dubai police launch Crime Prediction” or “Predictive policing substantially reduces crime in Los Angeles during months-long test” are few examples how AI and ML with Big Data is helping law enforcement.

Continuing with the above introduction, the highlighted issue is how to apply data science and machine learning methodologies to understand more about crimes at a particular region using historical dataset and provide high probability of a specific type of crime which might occur next in a specific location within a particular time frame. Building a successful high accuracy algorithm to predict where crime would be more prevalent will benefit the citizens and law enforcement authorities. With more knowledge and information about the crime which might occur in future would allow law enforcement authorities to better plan resource allocation to high crime zones. But a bad algorithm or prediction can create havoc or embarrassment to the authorities. So, to have better accuracy with predictions (algorithms) and better insights for action we need to follow the basics of data science which would include going through the different phases of a data analytics lifecycle, which also includes framing Initial Hypotheses.

But “can we actually predict crime?” is a question which might come tour mind. To answer this would like to point out one of the studies done by university of California where it was stated that “On Behavioural grounds, it has been found that once successful, criminals try to replicate the crime under similar conditions. They tend to operate in their comfort zone, and hence they look for similar locations and time for the next crime. This makes them predictable.”. In fact they compared the crime predictability with earthquake, where they said as we know that earthquakes cannot be predicted accurately, but after shocks can be. Similarly crime can be predicted by the looking at past data and criminal behaviours.

## Dataset
Here we have used datasets for real-word crimes in one of the city in US which is Chicago to construct our data mining models. The dataset is available to download from Chicago Data Portal reflects reported incidents of crime (with the exception of murders where data exists for each victim) that occurred in the City of Chicago from 2001 to 2017 (present, - 7 days). Data is extracted from the Chicago Police Department's CLEAR (Citizen Law Enforcement Analysis and Reporting) system. The dataset contains around 5.8 million rows with 22 columns where each row is a reported crime. The column definition in the dataset is as given below:
-	ID - Unique identifier for the record.
-	Case Number - The Chicago Police Department RD Number (Records Division Number), which is unique to the incident.
-	Date - Date when the incident occurred. this is sometimes a best estimate.
-	Block - The partially redacted address where the incident occurred, placing it on the same block as the actual address.
-	IUCR - The Illinois Unifrom Crime Reporting code. This is directly linked to the Primary Type and Description. See the list of IUCR codes at https://data.cityofchicago.org/d/c7ck-438e.
-	Primary Type - The primary description of the IUCR code.
-	Description - The secondary description of the IUCR code, a subcategory of the primary description.
-	Location Description - Description of the location where the incident occurred.
-	Arrest - Indicates whether an arrest was made.
-	Domestic - Indicates whether the incident was domestic-related as defined by the Illinois Domestic Violence Act.
-	Beat - Indicates the beat where the incident occurred. A beat is the smallest police geographic area – each beat has a dedicated police beat car. Three to five beats make up a police sector, and three sectors make up a police district. The Chicago Police Department has 22 police districts. See the beats at   https://data.cityofchicago.org/d/aerh-rz74.
-	District - Indicates the police district where the incident occurred. See the districts at https://data.cityofchicago.org/d/fthy-xz3r.
-	 Ward - The ward (City Council district) where the incident occurred. See the wards at https://data.cityofchicago.org/d/sp34-6z76.
-	Community Area - Indicates the community area where the incident occurred. Chicago has 77 community areas. See the community areas at https://data.cityofchicago.org/d/cauq-8yn6.
-	FBI Code - Indicates the crime classification as outlined in the FBI's National Incident-Based Reporting System (NIBRS). See the Chicago Police Department listing of these classifications at http://gis.chicagopolice.org/clearmap_crime_sums/crime_types.html.
-	X Coordinate - The x coordinate of the location where the incident occurred in State Plane Illinois East NAD 1983 projection. This location is shifted from the actual location for partial redaction but falls on the same block.
-	 Y Coordinate - The y coordinate of the location where the incident occurred in State Plane Illinois East NAD 1983 projection. This location is shifted from the actual location for partial redaction but falls on the same block.
-	Year - Year the incident occurred.
-	 Updated On - Date and time the record was last updated.
-	 Latitude - The latitude of the location where the incident occurred. This location is shifted from the actual location for partial redaction but falls on the same block.
-	Longitude - The longitude of the location where the incident occurred. This location is shifted from the actual location for partial redaction but falls on the same block.
-	Location - The location where the incident occurred in a format that allows for creation of maps and other geographic operations on this data portal. This location is shifted from the actual location for partial redaction but falls on the same block.

There are some small issues with this dataset like duplicate data specifically in the Case Number column or missing values specifically in Latitude and Longitude details.The duplicate data can be handled easily here by removing the duplicates, but missing value imputation is a critical ingredient and is very important for modelling as there would a lot of data missing and not using or deleting this data might impact our final model. So to irradicate this issue we generally, depending on the type of the missing data/variable, we substitute the values logically. But in our dataset the longitude and latitude variables represent the coordinates of the location where the crime incident occurred and it would  not be correct to substitute these values using simple mathematical logic. So, I had to actually ignore these observations and removed the missing values.
We also have other datasets which is the Chicago census data. Nothing much complicated about this data set and is also a clean data with no issues as seen in the last data. 

•	Community Area Number: Chicago has 77 community areas. See the community areas at https://data.cityofchicago.org/d/cauq-8yn6
•	COMMUNITY AREA NAME: Exact Name of the community in plain text	
•	PERCENT OF HOUSING CROWDED: Percent occupied housing units with more than one person per room
•	PERCENT HOUSEHOLDS BELOW POVERTY: Percent of households living below the federal poverty level
•	PERCENT AGED 16+ UNEMPLOYED : Percent of persons over the age of 16 years that are unemployed
•	PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA: Percent of persons over the age of 25 years without a high school education
•	PERCENT AGED UNDER 18 OR OVER 64: Percent of the population under 18 or over 64 years of age (i.e., dependency)
•	PER CAPITA INCOME: Community Area Per capita income is estimated as the sum of tract-level aggregate incomes divided by the total population
•	HARDSHIP INDEX: Score that incorporates each of the six selected socioeconomic indicators (see dataset description)
But one thing which we can observe  here by looking at the features and we could estimate that most of the features here are correlated and the question arises which feature should you use in our models and which to reject or shall we use all of them...all these questions will be answered in the feature selection section. 

