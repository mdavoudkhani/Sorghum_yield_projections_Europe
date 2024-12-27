#=================================================================#
#                                                                 #
#   R SCRIPT USED TO TRAIN MODELS TO PREDICT                      #
#   SOYBEAN YIELD FROM CLIMATE INPUTS AND                         #
#   ASSESS THEIR PREDICTIVE ABILITY THROUGH RESIDUAL ANALYSIS     #
#   AND TRANFERABILITY ASSESSMENT IN SPACE AND TIME               #
#                                                                 #
#=================================================================#
#
# AUTHOR      : Mohsen Davoudkhani (mohsen.davoudkhani@inrae.fr)
# LAST UPDATE : 25 December 2024 
# DESCRITPION : In this script, the Random Forest algorithm is used to model sorghum yield
#               from climate inputs. The model is then
#               analyzed in details, including :(i) an in-depth analysis of
#               model residuals, (ii) transferability in time, and 
#               (iii) transferability in space. 
#
# set working directory
# note: the script was initially stored in a folder named "Rscripts" 
# and the dataframe containing soybean yield and climate data was 
# initially stored in a folder named "outputs" that is 
# located is the same folder than the "Rscripts" folder
setwd("..")

# set seed
set.seed(32)

# packages
library(zoo)
library(ncdf4)
library(raster)
library(terra)
library(tidyverse)
library(ranger)
library(sp)
library(rworldmap)
library(caret) 
require(dplyr)
library(data.table)
library(plantecophys)
library(hash)
library(data.table)
library(ggplot2)
library(readxl)
library(sf)
library(viridis)
library(tictoc)
library(exactextractr)
library(foreach)
library(doParallel)

setwd("")

# read data
load("D:/sorghum_modeling/output/sorghum_yield_climate_data.Rdata")


# PART I ------------------------------------------------------------------
# Fit and save the Random Forest (RF) model over the whole dataset
# -------------------------------------------------------------------------


rf_sorghum_all  <- ranger(yield~.,data=sorghum_yield_climate_data[,7:47],num.tree=500,importance="impurity")
save(rf_sorghum_all,file="D:/sorghum_modeling/output/rf_sorghum_all.RData")

# PART II -----------------------------------------------------------------
# The performance of RF model is assessed using 5 fold cross-validation
# -------------------------------------------------------------------------

# define summary function

my.summaryFunction <- function(data, lev = NULL, model = NULL){
  EFF <- 1 - (sum((data$obs-data$pred)^2)/sum((data$obs-mean(data$obs))^2))
  res <- c(defaultSummary(data),EFF)
  names(res) <- c("RMSE","R2","MAE","EFF")
  return(res)
}

nCores <- detectCores() - 1
cl <- parallel::makeCluster(nCores, type = "SOCK")
on.exit(expr = parallel::stopCluster(cl), add = TRUE, after = FALSE)

doParallel::registerDoParallel(cl)

# apply to the RF (takes ~ 10 min)
RF_model<- train(yield ~ .,
                           data = sorghum_yield_climate_data [,7:47],
                                  method = "ranger",
                                  tuneGrid=expand.grid( 
                                                       splitrule = "variance"),
                                  importance ='impurity',
                                  trControl = trainControl(method="repeatedcv", 
                                                           number=5,repeats=3,
                                                           verboseIter  = TRUE,
                                                           summaryFunction = my.summaryFunction,
                                                           savePredictions = "final"))


# make plot of observed vs predicted for the RF model

predicted_value <- RF_model$finalModel$predictions
observed_value <- sorghum_yield_climate_data$yield

df_plot <- data.frame(predicted_value,observed_value)


ggplot(df_plot, mapping = aes(x=predicted_value, y=observed_value) ) +
  xlab("predicted") + ylab("observed")+
  ggtitle("Yield (tonnes/ha)")+theme(text = element_text(size = 24), axis.text = element_text(size = 24))+
  geom_bin2d(bins = 125) +
  scale_fill_continuous(type = "viridis",limits = c(0, 100)
                        ,option = "H") +
  xlim(-1, 10) +
  ylim(-1, 10) +
  geom_abline(slope=1)+
  theme_bw()
# PART III ----------------------------------------------------------------
# Analysis of RF model residuals
# -------------------------------------------------------------------------

# calculate residuals
df_plot$res <- df_plot$observed_value-df_plot$predicted_value


# plot histogram of residuals
hist(df_plot$res,main="",breaks=40,col="lightgreen",
     xlab=expression(paste("residuals (t ",ha^-1,")")),las=1,freq=FALSE,
     xlim=c(-3,3))
box()

# plot residuals as a function of yield

ggplot(df_plot, mapping = aes(x=observed_value, y=res) ) +
  xlab("observed yield") + ylab("residuals")+
  ggtitle("Residuals as a function of observed yield")+theme(text = element_text(size = 24), axis.text = element_text(size = 24))+
  geom_bin2d(bins = 125) +
  scale_fill_continuous(type = "viridis") +
  # xlim(0, 12) +
  # ylim(0, 12) +
  # geom_abline(slope=1)+
  theme_bw()


# PART III ----------------------------------------------------------------
# Assessing RF model transferability in time
# we first excluded yield data for one year from the dataset. 
# Then, the random forest was trained on the remaining yield 
# data to predict yields for the year excluded. 
# -------------------------------------------------------------------------


# define datasets for storing results
temporal_cross_validation<-data.frame()
observed_predicted_temporal_cross_validation<-data.frame()

for(i in 2000:2020) {
  # define train and test datasets for each year
  data.train <- subset(sorghum_yield_climate_data,
                       subset=!year%in%c(i)) 
  data.test <- subset(sorghum_yield_climate_data,
                      subset=year%in%c(i))

  # fit RF model
  rf<- ranger(yield~.,data=data.train[,7:47],
              num.tree=1000,min.node.size =6,mtry = 6,
              splitrule = "maxstat",alpha=0.09 )
  

  # make predictions on test dataset
  
  data.test$yield.pred.rf  <- predict(rf,data=data.test)$predictions
  # compute R2 between observed and predicted values on test dataset
  mod <- lm(yield~yield.pred.rf,data=data.test)
  mod_summary<-summary(mod)

  # compute  R2
  R2_test<-mod_summary$r.squared
  RMSE_test<-mltools::rmse( data.test$yield.pred.rf,data.test$yield)
  
  observed_mean_yield<- mean(data.test$yield)
  # compute  NRMSE
  NorRMSE_test=RMSE_test/observed_mean_yield*100
  # compute  Nash–Sutcliffe model efficiency (NSE)
  NSE<-vnse(data.test$yield.pred.rf,data.test$yield, na.rm = FALSE)
  
  
  # store the results
  parameter<-data.frame(test_year=i,
                        Number_of_training_data=number_train_data,
                        R2_train=R2_train,
                        RMSE_train=RMSE_train,
                        Number_of_testing_data=number_test_data,
                        R2_test=R2_test,
                        RMSE_test=RMSE_test,
                        NorRMSE_test=NorRMSE_test,
                        model_efficiency=NSE
  )
  
  parameter_ob_vs_pred<-data.frame(observed_yield=data.test$yield,
                                   predicted_yield=data.test$yield.pred.rf)
  
  parameter_ob_vs_pred$year_excluded<-i
  
  observed_predicted_temporal_cross_validation<-rbind(observed_predicted_temporal_cross_validation,
                                                      parameter_ob_vs_pred)
  
  temporal_cross_validation<-rbind(temporal_cross_validation,parameter)
  
  
  
}


# PART IV -----------------------------------------------------------------
# ASSESSING TRANSFERABILITY IN SPACE
# To assess the transferability in space, we followed a five-step procedure. 
# First, we randomly selected a department (county) in one of 
# the European countries included in our dataset (France, Italy, or Spain). 
# Then, six buffer zones of varying sizes (100, 200, 400, 600, 800, and 1,000 km)
# were defined around the barycenter of the selected department. 
# For each buffer zone, neighboring departments within the buffer zone 
# were removed to train the random forest model on the rest of the dataset outside 
# the buffer zone. Next, we predicted yields for the randomly selected department 
# and we computed R², NSE, and NRMSE between observed and predicted yields for
# the selected department. We repeated this procedure over 10 departments 
# in each European country (France, Italy, and Spain) 
# -------------------------------------------------------------------------

# define dataframes for saving results of predictions for different buffer zones
spatial_cross_validation_buffer_zone_50<-data.frame()
observed_predicted_spatial_cross_validation_buffer_zone_50<-data.frame()
spatial_cross_validation_buffer_zone_100<-data.frame()
observed_predicted_spatial_cross_validation_buffer_zone_100<-data.frame()
spatial_cross_validation_buffer_zone_200<-data.frame()
observed_predicted_spatial_cross_validation_buffer_zone_200<-data.frame()
spatial_cross_validation_buffer_zone_400<-data.frame()
observed_predicted_spatial_cross_validation_buffer_zone_400<-data.frame()
spatial_cross_validation_buffer_zone_600<-data.frame()
observed_predicted_spatial_cross_validation_buffer_zone_600<-data.frame()
spatial_cross_validation_buffer_zone_800<-data.frame()
observed_predicted_spatial_cross_validation_buffer_zone_800<-data.frame()
spatial_cross_validation_buffer_zone_1000<-data.frame()
observed_predicted_spatial_cross_validation_buffer_zone_1000<-data.frame()
# rbind departments of all countries
Europe_department_shapefiles<-rbind(french_departments,
                                    Italy_department,
                                    spain_department)

# Calculate the centroids of department polygons
centroids <- st_centroid(Europe_department_shapefiles)

# Extract the longitude and latitude coordinates of the centroids
centroids_lon_lat <- st_coordinates(centroids)

# Create a dataframe with department names and their corresponding centroids
Europe_department_centroids <- data.frame(
  County = Europe_department_shapefiles$County,
  Longitude = centroids_lon_lat[, "X"],
  Latitude = centroids_lon_lat[, "Y"]
)

# select three European countries
Europe_yield_data<- sorghum_yield_climate_data[sorghum_yield_climate_data$State %in% c("France", "Italy", "spain"), ]

# merge yield data of the three European countries and lat and long data
Europe_yield_data<-full_join(Europe_yield_data,Europe_department_centroids,
                             by=c("County"))

Europe_yield_data<-na.omit(Europe_yield_data)
# Get unique county names
Europe_unique_counties  <- unique(Europe_yield_data_unique_counties$County)
# Randomly select 10 unique counties without repetition
random_counties <- sample(Europe_unique_counties, 10, replace = FALSE)
# Subtract USA data and zero yield data from the whole dataset
USA_zero_yield<-anti_join(sorghum_yield_climate_data,
                          Europe_yield_data)
USA_zero_yield$Latitude<-"NI"
USA_zero_yield$Longitude<-"NI"

# transform data into spatial data
sdata <- SpatialPointsDataFrame(coords = subset(Europe_yield_data,select=c(Longitude,Latitude)),
                                data=Europe_yield_data,
                                proj4string=CRS("+proj=longlat +datum=WGS84"))




for (random_counties in Europe_yield_data_unique_counties$County){
  # select starting point
  # we don't want a desert/(ant)artic area as a starting point
  start <- subset(sdata,County==county_name)
  bzone.50  <- circles(start,d=50000)
  bzone.100  <- circles(start,d=100000)
  bzone.200  <- circles(start,d=200000)
  bzone.400  <- circles(start,d=400000)
  bzone.600  <- circles(start,d=600000)
  bzone.800  <- circles(start,d=800000)
  bzone.1000  <- circles(start,d=1000000)

  
  # create training datasets :
  # keep only gridcodes outside the buffer zones
  df.50  <- sdata@data[is.na(over(sdata,bzone.50@polygons)),]
  df.100  <- sdata@data[is.na(over(sdata,bzone.100@polygons)),]
  df.200  <- sdata@data[is.na(over(sdata,bzone.200@polygons)),]
  df.400  <- sdata@data[is.na(over(sdata,bzone.400@polygons)),]
  df.600  <- sdata@data[is.na(over(sdata,bzone.600@polygons)),]
  df.800  <- sdata@data[is.na(over(sdata,bzone.800@polygons)),]
  df.1000 <- sdata@data[is.na(over(sdata,bzone.1000@polygons)),]
  
  # bind data out side of buffer zones in Europe with the USA and zero yield data
  df.50<-rbind(df.50,USA_zero_yield)
  df.100<-rbind(df.100,USA_zero_yield)
  df.200<-rbind(df.200,USA_zero_yield)
  df.400<-rbind(df.400,USA_zero_yield)
  df.600<-rbind(df.600,USA_zero_yield)
  df.800<-rbind(df.800,USA_zero_yield)
  df.1000<-rbind(df.1000,USA_zero_yield)
  
  # test data-data inside the buffer zones
  test_data.50<-anti_join(sorghum_yield_climate_data,
                          df.50)
  number_test_data.50<-nrow(test_data.50)
  
  test_data.100<-anti_join(sorghum_yield_climate_data,
                           df.100)
  number_test_data.100<-nrow(test_data.100)
  
  test_data.200<-anti_join(sorghum_yield_climate_data,
                           df.200)
  number_test_data.200<-nrow(test_data.200)
  
  test_data.400<-anti_join(sorghum_yield_climate_data,
                           df.400)
  number_test_data.400<-nrow(test_data.400)
  
  test_data.600<-anti_join(sorghum_yield_climate_data,
                           df.600)
  number_test_data.600<-nrow(test_data.600)
  
  test_data.800<-anti_join(sorghum_yield_climate_data,
                           df.800)
  number_test_data.800<-nrow(test_data.800)
  
  test_data.1000<-anti_join(sorghum_yield_climate_data,
                            df.1000)
  number_test_data.1000<-nrow(test_data.1000)
  
  
  # fit models
  print("model fitting")
  mod.rf.50  <- ranger(yield~.,data=df.50[,7:47],
                       num.tree=1000,min.node.size =6,mtry = 6,
                       splitrule = "variance")
  mod.rf.100  <- ranger(yield~.,data=df.100[,7:47],
                        num.tree=1000,min.node.size =6,mtry = 6,
                        splitrule = "variance")
  mod.rf.200  <- ranger(yield~.,data=df.200[,7:47],
                        num.tree=1000,min.node.size =6,mtry = 6,
                        splitrule = "variance")
  mod.rf.400  <- ranger(yield~.,data=df.400[,7:47],
                        num.tree=1000,min.node.size =6,mtry = 6,
                        splitrule = "variance")
  mod.rf.600  <- ranger(yield~.,data=df.600[,7:47],
                        num.tree=1000,min.node.size =6,mtry = 6,
                        splitrule = "variance")
  mod.rf.800  <- ranger(yield~.,data=df.800[,7:47],
                        num.tree=1000,min.node.size =6,mtry = 6,
                        splitrule = "variance")
  mod.rf.1000  <- ranger(yield~.,data=df.1000[,7:47],
                         num.tree=1000,min.node.size =6,mtry = 6,
                         splitrule = "variance")
  
  # make predictions
  ## 50  buffer zone
  # Predict the yield for test data in the 50 km buffer zone using the Random Forest model trained on data outside the buffer zone.
  
  test_data.50$yield.pred.rf  <- predict(mod.rf.50,data=test_data.50)$predictions
  # compute R2 between observed and predicted values on test dataset
  mod <- lm(yield~yield.pred.rf,data=test_data.50)
  mod_summary<-summary(mod)
  R2_test<-mod_summary$r.squared

  # Calculate the Root Mean Squared Error (RMSE) for the predicted yields in the 50 km buffer zone.
  
  RMSE_test<-mltools::rmse( test_data.50$yield.pred.rf,test_data.50$yield)
  # Calculate the mean observed yield for the test data in the 50 km buffer zone.
  
  observed_mean_yield<- mean(test_data.50$yield)
  # Compute the Normalized RMSE (NRMSE) as a percentage of the mean observed yield.
  
  NorRMSE_test=RMSE_test/observed_mean_yield*100
  # Compute the Nash-Sutcliffe Efficiency (NSE) to evaluate the predictive accuracy of the model.
  
  NSE<-vnse(test_data.50$yield.pred.rf,test_data.50$yield, na.rm = FALSE)
  
  
  # Create a dataframe to store evaluation metrics for the 50 km buffer zone.
  # Includes the department name, buffer zone size, number of testing data points,
  # R-squared, RMSE, Normalized RMSE, and Nash-Sutcliffe Efficiency (NSE).
  parameter<-data.frame(department=county_name,
                        Buffer_zone<-50,
                        Number_of_testing_data=number_test_data.50,
                        R2_test=R2_test,
                        RMSE_test=RMSE_test,
                        NorRMSE_test=NorRMSE_test,
                        model_efficiency=NSE
  )
  
  # Create a dataframe to store observed and predicted yield values for the 50 km buffer zone.
  
  parameter_ob_vs_pred_buffer_zone<-data.frame(observed_yield=test_data.50$yield,
                                               predicted_yield=test_data.50$yield.pred.rf)
  
  # Add a column to indicate the buffer zone size (50 km) in the observed vs. predicted dataframe.
  
  parameter_ob_vs_pred_buffer_zone$Buffer_zone<-50
  # Add a column to indicate the department name in the observed vs. predicted dataframe.
  
  parameter_ob_vs_pred_buffer_zone$department<-county_name
  
  # Append the observed vs. predicted data for the 50 km buffer zone to the main dataframe for all buffer zones.
  
  observed_predicted_spatial_cross_validation_buffer_zone_50<-rbind(observed_predicted_spatial_cross_validation_buffer_zone_50,
                                                                    parameter_ob_vs_pred_buffer_zone)
  
  # Append the evaluation metrics for the 50 km buffer zone to the main dataframe for all buffer zones.
  
  spatial_cross_validation_buffer_zone_50<-rbind(spatial_cross_validation_buffer_zone_50,parameter)
  
  
  # Predict the yield for test data in the 100 km buffer zone using the Random Forest model trained on data outside the buffer zone.
  test_data.100$yield.pred.rf  <- predict(mod.rf.100, data=test_data.100)$predictions
  
  # Compute R-squared value to assess the correlation between observed and predicted yields for the 100 km buffer zone.
  mod <- lm(yield ~ yield.pred.rf, data=test_data.100)
  mod_summary <- summary(mod)
  R2_test <- mod_summary$r.squared
  
  # Calculate Root Mean Squared Error (RMSE) for the predictions in the 100 km buffer zone.
  RMSE_test <- mltools::rmse(test_data.100$yield.pred.rf, test_data.100$yield)
  
  # Compute the mean observed yield for the 100 km buffer zone.
  observed_mean_yield <- mean(test_data.100$yield)
  
  # Calculate the Normalized RMSE (NRMSE) as a percentage of the mean observed yield.
  NorRMSE_test <- RMSE_test / observed_mean_yield * 100
  
  # Compute Nash-Sutcliffe Efficiency (NSE) to evaluate the predictive performance of the model.
  NSE <- vnse(test_data.100$yield.pred.rf, test_data.100$yield, na.rm = FALSE)
  
  # Create a dataframe to store the evaluation metrics for the 100 km buffer zone.
  parameter <- data.frame(department=county_name,
                          Buffer_zone=100,
                          Number_of_testing_data=number_test_data.100,
                          R2_test=R2_test,
                          RMSE_test=RMSE_test,
                          NorRMSE_test=NorRMSE_test,
                          model_efficiency=NSE)
  
  # Create a dataframe to store observed and predicted yield values for the 100 km buffer zone.
  parameter_ob_vs_pred_buffer_zone <- data.frame(observed_yield=test_data.100$yield,
                                                 predicted_yield=test_data.100$yield.pred.rf)
  
  # Add columns for buffer zone size and department name to the observed vs. predicted dataframe.
  parameter_ob_vs_pred_buffer_zone$Buffer_zone <- 100
  parameter_ob_vs_pred_buffer_zone$department <- county_name
  
  # Append the observed vs. predicted data for the 100 km buffer zone to the main dataframe.
  observed_predicted_spatial_cross_validation_buffer_zone_100 <- rbind(observed_predicted_spatial_cross_validation_buffer_zone_100,
                                                                       parameter_ob_vs_pred_buffer_zone)
  
  # Append the evaluation metrics for the 100 km buffer zone to the main dataframe.
  spatial_cross_validation_buffer_zone_100 <- rbind(spatial_cross_validation_buffer_zone_100, parameter)
  
  # Repeat similar steps for the 200 km buffer zone:
  # Predict yield, compute metrics (R2, RMSE, NorRMSE, NSE), store results, and append to respective dataframes.
  test_data.200$yield.pred.rf  <- predict(mod.rf.200, data=test_data.200)$predictions
  mod <- lm(yield ~ yield.pred.rf, data=test_data.200)
  mod_summary <- summary(mod)
  R2_test <- mod_summary$r.squared
  RMSE_test <- mltools::rmse(test_data.200$yield.pred.rf, test_data.200$yield)
  observed_mean_yield <- mean(test_data.200$yield)
  NorRMSE_test <- RMSE_test / observed_mean_yield * 100
  NSE <- vnse(test_data.200$yield.pred.rf, test_data.200$yield, na.rm = FALSE)
  parameter <- data.frame(department=county_name,
                          Buffer_zone=200,
                          Number_of_testing_data=number_test_data.200,
                          R2_test=R2_test,
                          RMSE_test=RMSE_test,
                          NorRMSE_test=NorRMSE_test,
                          model_efficiency=NSE)
  parameter_ob_vs_pred_buffer_zone <- data.frame(observed_yield=test_data.200$yield,
                                                 predicted_yield=test_data.200$yield.pred.rf)
  parameter_ob_vs_pred_buffer_zone$Buffer_zone <- 200
  parameter_ob_vs_pred_buffer_zone$department <- county_name
  observed_predicted_spatial_cross_validation_buffer_zone_200 <- rbind(observed_predicted_spatial_cross_validation_buffer_zone_200,
                                                                       parameter_ob_vs_pred_buffer_zone)
  spatial_cross_validation_buffer_zone_200 <- rbind(spatial_cross_validation_buffer_zone_200, parameter)
  
  # Repeat the same steps for 400 km, 600 km, 800 km, and 1000 km buffer zones as needed.
  # Add predictions, compute metrics, store results, and append to corresponding dataframes.
  
  # Predict the yield for test data in the 400 km buffer zone using the Random Forest model trained on data outside the buffer zone.
  test_data.400$yield.pred.rf  <- predict(mod.rf.400,data=test_data.400)$predictions
  # compute R2 between observed and predicted values on test dataset
  mod <- lm(yield~yield.pred.rf,data=test_data.400)
  mod_summary<-summary(mod)
  R2_test<-mod_summary$r.squared
  RMSE_test<-mltools::rmse( test_data.400$yield.pred.rf,test_data.400$yield)
  
  observed_mean_yield<- mean(test_data.400$yield)
  
  NorRMSE_test=RMSE_test/observed_mean_yield*100
  
  NSE<-vnse(test_data.400$yield.pred.rf,test_data.400$yield, na.rm = FALSE)
  
  
  
  parameter<-data.frame(department=county_name,
                        Buffer_zone<-400,
                        Number_of_testing_data=number_test_data.400,
                        R2_test=R2_test,
                        RMSE_test=RMSE_test,
                        NorRMSE_test=NorRMSE_test,
                        model_efficiency=NSE
  )
  
  parameter_ob_vs_pred_buffer_zone<-data.frame(observed_yield=test_data.400$yield,
                                               predicted_yield=test_data.400$yield.pred.rf)
  
  parameter_ob_vs_pred_buffer_zone$Buffer_zone<-400
  parameter_ob_vs_pred_buffer_zone$department<-county_name
  
  
  observed_predicted_spatial_cross_validation_buffer_zone_400<-rbind(observed_predicted_spatial_cross_validation_buffer_zone_400,
                                                                     parameter_ob_vs_pred_buffer_zone)
  
  # Append the evaluation metrics for the 400 km buffer zone to the main dataframe for all buffer zones.
  
  spatial_cross_validation_buffer_zone_400<-rbind(spatial_cross_validation_buffer_zone_400,parameter)
  
  # Predict the yield for test data in the 600 km buffer zone using the Random Forest model trained on data outside the buffer zone.
  test_data.600$yield.pred.rf  <- predict(mod.rf.600,data=test_data.600)$predictions
  # compute R2 between observed and predicted values on test dataset
  mod <- lm(yield~yield.pred.rf,data=test_data.600)
  mod_summary<-summary(mod)
  R2_test<-mod_summary$r.squared
  data.test<-na.omit(test_data.600)
  RMSE_test<-mltools::rmse( test_data.600$yield.pred.rf,test_data.600$yield)
  
  observed_mean_yield<- mean(test_data.600$yield)
  
  NorRMSE_test=RMSE_test/observed_mean_yield*100
  
  NSE<-vnse(test_data.600$yield.pred.rf,test_data.600$yield, na.rm = FALSE)
  
  
  
  parameter<-data.frame(department=county_name,
                        Buffer_zone<-600,
                        Number_of_testing_data=number_test_data.600,
                        R2_test=R2_test,
                        RMSE_test=RMSE_test,
                        NorRMSE_test=NorRMSE_test,
                        model_efficiency=NSE
  )
  
  parameter_ob_vs_pred_buffer_zone<-data.frame(observed_yield=test_data.600$yield,
                                               predicted_yield=test_data.600$yield.pred.rf)
  
  parameter_ob_vs_pred_buffer_zone$Buffer_zone<-600
  parameter_ob_vs_pred_buffer_zone$department<-county_name
  
  
  observed_predicted_spatial_cross_validation_buffer_zone_600<-rbind(observed_predicted_spatial_cross_validation_buffer_zone_600,
                                                                     parameter_ob_vs_pred_buffer_zone)
  
  # Append the evaluation metrics for the 600 km buffer zone to the main dataframe for all buffer zones.
  
  spatial_cross_validation_buffer_zone_600<-rbind(spatial_cross_validation_buffer_zone_600,parameter)
  
  
  # Predict the yield for test data in the 800 km buffer zone using the Random Forest model trained on data outside the buffer zone.
  test_data.800$yield.pred.rf  <- predict(mod.rf.800,data=test_data.800)$predictions
  # compute R2 between observed and predicted values on test dataset
  mod <- lm(yield~yield.pred.rf,data=test_data.800)
  mod_summary<-summary(mod)
  R2_test<-mod_summary$r.squared
  data.test<-na.omit(test_data.800)
  RMSE_test<-mltools::rmse( test_data.800$yield.pred.rf,test_data.800$yield)
  
  observed_mean_yield<- mean(test_data.800$yield)
  
  NorRMSE_test=RMSE_test/observed_mean_yield*100
  
  NSE<-vnse(test_data.800$yield.pred.rf,test_data.800$yield, na.rm = FALSE)
  
  
  
  parameter<-data.frame(department=county_name,
                        Buffer_zone<-800,
                        Number_of_testing_data=number_test_data.800,
                        R2_test=R2_test,
                        RMSE_test=RMSE_test,
                        NorRMSE_test=NorRMSE_test,
                        model_efficiency=NSE
  )
  
  parameter_ob_vs_pred_buffer_zone<-data.frame(observed_yield=test_data.800$yield,
                                               predicted_yield=test_data.800$yield.pred.rf)
  
  parameter_ob_vs_pred_buffer_zone$Buffer_zone<-800
  parameter_ob_vs_pred_buffer_zone$department<-county_name
  
  
  observed_predicted_spatial_cross_validation_buffer_zone_800<-rbind(observed_predicted_spatial_cross_validation_buffer_zone_800,
                                                                     parameter_ob_vs_pred_buffer_zone)
  
  # Append the evaluation metrics for the 800 km buffer zone to the main dataframe for all buffer zones.
  
  spatial_cross_validation_buffer_zone_800<-rbind(spatial_cross_validation_buffer_zone_800,parameter)
  
  # Predict the yield for test data in the 1000 km buffer zone using the Random Forest model trained on data outside the buffer zone.
  test_data.1000$yield.pred.rf  <- predict(mod.rf.1000,data=test_data.1000)$predictions
  # compute R2 between observed and predicted values on test dataset
  mod <- lm(yield~yield.pred.rf,data=test_data.1000)
  mod_summary<-summary(mod)
  R2_test<-mod_summary$r.squared
  data.test<-na.omit(test_data.1000)
  RMSE_test<-mltools::rmse( test_data.1000$yield.pred.rf,test_data.1000$yield)
  
  observed_mean_yield<- mean(test_data.1000$yield)
  
  NorRMSE_test=RMSE_test/observed_mean_yield*100
  
  NSE<-vnse(test_data.1000$yield.pred.rf,test_data.1000$yield, na.rm = FALSE)
  
  
  
  parameter<-data.frame(department=county_name,
                        Buffer_zone<-1000,
                        Number_of_testing_data=number_test_data.1000,
                        R2_test=R2_test,
                        RMSE_test=RMSE_test,
                        NorRMSE_test=NorRMSE_test,
                        model_efficiency=NSE
  )
  
  parameter_ob_vs_pred_buffer_zone<-data.frame(observed_yield=test_data.1000$yield,
                                               predicted_yield=test_data.1000$yield.pred.rf)
  
  parameter_ob_vs_pred_buffer_zone$Buffer_zone<-1000
  parameter_ob_vs_pred_buffer_zone$department<-county_name
  
  
  observed_predicted_spatial_cross_validation_buffer_zone_1000<-rbind(observed_predicted_spatial_cross_validation_buffer_zone_1000,
                                                                      parameter_ob_vs_pred_buffer_zone)
  
  # Append the evaluation metrics for the 1000 km buffer zone to the main dataframe for all buffer zones.
  
  spatial_cross_validation_buffer_zone_1000<-rbind(spatial_cross_validation_buffer_zone_1000,parameter)
  
}



