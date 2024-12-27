#=================================================================#
#                                                                 #
#   RSCRIPT USED TO MAKE SORGHUM YIELD PROJECTIONS                #
#   IN EUROPE UNDER CURRENT AND FUTURE CLIMATE SCENARIOS          #
#                                                                 #
#=================================================================#

# AUTHOR      : Mohsen Davoudkhani (mohsen.davoudkhani@inrae.fr)
# LAST UPDATE : 25 December 2024 
# DESCRIPTION : This script uses the Random Forest model fitted
#               in a previous R script to make 
#               projections of sorghum yield in Europe under current
#               and future climate scenarios. Projections outputs are
#               saved as netCDF files (one file by year and climate model).

# Set working directory and random seed for reproducibility
setwd("..")
set.seed(32)

# Load necessary libraries for data processing, modeling, and geospatial analysis
library(plyr)
library(tidyverse)
library(spatialrisk)
library(sf) 
library(zoo)
library(ncdf4)
library(raster)
library(terra)
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
library(viridis)
library(tictoc)
library(exactextractr)
library(foreach)
library(doParallel)

# Load the pre-trained Random Forest model for sorghum yield predictions
load("D:/sorghum_modeling/output/rf_sorghum_all.RData")

# Define time periods for projections
time_period <- c("2041_2060", "2081_2100")

# Define climate models and scenarios
climate_model <- c("UKESM1-0-LL", "MRI-ESM2-0", "MPI-ESM1-2-HR", "IPSL-CM6A-LR", "GFDL-ESM4")
Climate_scenario <- c("ssp126", "ssp370", "ssp585")

# Create all combinations of climate models, time periods, and scenarios
isimip_total_combination <- as.data.frame(crossing(climate_model, time_period, Climate_scenario))

# Map original climate model names to alternative names for filenames
isimip_total_combination <- isimip_total_combination %>%
  mutate(climate_model_2 = case_when(
    climate_model == "MRI-ESM2-0" ~ "mri-esm2-0",
    climate_model == "UKESM1-0-LL" ~ "ukesm1-0-ll",
    climate_model == "MPI-ESM1-2-HR" ~ "mpi-esm1-2-hr",
    climate_model == "IPSL-CM6A-LR" ~ "ipsl-cm6a-lr",
    climate_model == "GFDL-ESM4" ~ "GFDL_ESM4",
    TRUE ~ NA_character_  # Default value if no match is found
  ))

# Add columns for starting, ending, mid-period, and mid-period plus one year
isimip_total_combination <- isimip_total_combination %>%
  mutate(starting_year = case_when(
    time_period == "2041_2060" ~ "2041",
    time_period == "2081_2100" ~ "2081",
    TRUE ~ NA_character_
  ),
  ending_year = case_when(
    time_period == "2041_2060" ~ "2060",
    time_period == "2081_2100" ~ "2100",
    TRUE ~ NA_character_
  ),
  mid_period = case_when(
    time_period == "2041_2060" ~ "2050",
    time_period == "2081_2100" ~ "2090",
    TRUE ~ NA_character_
  ),
  mid_period_plus = case_when(
    time_period == "2041_2060" ~ "2051",
    time_period == "2081_2100" ~ "2091",
    TRUE ~ NA_character_
  )) %>%
  rowwise()  # Ensure operations are applied row-wise

# Set the number of cores for parallel processing
num_cores <- 3  # Adjust based on system capacity

# Initialize parallel backend for distributed computing
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Loop through all combinations of climate models, scenarios, and time periods
Future_projections <- foreach(n = 1:nrow(isimip_total_combination), .combine = 'rbind',
                              .packages = c('raster', "data.table", "terra",
                                            "hash", "exactextractr", "dplyr",
                                            "purrr", "zoo", "plyr", "ranger", "caret",
                                            "sf", "tidyverse", "ncdf4", "viridis", "ggplot2",
                                           )) %dopar% {
# Extract details for the current combination
climate_model <- isimip_total_combination$climate_model[[n]]
time_period <- isimip_total_combination$time_period[[n]]
Climate_scenario <- isimip_total_combination$Climate_scenario[[n]]
climate_model_2 <- isimip_total_combination$climate_model_2[[n]]
starting_year <- isimip_total_combination$starting_year[[n]]
ending_year <- isimip_total_combination$ending_year[[n]]
mid_period <- isimip_total_combination$mid_period[[n]]
mid_period_plus <- isimip_total_combination$mid_period_plus[[n]]

# Define filenames for input climate data
filenames_1 <- paste(climate_model, "_", Climate_scenario, "/monthly/", climate_model_2, "_",
                     Climate_scenario, "_", c("hurs", "pr", "rsds", "tasmax", "tasmin"), "_", 
                     starting_year, "_", mid_period, "_monthly.nc", sep = "")
filenames_2 <- paste(climate_model, "_", Climate_scenario, "/monthly/", climate_model_2, "_",
                     Climate_scenario, "_", c("hurs", "pr", "rsds", "tasmax", "tasmin"), "_", 
                     mid_period_plus, "_", ending_year, "_monthly.nc", sep = "")
filenames_total <- c(filenames_1, filenames_2)

# Convert starting and ending years to numeric for calculations
starting_year <- as.numeric(starting_year)
ending_year <- as.numeric(ending_year)

# Function to extract climate parameters for a given year
get_climate_parameters <- function(year) {
  # Define layers for the growing season (April to November)
  layer.start <- 4 + ((year - starting_year) * 12)  # Start in April
  layer.end <- layer.start + 7  # End in November
  
  # Extract humidity data for the growing season
  parameter_hurs <- subset(filenames_total, subset = (filenames_total %like% "hurs"))
  parameter_hurs <- stack(parameter_hurs)
  parameter_hurs <- parameter_hurs[[layer.start:layer.end]]
  names(parameter_hurs) <- paste("hurs", 1:8, sep = ".")
  
  # Extract precipitation data for the growing season
  parameter_pr <- subset(filenames_total, subset = (filenames_total %like% "pr"))
  parameter_pr <- stack(parameter_pr)
  parameter_pr <- parameter_pr[[layer.start:layer.end]]
  names(parameter_pr) <- paste("pr", 1:8, sep = ".")
  
  # Extract solar radiation data for the growing season
  parameter_rsds <- subset(filenames_total, subset = (filenames_total %like% "rsds"))
  parameter_rsds <- stack(parameter_rsds)
  parameter_rsds <- parameter_rsds[[layer.start:layer.end]]
  names(parameter_rsds) <- paste("rsds", 1:8, sep = ".")
  
  # Extract maximum temperature data for the growing season
  parameter_tasmax <- subset(filenames_total, subset = (filenames_total %like% "tasmax"))
  parameter_tasmax <- stack(parameter_tasmax)
  parameter_tasmax <- parameter_tasmax[[layer.start:layer.end]]
  names(parameter_tasmax) <- paste("tasmax", 1:8, sep = ".")
  
  # Extract minimum temperature data for the growing season
  parameter_tasmin <- subset(filenames_total, subset = (filenames_total %like% "tasmin"))
  parameter_tasmin <- stack(parameter_tasmin)
  parameter_tasmin <- parameter_tasmin[[layer.start:layer.end]]
  names(parameter_tasmin) <- paste("tasmin", 1:8, sep = ".")
  
  # Combine all climate parameters into a single raster stack
  df <- stack(parameter_hurs, parameter_pr, parameter_rsds, parameter_tasmax, parameter_tasmin)
  return(df)
}

# Define input years for projections
inputs <- data.frame(year = starting_year:ending_year)

# Apply the Random Forest model to make yield projections for each year
mlply(.data = inputs, .progress = "text", .fun = function(year) {
  # Extract climate parameters for the given year
  newExpl <- get_climate_parameters(year)
  
  # Use the Random Forest model to predict sorghum yield
  proj <- predict(newExpl, Final.Northen_Ha_CropGrids_Erra5_RH_P_zero_yield_changed_column_names$finalModel, 
                  type = 'response', fun = function(model, ...) {predict(model, ...)$predictions})
  
  # Save the projection as a netCDF file
  file.name <- paste("D:/sorghum_modeling/Projection/", climate_model, "/", Climate_scenario, "/", time_period, "/",
                     climate_model_2, "_", Climate_scenario, "_projection_", year, sep = "")
  writeRaster(proj, file.name, format = "CDF", overwrite = TRUE)
})
}






