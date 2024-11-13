NASA ATP Challenge:


# Overview

This is a documentation of my progress on the NASA ATP Challenge. The biggest issue in this challenge is the data: in its current form it’s disorganized and unusable. The prediction is the FUSER runways\_data\_set.csv, which has the following format:

|                                       |                               |                         |                                 |                           |                |
| ------------------------------------- | ----------------------------: | ----------------------- | ------------------------------- | ------------------------- | -------------: |
| gufi                                  | arrival\_runway\_actual\_time | arrival\_runway\_actual | departure\_runway\_actual\_time | departure\_runway\_actual | relevant\_date |
| AAL2079.DFW\.MEM.220830.2235.0126.TFM |                   9/1/22 0:14 | 36C                     |                                 |                           |    9/1/22 0:14 |
| SWA727.ATL.MEM.220831.0005.0179.TFM   |                   9/1/22 1:24 | 36L                     |                                 |                           |    9/1/22 1:24 |
| UAL2000.IAH.MEM.220830.2316.0073.TFM  |                   9/1/22 1:28 | 36L                     |                                 |                           |    9/1/22 1:28 |

Our model should work as follows: given an airport ABC, and a timestamp T, what is the expected number of arriving flights (throughput) in the next 3 hours with a resolution of 15-minutes time buckets.

The ‘input’ and ‘output’ to the model looks like this:

|                        |       |
| ---------------------- | ----- |
| ID                     | Value |
| KDEN\_220925\_0100\_15 | 99    |
| KDEN\_220925\_0100\_30 | 99    |
| KDEN\_220925\_0100\_45 | 99    |

So yeah, not that informative. 

The first step is to get a numerical y value. To do this, we’ll create a script that runs through the .csv’s and put them in sets of AIRPORT, DATE, PRED\_TIME, BUCKET. We can then expand the input data further into METAR, TAF, CWAM, and FUSAR data. Being able to comb through the data to get more input data will be the hardest part. Additionally, I believe we will have to combine the data in order to train the model. This could pose a challenge as the complete dataset is >200GB, which is wayyyyy too much for any machine to learn. 

Here’s some tips i found at this [post](https://datascience.stackexchange.com/questions/13901/machine-learning-best-practices-for-big-dataset) and this [paper](https://www.cs.columbia.edu/~vondrick/largetrain.pdf) that goes into large dataset strategies. 

So the second step will be to build scripts to convert the METAR, TAF, CWAM, and FUSAR data to usable data, then split the dataset into usable chunks.


# Models

I believe the best strategy for prediction is a 2 step process:

Lets say we’re given AIRPORT|DATE|TIME, and we want to predict 12 timesteps into the future at timestep=15min:

- We predict weather/other factors 12 timesteps into the future. 

- We then input those predictions into a second model that takes as input those factors and produces a predicted # arrivals

Thus we’ll need 2 sets of models: 

- Forecasting Models

  - Models to forecast weather, departures, predicted arrivals, etc

- Regression Model:

  - Takes in output of Forecasting Models to make prediction for each timestep. 

Training Data

First, we write a script that will comb through the FUSAR runways\_data\_set.csv data and produce as output a table of form:

|         |      |                         |                     |
| ------- | ---- | ----------------------- | ------------------- |
| AIRPORT | DATE | TIME (15 min intervals) | # arrivals (target) |
